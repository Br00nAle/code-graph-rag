from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import ValidationError
from pydantic_ai import Agent, DeferredToolRequests, Tool
from pydantic_ai.agent import AgentRetries

from .. import constants as cs
from .. import exceptions as ex
from .. import logs as ls
from ..config import ModelConfig, load_cgr_instructions, settings
from ..prompts import (
    build_cypher_system_prompt,
    build_local_cypher_system_prompt,
    build_rag_orchestrator_prompt,
)
from ..providers.base import get_provider_from_config
from ..query_intent import INTENT_PROMPT, QueryIntent, ScopedQuery, compile_query

if TYPE_CHECKING:
    from pydantic_ai.models import Model


def _create_provider_model(config: ModelConfig) -> Model:
    provider = get_provider_from_config(config)
    return provider.create_model(config.model_id)


def _clean_cypher_response(response_text: str) -> str:
    query = response_text.strip()

    if "```" in query:
        parts = query.split("```")
        if len(parts) >= 3:
            block = parts[1]
            if block.lower().startswith("cypher"):
                block = block[len("cypher") :]
            query = block.strip()
    else:
        while "**" in query:
            start = query.index("**")
            end = query.find("**", start + 2)
            if end == -1:
                break
            after = end + 2
            if after < len(query) and query[after] == ":":
                after += 1
            query = query[:start] + query[after:].lstrip()
        query = query.replace(cs.CYPHER_BACKTICK, "")
        if query.lower().startswith(cs.CYPHER_PREFIX):
            query = query[len(cs.CYPHER_PREFIX) :].strip()

    if not query.endswith(cs.CYPHER_SEMICOLON):
        query += cs.CYPHER_SEMICOLON
    return query


_COMMENT_OR_WS = r"(?:\s|//[^\n]*|/\*.*?\*/)+"


def _build_keyword_pattern(keyword: str) -> re.Pattern[str]:
    parts = keyword.split()
    if len(parts) == 1:
        return re.compile(rf"\b{re.escape(parts[0])}\b")
    joined = _COMMENT_OR_WS.join(re.escape(p) for p in parts)
    return re.compile(rf"\b{joined}\b", re.DOTALL)


_CYPHER_DANGEROUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (kw, _build_keyword_pattern(kw)) for kw in cs.CYPHER_DANGEROUS_KEYWORDS
]


_VARLEN_PATTERN = re.compile(r"\[[^\]]*?\*([^\]]*)\]")
_PROCEDURE_CALL_PATTERN = re.compile(r"\bCALL\s+([\w\.]+)", re.IGNORECASE)
# Some models incorrectly pass a list property through toLower(), e.g.
# ``ANY(d IN toLower(n.decorators) WHERE ...)``.  Memgraph accepts toLower()
# for strings only; repair this common shape before it reaches the database.
_LOWERED_LIST_ITERATOR_PATTERN = re.compile(
    r"\bANY\(\s*(?P<item>[A-Za-z_]\w*)\s+IN\s+"
    r"toLower\(\s*(?P<list>[A-Za-z_]\w*\.[A-Za-z_]\w*)\s*\)",
    re.IGNORECASE,
)


def _repair_lowered_list_iterators(query: str) -> str:
    """Repair model output that applies ``toLower`` to a list property.

    List properties must be iterated first, with case folding applied to the
    individual value.  The generated predicate remains responsible for
    comparing the item to the requested value; this repair only makes the
    iterator expression valid Cypher.
    """

    return _LOWERED_LIST_ITERATOR_PATTERN.sub(
        lambda match: f"ANY({match.group('item')} IN {match.group('list')}",
        query,
    )


def _validate_cypher_read_only(query: str) -> None:
    upper_query = query.upper()
    for keyword, pattern in _CYPHER_DANGEROUS_PATTERNS:
        if pattern.search(upper_query):
            raise ex.LLMGenerationError(
                ex.LLM_DANGEROUS_QUERY.format(keyword=keyword, query=query)
            )


def _validate_no_unbounded_paths(query: str) -> None:
    for match in _VARLEN_PATTERN.finditer(query):
        spec = match.group(1).strip()
        if not spec:
            raise ex.LLMGenerationError(ex.LLM_UNBOUNDED_PATH.format(query=query))
        if ".." in spec:
            upper = spec.split("..", 1)[1].lstrip()
            if not upper or not upper[0].isdigit():
                raise ex.LLMGenerationError(ex.LLM_UNBOUNDED_PATH.format(query=query))


def _validate_call_procedures(query: str) -> None:
    for match in _PROCEDURE_CALL_PATTERN.finditer(query):
        name = match.group(1)
        if not any(
            name.startswith(prefix) for prefix in cs.CYPHER_ALLOWED_PROCEDURE_PREFIXES
        ):
            raise ex.LLMGenerationError(
                ex.LLM_DISALLOWED_PROCEDURE.format(name=name, query=query)
            )


class CypherGenerator:
    __slots__ = ("agent", "projects", "structured")

    def __init__(self, active_projects: list[str] | None = None) -> None:
        try:
            config = settings.active_cypher_config
            llm = _create_provider_model(config)
            self.structured = config.provider == cs.Provider.LITELLM_PROXY and config.model_id == "rag-query-gen"
            self.projects = list(active_projects or [])
            if self.structured:
                llm.client.max_retries = 0
                llm.profile["openai_chat_supports_max_completion_tokens"] = False

            system_prompt = (
                build_local_cypher_system_prompt(active_projects)
                if config.provider == cs.Provider.OLLAMA
                else build_cypher_system_prompt(active_projects)
            )

            self.agent = Agent(
                model=llm,
                system_prompt=INTENT_PROMPT if self.structured else system_prompt,
                output_type=str,
                retries=0 if self.structured else settings.AGENT_RETRIES,
                model_settings={"max_tokens":512,"timeout":120} if self.structured else None,
            )
        except Exception as e:
            raise ex.LLMGenerationError(ex.LLM_INIT_CYPHER.format(error=e)) from e

    async def generate(self, natural_language_query: str) -> str | ScopedQuery:
        logger.info(ls.CYPHER_GENERATING.format(query=natural_language_query))
        try:
            if self.structured and len(self.projects) != 1:
                raise ValueError("Select exactly one project before querying")
            result = await self.agent.run(natural_language_query)
            if self.structured:
                for attempt in range(3):
                    output = result.output.strip()
                    if output.startswith("```") and output.endswith("```"):
                        output = output.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                    try:
                        intent = QueryIntent.model_validate_json(output)
                        # Compilation also validates operation-specific requirements.
                        if intent.operation == "unsupported":
                            break
                        return compile_query(intent, self.projects[0])
                    except (ValidationError, ValueError) as error:
                        if attempt == 2:
                            raise ValueError(
                                f"Intent generation failed after 3 attempts: {error}"
                            ) from error
                        logger.warning(
                            "Intent validation failed; correction retry {}/2: {}",
                            attempt + 1,
                            error,
                        )
                        result = await self.agent.run(
                            f"Original question: {natural_language_query}\n"
                            f"Previous response: {result.output}\n"
                            f"Validation error: {error}\n"
                            "Correct the response using the intent schema. Return only JSON. "
                            "kind must be callable, function, method, or class. "
                            "For unsupported requests use operation=unsupported, "
                            "symbol empty, and kind=callable. Preserve the original request; "
                            "do not substitute a different supported operation."
                        )
                return compile_query(intent, self.projects[0])
            if (
                not isinstance(result.output, str)
                or cs.CYPHER_MATCH_KEYWORD not in result.output.upper()
            ):
                raise ex.LLMGenerationError(
                    ex.LLM_INVALID_QUERY.format(output=result.output)
                )

            query = _clean_cypher_response(result.output)
            query = _repair_lowered_list_iterators(query)
            _validate_cypher_read_only(query)
            _validate_no_unbounded_paths(query)
            _validate_call_procedures(query)
            logger.info(ls.CYPHER_GENERATED.format(query=query))
            return query
        except Exception as e:
            logger.error(ls.CYPHER_ERROR.format(error=e))
            raise ex.LLMGenerationError(ex.LLM_GENERATION_FAILED.format(error=e)) from e


def create_rag_orchestrator(
    tools: list[Tool],
    project_root: Path | None = None,
    load_instructions: bool = True,
    active_projects: list[str] | None = None,
) -> tuple[Agent, str]:
    try:
        config = settings.active_orchestrator_config
        llm = _create_provider_model(config)

        project_instructions = (
            load_cgr_instructions(project_root) if load_instructions else None
        )
        system_prompt = build_rag_orchestrator_prompt(
            tools,
            project_instructions=project_instructions,
            active_projects=active_projects,
        )

        agent = Agent(
            model=llm,
            system_prompt=system_prompt,
            tools=tools,
            retries=AgentRetries(
                tools=settings.AGENT_RETRIES,
                output=settings.ORCHESTRATOR_OUTPUT_RETRIES,
            ),
            output_type=[str, DeferredToolRequests],
        )
        return agent, system_prompt
    except Exception as e:
        raise ex.LLMGenerationError(ex.LLM_INIT_ORCHESTRATOR.format(error=e)) from e
