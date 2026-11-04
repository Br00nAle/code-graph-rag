from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from codebase_rag.exceptions import LLMGenerationError
from codebase_rag.services.llm import CypherGenerator
from codebase_rag.tools.codebase_query import create_query_tool


def generator(*outputs):
    gen = CypherGenerator.__new__(CypherGenerator)
    gen.structured = True
    gen.projects = ["chosen-project"]
    gen.agent = SimpleNamespace(
        run=AsyncMock(side_effect=[SimpleNamespace(output=o) for o in outputs])
    )
    return gen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        '{"operation":"find_symbol","symbol":"run","kind":"unsupported"}',
        "not json",
        '{"operation":"find_symbol","symbol":""}',
        '{"operation":"find_symbol","symbol":"run","project":"other"}',
    ],
)
async def test_corrects_invalid_intent_with_feedback(invalid):
    gen = generator(
        invalid, '{"operation":"find_symbol","symbol":"run","kind":"callable"}'
    )
    query = await gen.generate("Find run")
    assert query.parameters == {
        "project_prefix": "chosen-project.",
        "symbol": "run",
        "symbol_suffix": ".run",
    }
    assert gen.agent.run.await_count == 2
    feedback = gen.agent.run.call_args.args[0]
    assert (
        "Validation error:" in feedback
        and invalid in feedback
        and "Find run" in feedback
    )


@pytest.mark.asyncio
async def test_accepts_fenced_json_without_retry():
    gen = generator('```json\n{"operation":"find_symbol","symbol":"run"}\n```')
    assert (await gen.generate("Find run")).parameters["symbol"] == "run"
    assert gen.agent.run.await_count == 1


@pytest.mark.asyncio
async def test_exhaustion_is_error_and_never_queries_database():
    gen = generator("bad", "bad", "bad")
    ingestor = MagicMock()
    result = await create_query_tool(ingestor, gen).function("Find run")
    assert "after 3 attempts" in result.error
    assert gen.agent.run.await_count == 3
    ingestor.fetch_all.assert_not_called()


@pytest.mark.asyncio
async def test_unsupported_request_is_not_reinterpreted():
    gen = generator(
        '{"operation":"unsupported","kind":"unsupported"}',
        '{"operation":"unsupported","kind":"callable"}',
    )
    with pytest.raises(LLMGenerationError, match="Unsupported graph operation"):
        await gen.generate("Find a file by path")
    assert gen.agent.run.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, "Intent generation failed after 3 attempts"])
async def test_mcp_protocol_marks_generation_errors(error, tmp_path):
    from unittest.mock import patch

    from mcp.types import CallToolRequest, CallToolRequestParams

    from codebase_rag.mcp import server as srv

    payload = {"query_used": "N/A", "results": [], "summary": "Empty"}
    if error:
        payload["error"] = error
    registry = MagicMock()
    registry.get_tool_handler.return_value = (AsyncMock(return_value=payload), True)
    with (
        patch.object(srv, "settings"),
        patch.object(srv, "get_project_root", return_value=tmp_path),
        patch.object(srv, "MemgraphIngestor"),
        patch.object(srv, "CypherGenerator"),
        patch.object(srv, "create_mcp_tools_registry", return_value=registry),
    ):
        server, _ = srv.create_server()
    with patch.object(
        server, "_get_cached_tool_definition", AsyncMock(return_value=None)
    ):
        response = await server.request_handlers[CallToolRequest](
            CallToolRequest(
                params=CallToolRequestParams(
                    name="query_code_graph",
                    arguments={"natural_language_query": "Find run"},
                )
            )
        )
    assert response.root.isError is bool(error)
