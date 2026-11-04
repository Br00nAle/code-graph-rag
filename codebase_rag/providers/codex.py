"""Use the authenticated Codex CLI as CGR's model, with CGR executing tools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


class _ToolCall(BaseModel, extra="forbid"):
    name: str
    arguments: str


class _Reply(BaseModel, extra="forbid"):
    text: str
    tool_calls: list[_ToolCall]


async def _request(
    model_id: str, messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_json_schema,
        }
        for tool in [*info.function_tools, *info.output_tools]
    ]
    instructions = (
        "You are the JSON model transport for a separate CGR agent. Continue the "
        "conversation in the request JSON, following its embedded system prompts "
        "and instructions. Other coding-worker and AGENTS.md workflows do not "
        "apply here. Do not use native Codex tools or inspect files. "
        "The tools listed in the request JSON ARE available through CGR, even "
        "though they are not native Codex tools. Request them in your final JSON "
        "response, for example: "
        '{"text":"","tool_calls":[{"name":"tool_name",'
        '"arguments":"{\\"parameter\\":\\"value\\"}"}]}. '
        "Use the exact listed tool name and a JSON object encoded as a string "
        "for arguments. CGR executes these requests and supplies tool results "
        "in the next conversation. Never invent results or claim these tools "
        "are unavailable because they are absent from Codex's native tools. "
        "When no more tools are needed, answer in text with tool_calls empty."
    )
    prompt = json.dumps(
        {
            "instructions": info.instructions,
            "allow_text_output": info.allow_text_output,
            "tools": tools,
            "messages": json.loads(ModelMessagesTypeAdapter.dump_json(messages)),
        }
    )
    # ponytail: one CLI process per model turn; use the app server if startup cost matters.
    with TemporaryDirectory(prefix="cgr-codex-") as directory:
        output = Path(directory) / "reply.json"
        schema = Path(directory) / "schema.json"
        schema.write_text(json.dumps(_Reply.model_json_schema()))
        command = [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--json",
            "--model",
            model_id,
            "-c",
            'model_reasoning_effort="low"',
            "-c",
            'web_search="disabled"',
            "-c",
            f"developer_instructions={json.dumps(instructions)}",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--cd",
            directory,
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(output),
        ]
        for feature in (
            "shell_tool",
            "multi_agent",
            "apps",
            "plugins",
            "image_generation",
            "browser_use",
            "computer_use",
            "view_image",
        ):
            command.extend(("--disable", feature))
        process = await asyncio.create_subprocess_exec(
            *command,
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode()), 120
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(
                f"Codex CLI failed (exit {process.returncode}): "
                f"{(stderr + (stdout or b'')).decode(errors='replace')[-4000:]}"
            )
        reply = _Reply.model_validate_json(output.read_text())

    parts: list[TextPart | ToolCallPart] = []
    if reply.text:
        parts.append(TextPart(reply.text))
    for call in reply.tool_calls:
        arguments = json.loads(call.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Codex tool arguments must be a JSON object")
        parts.append(ToolCallPart(call.name, arguments))
    if not parts:
        raise ValueError("Codex returned an empty response")
    return ModelResponse(parts=parts, model_name=model_id)


def create_codex_model(model_id: str) -> FunctionModel:
    async def request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return await _request(model_id, messages, info)

    return FunctionModel(request, model_name=model_id)
