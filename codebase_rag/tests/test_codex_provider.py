import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic_ai import Agent

from codebase_rag.config import ModelConfig
from codebase_rag.providers.base import get_provider_from_config


async def test_codex_returns_tool_calls_to_cgr(monkeypatch):
    monkeypatch.setattr(
        "codebase_rag.providers.base.shutil.which", lambda _: "/bin/codex"
    )
    prompts = []
    replies = iter(
        [
            {
                "text": "",
                "tool_calls": [{"name": "double", "arguments": '{"value":21}'}],
            },
            {"text": "42", "tool_calls": []},
        ]
    )

    async def spawn(*command, **kwargs):
        assert command[command.index("--model") + 1] == "gpt-5.3-codex-spark"
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert "--ignore-user-config" in command and "shell_tool" in command
        assert any(arg.startswith("developer_instructions=") for arg in command)
        Path(command[command.index("--output-last-message") + 1]).write_text(
            json.dumps(next(replies))
        )

        async def communicate(prompt):
            prompts.append(prompt.decode())
            return None, b""

        return Mock(returncode=0, communicate=communicate)

    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
    config = ModelConfig("codex", "gpt-5.3-codex-spark")
    config.validate_api_key()
    model = get_provider_from_config(config).create_model(config.model_id)
    called = []

    def double(value: int) -> int:
        called.append(value)
        return value * 2

    result = await Agent(model, tools=[double]).run("Double 21")
    assert result.output == "42" and called == [21]
    assert '"tool-return"' in prompts[1] and '"content": 42' in prompts[1]


@pytest.mark.parametrize(
    "failure", ["exit", "stdout_exit", "timeout", "invalid_arguments"]
)
async def test_codex_failure_does_not_execute_tools(monkeypatch, failure):
    monkeypatch.setattr(
        "codebase_rag.providers.base.shutil.which", lambda _: "/bin/codex"
    )
    process = Mock(returncode=1 if failure in ("exit", "stdout_exit") else None)
    process.wait = AsyncMock()
    process.communicate = AsyncMock(return_value=(None, b"login failed"))
    if failure == "stdout_exit":
        process.communicate.return_value = (
            b'{"type":"error","message":"model is not supported"}',
            b"",
        )
    if failure == "timeout":
        process.communicate.side_effect = TimeoutError
    if failure == "invalid_arguments":
        process.returncode = 0

    async def spawn(*command, **kwargs):
        if failure == "invalid_arguments":
            Path(command[command.index("--output-last-message") + 1]).write_text(
                json.dumps(
                    {"text": "", "tool_calls": [{"name": "double", "arguments": "[]"}]}
                )
            )
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
    config = ModelConfig("codex", "gpt-5.3-codex-spark")
    model = get_provider_from_config(config).create_model(config.model_id)
    tool = Mock(return_value=42)

    def double(value: int) -> int:
        return tool(value)

    error = {
        "exit": RuntimeError,
        "stdout_exit": RuntimeError,
        "timeout": TimeoutError,
        "invalid_arguments": ValueError,
    }[failure]
    with pytest.raises(error) as caught:
        await Agent(model, tools=[double]).run("Double 21")
    if failure == "stdout_exit":
        assert "model is not supported" in str(caught.value)
    tool.assert_not_called()
    if failure == "timeout":
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()
