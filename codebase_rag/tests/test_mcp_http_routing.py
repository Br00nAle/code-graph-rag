from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from codebase_rag.mcp import server as srv


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", ["/", "/mcp", "/mcp/"])
async def test_http_endpoint_routes_without_redirects(configured):
    endpoint = AsyncMock()

    async def respond(scope, receive, send):
        assert scope["path"] == "/" and scope["raw_path"] == b"/"
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    endpoint.side_effect = respond
    manager = MagicMock(handle_request=endpoint)

    async def exercise():
        app = config.call_args.args[0]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            paths = ["/"] if configured == "/" else ["/mcp", "/mcp/"]
            for path in paths:
                assert (await client.post(path)).status_code == 204
            assert (await client.post("/other")).status_code == 404
        assert endpoint.await_count == len(paths)

    with (
        patch.object(srv.settings, "MCP_HTTP_ENDPOINT_PATH", configured),
        patch.object(srv.settings, "MCP_HTTP_AUTH_TOKEN", None),
        patch.object(srv, "create_server", return_value=(MagicMock(), MagicMock())),
        patch(
            "mcp.server.streamable_http_manager.StreamableHTTPSessionManager",
            return_value=manager,
        ),
        patch("uvicorn.Config") as config,
        patch("uvicorn.Server") as server,
    ):
        server.return_value.serve = AsyncMock(side_effect=exercise)
        await srv.serve_http(host="127.0.0.1", port=8000)
