"""Live read-only smoke of the Arch-local MCP endpoint; requires its services."""

import asyncio
import json
from datetime import timedelta

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    # systemd Type=simple becomes active before the MCP listener is ready.
    async with httpx.AsyncClient(timeout=2) as client:
        for attempt in range(30):
            try:
                await client.get("http://127.0.0.1:8001/")
                break
            except httpx.TransportError:
                if attempt == 29:
                    raise
                await asyncio.sleep(1)
    async with streamablehttp_client(
        "http://127.0.0.1:8001/mcp",
        timeout=timedelta(seconds=600),
        sse_read_timeout=timedelta(seconds=600),
    ) as (reader, writer, _):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=600)
        ) as session:
            await session.initialize()

            async def call(name, args):
                result = await session.call_tool(name, args)
                assert not result.isError, result
                return "\n".join(
                    block.text for block in result.content if block.type == "text"
                )

            projects = json.loads(await call("list_projects", {}))
            assert "ai-cluster-blueprint" in projects["projects"]
            source = await call(
                "read_file", {"file_path": "cluster/parent_watch.py", "limit": 100}
            )
            assert "def identity(" in source and "def watch(" in source
            assert "if identity(parent) != stamp:" in source
            semantic = await call(
                "semantic_search",
                {
                    "natural_language_query": "watch parent process and stop child when parent exits",
                    "top_k": 3,
                },
            )
            assert "ai-cluster-blueprint.cluster.parent_watch." in semantic, semantic
            assert "watermelon-mania." not in semantic, semantic
            query = json.loads(
                await call(
                    "query_code_graph",
                    {
                        "natural_language_query": "Which functions directly call ai-cluster-blueprint.cluster.parent_watch.identity? Scope both caller and callee to ai-cluster-blueprint. Return their qualified names."
                    },
                )
            )
            assert query["results"], query
            assert "ai-cluster-blueprint." in query["query_used"], query
            assert any("parent_watch.watch" in str(row) for row in query["results"]), (
                query
            )
            print(  # noqa: T201 -- live smoke result
                json.dumps(
                    {
                        "projects": projects["count"],
                        "semantic": semantic,
                        "callers": query,
                        "passed": True,
                        "graph_results_require_source_verification": True,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
