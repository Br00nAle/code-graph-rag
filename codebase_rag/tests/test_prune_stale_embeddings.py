from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from codebase_rag import vector_store as vs


@pytest.mark.parametrize("current", [{1: "app.live", 3: "app.reused"}, {}])
def test_prune_is_scoped_paginated_and_checks_identity(monkeypatch, current):
    def point(node_id, name):
        return SimpleNamespace(
            id=node_id, payload={"node_id": node_id, "qualified_name": name}
        )

    client = Mock()
    client.scroll.side_effect = [
        ([point(1, "app.live"), point(2, "app.deleted")], 3),
        ([point(3, "app.old"), point(4, "app-other.live")], None),
    ]
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(vs.settings, "VECTOR_STORE_BACKEND", "qdrant")
    assert vs.prune_stale_qdrant_embeddings("app", current) == (2 if current else 3)
    assert [c.kwargs["points_selector"] for c in client.delete.call_args_list] == [
        [2] if current else [1, 2],
        [3],
    ]
    assert client.scroll.call_args.kwargs["offset"] == 3


def test_prune_failure_propagates(monkeypatch):
    client = Mock()
    client.scroll.side_effect = RuntimeError("offline")
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(vs.settings, "VECTOR_STORE_BACKEND", "qdrant")
    with pytest.raises(RuntimeError, match="offline"):
        vs.prune_stale_qdrant_embeddings("app", {})


@pytest.mark.parametrize("failure", [False, True])
def test_embedding_pass_prunes_even_when_project_has_no_functions(monkeypatch, failure):
    from codebase_rag.graph_updater import GraphUpdater
    from codebase_rag.services.graph_service import MemgraphIngestor

    updater = GraphUpdater.__new__(GraphUpdater)
    updater.skip_embeddings = False
    updater.project_name = "app"
    updater.ingestor = Mock(spec=MemgraphIngestor)
    updater.ingestor.fetch_all.return_value = []
    prune = Mock(side_effect=RuntimeError("offline") if failure else None)
    close = Mock()
    monkeypatch.setattr(vs, "prune_stale_qdrant_embeddings", prune)
    monkeypatch.setattr(vs, "close_qdrant_client", close)
    monkeypatch.setattr(
        "codebase_rag.graph_updater.has_semantic_dependencies", lambda: True
    )
    updater._generate_semantic_embeddings()
    prune.assert_called_once_with("app", {})
    close.assert_called_once_with()


@pytest.mark.parametrize("failure", [None, "embedding", "storage"])
def test_prune_follows_successful_replacement_writes(monkeypatch, failure):
    from codebase_rag.graph_updater import GraphUpdater
    from codebase_rag.services.graph_service import MemgraphIngestor

    updater = GraphUpdater.__new__(GraphUpdater)
    updater.skip_embeddings = False
    updater.project_name = "app"
    updater.ingestor = Mock(spec=MemgraphIngestor)
    updater.ingestor.fetch_all.return_value = [
        dict(
            node_id=1, qualified_name="app.live", start_line=1, end_line=2, path="a.py"
        )
    ]
    events = []

    def embed(snippets):
        if failure == "embedding":
            raise RuntimeError("provider offline")
        return [[0.1]]

    def store(points):
        events.append("store")
        return 0 if failure == "storage" else len(points)

    monkeypatch.setattr("codebase_rag.embedder.embed_code_batch", embed)
    monkeypatch.setattr("codebase_rag.embedder.get_embedding_cache", Mock())
    monkeypatch.setattr(vs, "store_embedding_batch", store)
    monkeypatch.setattr(
        vs, "prune_stale_qdrant_embeddings", lambda *a: events.append("prune")
    )
    monkeypatch.setattr(vs, "close_qdrant_client", lambda: None)
    monkeypatch.setattr(
        GraphUpdater, "_extract_source_code", lambda *a: "def live(): pass"
    )
    monkeypatch.setattr(GraphUpdater, "_reconcile_embeddings", lambda *a: None)
    monkeypatch.setattr(
        "codebase_rag.graph_updater.has_semantic_dependencies", lambda: True
    )
    updater._generate_semantic_embeddings()
    assert events == (
        [] if failure == "embedding" else ["store"] if failure else ["store", "prune"]
    )
