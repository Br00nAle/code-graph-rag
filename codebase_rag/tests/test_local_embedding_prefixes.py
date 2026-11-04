from unittest.mock import MagicMock, patch

from codebase_rag import constants as cs
from codebase_rag.config import settings
from codebase_rag.embedder import clear_embedding_cache, embed_code, embed_code_batch
from codebase_rag.tools.semantic_search import semantic_code_search


def test_local_embedding_roles_have_distinct_inputs_and_cache(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", cs.EmbeddingProvider.OPENAI)
    monkeypatch.setattr(settings, "OPENAI_EMBEDDING_QUERY_PREFIX", "search_query: ")
    monkeypatch.setattr(
        settings, "OPENAI_EMBEDDING_DOCUMENT_PREFIX", "search_document: "
    )
    clear_embedding_cache()
    try:
        with patch(
            "codebase_rag.embedder._openai_embed_batch", return_value=[[1.0]]
        ) as embed:
            embed_code_batch(["unique local prefix test"])
            embed.assert_called_once_with(["search_document: unique local prefix test"])
            embed_code("search_query: unique local prefix test")
            assert embed.call_count == 2
            embed_code_batch(["unique local prefix test"])
            assert embed.call_count == 2
        with (
            patch(
                "codebase_rag.tools.semantic_search.has_semantic_dependencies",
                return_value=True,
            ),
            patch("codebase_rag.embedder.embed_code", return_value=[1.0]) as query,
            patch(
                "codebase_rag.vector_store.search_embeddings", return_value=[]
            ) as search,
        ):
            semantic_code_search(MagicMock(), "find callers", project="fixture")
            query.assert_called_once_with("search_query: find callers")
            search.assert_called_once_with([1.0], top_k=5, project="fixture")
    finally:
        clear_embedding_cache()
