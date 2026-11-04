"""Run a CGR command with Arch-only inference and a separate embedding collection."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure():
    key = (ROOT / ".tmp/local-cgr/key").read_text().strip()
    os.environ.update(
        {
            "MEMGRAPH_HOST": "127.0.0.1",
            "MEMGRAPH_PORT": "7687",
            "QDRANT_URL": "http://127.0.0.1:6333",
            "ORCHESTRATOR_PROVIDER": "litellm_proxy",
            "CYPHER_PROVIDER": "litellm_proxy",
            "ORCHESTRATOR_MODEL": "cgr-chat",
            "CYPHER_MODEL": "cgr-chat",
            "ORCHESTRATOR_ENDPOINT": "http://127.0.0.1:18185/v1",
            "CYPHER_ENDPOINT": "http://127.0.0.1:18185/v1",
            "ORCHESTRATOR_API_KEY": key,
            "CYPHER_API_KEY": key,
            "CGR_EMBEDDING_PROVIDER": "openai",
            "OPENAI_EMBEDDING_BASE_URL": "http://127.0.0.1:18186/v1",
            "OPENAI_EMBEDDING_MODEL": "cgr-embed",
            "OPENAI_EMBEDDING_API_KEY": key,
            "OPENAI_EMBEDDING_DIMENSIONS": "768",
            "OPENAI_EMBEDDING_BATCH_SIZE": "1",
            "OPENAI_EMBEDDING_MAX_INPUT_CHARS": "1600",
            "QUERY_RESULT_MAX_TOKENS": "3000",
            "OPENAI_EMBEDDING_QUERY_PREFIX": "search_query: ",
            "OPENAI_EMBEDDING_DOCUMENT_PREFIX": "search_document: ",
            "QDRANT_COLLECTION_NAME": "code_embeddings_nomic_v15_q4_768",
            "QDRANT_VECTOR_DIM": "768",
            "QDRANT_DB_PATH": str(ROOT / ".tmp/local-cgr/cache"),
            "ORCHESTRATOR_OUTPUT_RETRIES": "2",
            "AGENT_RETRIES": "2",
        }
    )


if __name__ == "__main__":
    configure()
    os.chdir(ROOT)
    os.execv(str(ROOT / ".venv/bin/cgr"), ["cgr", *sys.argv[1:]])
