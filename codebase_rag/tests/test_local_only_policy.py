"""Deployment policy: reject cloud/legacy paths before model or network use."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from pydantic_ai.exceptions import UnexpectedModelBehavior

import pytest

from codebase_rag.config import AppConfig, ModelConfig, settings
from codebase_rag.providers.base import get_provider_from_config
from codebase_rag.services.llm import CypherGenerator, create_rag_orchestrator
from codebase_rag.exceptions import LLMGenerationError

APPROVED = dict(CGR_LOCAL_ONLY=True, CYPHER_PROVIDER='litellm_proxy',
    CYPHER_MODEL='rag-query-gen', CYPHER_ENDPOINT='http://127.0.0.1:18194/v1',
    EMBEDDING_PROVIDER='openai', OPENAI_EMBEDDING_BASE_URL='http://127.0.0.1:18190/v1',
    OPENAI_EMBEDDING_MODEL='embedding', OPENAI_EMBEDDING_DIMENSIONS=1024,
    OPENAI_EMBEDDING_QUERY_PREFIX='', OPENAI_EMBEDDING_DOCUMENT_PREFIX='',
    MEMGRAPH_HOST='localhost', VECTOR_STORE_BACKEND='qdrant', QDRANT_URL='http://127.0.0.1:6333')


@pytest.mark.parametrize('field,value', [
    ('CYPHER_PROVIDER', 'codex'), ('CYPHER_MODEL', 'cloud-model'),
    ('CYPHER_ENDPOINT', 'https://cloud.invalid/v1'),
    ('OPENAI_EMBEDDING_BASE_URL', 'http://192.168.1.100:11434/v1'),
    ('OPENAI_EMBEDDING_MODEL', 'wrong'), ('OPENAI_EMBEDDING_DIMENSIONS', 768),
    ('OPENAI_EMBEDDING_DOCUMENT_PREFIX', 'wrong'),
    ('MEMGRAPH_HOST', 'cloud.invalid'), ('QDRANT_URL', 'https://cloud.invalid'),
])
def test_changed_endpoints_fail_at_configuration_load(field, value):
    AppConfig(_env_file=None, **{AppConfig.model_fields[k].validation_alias or k: v for k, v in APPROVED.items()})
    with pytest.raises(ValueError, match='local-only'):
        AppConfig(_env_file=None, **{AppConfig.model_fields[k].validation_alias or k: v for k, v in (APPROVED | {field: value}).items()})


@pytest.mark.parametrize('provider,model,endpoint', [
    ('codex', 'gpt-5.6-luna', None), ('codex', 'gpt-5.3-codex-spark', None),
    ('openai', 'cloud-model', 'https://api.openai.com/v1'),
    ('litellm_proxy', 'rag-query-gen', 'https://cloud.invalid/v1'),
    ('litellm_proxy', 'other-model', 'http://127.0.0.1:18194/v1'),
])
def test_provider_overrides_fail_before_construction(monkeypatch, provider, model, endpoint):
    monkeypatch.setattr(settings, 'CGR_LOCAL_ONLY', True)
    with pytest.raises(ValueError, match='local-only'):
        get_provider_from_config(ModelConfig(provider, model, endpoint=endpoint))


def test_legacy_agent_never_constructs_model(monkeypatch):
    monkeypatch.setattr(settings, 'CGR_LOCAL_ONLY', True)
    factory = Mock(side_effect=AssertionError('Model must not be constructed'))
    monkeypatch.setattr('codebase_rag.services.llm._create_provider_model', factory)
    with pytest.raises(LLMGenerationError, match='ask_code'):
        create_rag_orchestrator([])
    factory.assert_not_called()


async def test_invalid_intent_gets_at_most_two_attempts(monkeypatch):
    monkeypatch.setattr(settings, 'CGR_LOCAL_ONLY', True)
    gen = CypherGenerator.__new__(CypherGenerator)
    gen.structured = True
    gen.projects = ['chosen-project']
    gen.agent = SimpleNamespace(run=AsyncMock(side_effect=UnexpectedModelBehavior('not json')))
    with pytest.raises(LLMGenerationError, match='after 2 attempts'):
        await gen.generate('Find run')
    assert gen.agent.run.await_count == 2


def test_embedding_client_ignores_proxy_environment(monkeypatch):
    from codebase_rag.embedder import _openai_client
    for field, value in APPROVED.items():
        monkeypatch.setattr(settings, field, value)
    monkeypatch.setenv('HTTP_PROXY', 'http://cloud.invalid:8080')
    monkeypatch.setenv('HTTPS_PROXY', 'http://cloud.invalid:8080')
    with _openai_client() as client:
        assert not client.trust_env and not client.follow_redirects
    monkeypatch.setattr(settings, 'OPENAI_EMBEDDING_BASE_URL', 'https://cloud.invalid/v1')
    with pytest.raises(ValueError, match='local-only'):
        _openai_client()


async def test_query_client_ignores_proxies_and_redirects(monkeypatch):
    from codebase_rag.providers.litellm import LiteLLMProvider
    monkeypatch.setattr(settings, 'CGR_LOCAL_ONLY', True)
    health = Mock()
    monkeypatch.setattr(LiteLLMProvider, 'validate_config', health)
    with pytest.raises(ValueError):
        LiteLLMProvider(endpoint='https://cloud.invalid/v1').create_model('rag-query-gen')
    health.assert_not_called()
    model = LiteLLMProvider(api_key='test', endpoint='http://127.0.0.1:18194/v1').create_model('rag-query-gen')
    client = model.client._client
    assert not client.trust_env and not client.follow_redirects
    await client.aclose()


def test_qdrant_ignores_proxy_environment(monkeypatch):
    import httpx
    from codebase_rag import vector_store
    from qdrant_client import QdrantClient
    for field, value in APPROVED.items():
        monkeypatch.setattr(settings, field, value)
    monkeypatch.setenv('HTTP_PROXY', 'http://cloud.invalid:8080')
    monkeypatch.setenv('HTTPS_PROXY', 'http://cloud.invalid:8080')
    monkeypatch.setenv('NO_PROXY', '')
    monkeypatch.setattr(vector_store, '_CLIENT', None)
    monkeypatch.setattr(vector_store, '_CLIENT_BACKEND', None)
    # The version probe uses a separate proxy-aware client and must never run.
    probe = Mock(side_effect=AssertionError('Compatibility probe must be disabled'))
    monkeypatch.setattr('qdrant_client.qdrant_remote.get_server_version', probe)
    monkeypatch.setattr(QdrantClient, 'collection_exists', lambda *a, **kw: True)
    client = vector_store.get_qdrant_client()
    try:
        transport = client._client.openapi_client.client._client
        probe.assert_not_called()
        assert not transport.trust_env and not transport.follow_redirects
        assert transport._transport_for_url(httpx.URL(settings.QDRANT_URL)) is transport._transport
    finally:
        client.close()


def test_disabled_orchestrator_needs_no_startup_credential(monkeypatch):
    from codebase_rag.constants import ModelRole
    monkeypatch.setattr(settings, 'CGR_LOCAL_ONLY', True)
    ModelConfig('disabled', 'use-cluster-ask-code').validate_api_key(ModelRole.ORCHESTRATOR)
    with pytest.raises(ValueError, match='API Key Missing'):
        ModelConfig('litellm_proxy', 'rag-query-gen').validate_api_key(ModelRole.CYPHER)
