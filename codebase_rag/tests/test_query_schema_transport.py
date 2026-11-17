"""Exercise the real PydanticAI/OpenAI request path without inference or network."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from codebase_rag.config import ModelConfig
from codebase_rag.exceptions import LLMGenerationError
from codebase_rag.services.llm import CypherGenerator


def generator(monkeypatch, respond):
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
    model = OpenAIChatModel('rag-query-gen', provider=OpenAIProvider(openai_client=AsyncOpenAI(
        api_key='test', base_url='http://127.0.0.1:18194/v1', http_client=client)))
    monkeypatch.setattr('codebase_rag.services.llm._create_provider_model', lambda config: model)
    monkeypatch.setattr('codebase_rag.services.llm.settings', SimpleNamespace(
        active_cypher_config=ModelConfig('litellm_proxy', 'rag-query-gen'), AGENT_RETRIES=3))
    return CypherGenerator(active_projects=['demo']), client


def reply(content, status=200):
    return httpx.Response(status, json={'id':'test', 'object':'chat.completion', 'created':0,
        'model':'rag-query-gen', 'choices':[{'index':0, 'finish_reason':'stop',
        'message':{'role':'assistant', 'content':content}}]})


@pytest.mark.parametrize('first', ['not json', '{"operation":"find_symbol","symbol":"x","kind":"callable"} extra',
    '{"operation":"find_symbol","symbol":"","kind":"callable"}'])
async def test_native_schema_and_one_correction(monkeypatch, first):
    calls = []
    def respond(request):
        calls.append(json.loads(request.content))
        return reply(first if len(calls) == 1 else json.dumps({
            'operation':'find_callers', 'symbol':['first', 'second'], 'kind':'callable'}))
    gen, client = generator(monkeypatch, respond)
    try:
        result = await gen.generate('Who calls first and second?')
    finally:
        await client.aclose()
    assert len(calls) == 2
    for call in calls:
        fmt = call['response_format']
        assert fmt['type'] == 'json_schema' and fmt['json_schema']['strict']
        assert fmt['json_schema']['name'] == 'query_intent'
        schema = fmt['json_schema']['schema']
        assert schema['additionalProperties'] is False
        assert set(schema['required']) == {'operation', 'symbol', 'kind'}
        assert schema['properties']['symbol']['anyOf'][1]['maxItems'] == 8
        assert call['max_tokens'] == 512
        assert not call.get('tools') and not call.get('stream')
        assert all(m['role'] in {'system', 'user'} for m in call['messages'])
    assert result.parameters['symbols'] == ['first', 'second']


@pytest.mark.parametrize('content,status,attempts', [('not json',200,2), ('unavailable',503,1),
    ('{"operation":"unsupported","symbol":"","kind":"callable"}',200,1)])
async def test_failure_limits_and_no_service_retry(monkeypatch, content, status, attempts):
    calls = []
    def respond(request):
        calls.append(request)
        return reply(content, status)
    gen, client = generator(monkeypatch, respond)
    try:
        with pytest.raises(LLMGenerationError):
            await gen.generate('Find x')
    finally:
        await client.aclose()
    assert len(calls) == attempts


async def test_concurrent_requests_keep_their_symbols_and_limits(monkeypatch):
    calls = []
    async def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        symbol = body['messages'][-1]['content'].split()[-1]
        await asyncio.sleep(0)
        return reply(json.dumps({'operation':'find_symbol', 'symbol':symbol, 'kind':'callable'}))
    gen, client = generator(monkeypatch, respond)
    try:
        results = await asyncio.gather(*(gen.generate('Find ' + s) for s in ['one', 'two', 'three']))
        for question in ['', 'x' * 4001]:
            with pytest.raises(LLMGenerationError):
                await gen.generate(question)
        gen.projects = ['demo', 'other']
        with pytest.raises(LLMGenerationError):
            await gen.generate('Find four')
    finally:
        await client.aclose()
    assert len(calls) == 3
    assert [r.parameters['symbol'] for r in results] == ['one', 'two', 'three']
    assert all(r.parameters['project_prefix'] == 'demo.' for r in results)
