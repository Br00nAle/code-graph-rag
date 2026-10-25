"""Exercise the installed SDK's real serialization and retry behavior, without inference."""
import json
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import pytest

from codebase_rag.config import ModelConfig
from codebase_rag.exceptions import LLMGenerationError
from codebase_rag.services.llm import CypherGenerator


@pytest.mark.asyncio
async def test_cluster_query_wire_budget_and_no_transport_retry():
    for status in (200, 503):
        bodies = []
        def handle(request):
            bodies.append(json.loads(request.content))
            if status == 503:
                return httpx.Response(503,json={'error':{'message':'bounded failure'}})
            return httpx.Response(200,json={'id':'local-test','object':'chat.completion','created':1,'model':'rag-query-gen',
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant',
                 'content':'{"operation":"find_symbol","symbol":"test_name","kind":"callable"}'}}],
                'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            sdk=AsyncOpenAI(api_key='test',base_url='http://127.0.0.1:18194/v1',http_client=http)
            model=OpenAIChatModel('rag-query-gen',provider=OpenAIProvider(openai_client=sdk))
            config=ModelConfig(provider='litellm_proxy',model_id='rag-query-gen')
            with patch('codebase_rag.services.llm._create_provider_model',return_value=model), \
                 patch('codebase_rag.services.llm.settings') as settings:
                settings.active_cypher_config=config
                generator=CypherGenerator(active_projects=['test'])
                if status==200:
                    result=await generator.generate('Find functions')
                    assert result.parameters['project_prefix']=='test.'
                    assert result.parameters['symbol']=='test_name'
                else:
                    with pytest.raises(LLMGenerationError):
                        await generator.generate('Find functions')
            assert len(bodies)==1
            assert bodies[0]['max_tokens']==512
            assert 'max_completion_tokens' not in bodies[0]
