"""Compile a small read-only query vocabulary; project and symbols are bound data."""
from dataclasses import dataclass
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryIntent(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    operation: Literal['find_symbol','find_callers','find_callees','class_methods','count_symbols','unsupported']
    symbol: str = Field(default='',max_length=512)
    kind: Literal['callable','function','method','class'] = 'callable'


INTENT_PROMPT = '''Translate the question into one JSON object, without Markdown or Cypher.
Allowed fields: operation, symbol, kind. Never emit a project or query field.
Operations:
- find_symbol: find a named function, method or class.
- find_callers: functions/methods that directly call the named symbol.
- find_callees: functions/methods directly called by the named symbol.
- class_methods: methods defined by a named class.
- count_symbols: count functions, methods, classes or both callable types.
- unsupported: anything else, including writes, paths, multi-hop or cross-project requests.
kind is callable (Function OR Method), function (Function only), method, or class.
Copy the symbol from the question; use a qualified or dotted name if provided.
For counts symbol is empty. For unsupported symbol is empty.
Examples:
"Who calls GameManager.start_shift?" -> {"operation":"find_callers","symbol":"GameManager.start_shift","kind":"callable"}
"What does GameManager.start_shift call?" -> {"operation":"find_callees","symbol":"GameManager.start_shift","kind":"callable"}
"How many Function nodes?" -> {"operation":"count_symbols","symbol":"","kind":"function"}
"Methods in Inference" -> {"operation":"class_methods","symbol":"Inference","kind":"class"}
Project scope is supplied by the caller and enforced by code. Do not translate project scope into the symbol.
'''


@dataclass(frozen=True)
class ScopedQuery:
    query: str
    parameters: dict[str,str]

    @property
    def display(self) -> str:
        # Diagnostic text only. Execution always uses query + bound parameters.
        return self.query + '\n// bound parameters: ' + json.dumps(self.parameters,ensure_ascii=True)


def compile_query(intent: QueryIntent, project: str) -> ScopedQuery:
    if not isinstance(project,str) or not project or project != project.strip():
        raise ValueError('One explicit project scope is required')
    if intent.operation == 'unsupported':
        raise ValueError('Unsupported graph operation; use symbol lookup, direct callers/callees, class methods or counts')
    params = {'project_prefix':project+'.'}
    kinds = {'callable':'(n:Function OR n:Method)','function':'n:Function','method':'n:Method','class':'n:Class'}
    scope = 'n.qualified_name STARTS WITH $project_prefix'
    predicate = '(n.name = $symbol OR n.qualified_name = $symbol OR n.qualified_name ENDS WITH $symbol_suffix)'
    result = 'RETURN DISTINCT n.name AS name, n.qualified_name AS qualified_name LIMIT 50;'
    if intent.operation == 'count_symbols':
        if intent.symbol:
            raise ValueError('Counts do not accept a symbol filter')
        query = f'MATCH (n) WHERE {kinds[intent.kind]} AND {scope} RETURN count(n) AS total;'
    else:
        symbol = intent.symbol.strip()
        if not symbol:
            raise ValueError('A symbol is required')
        params.update(symbol=symbol,symbol_suffix='.'+symbol)
        if intent.operation == 'find_symbol':
            query = f'MATCH (n) WHERE {kinds[intent.kind]} AND {scope} AND {predicate} {result}'
        elif intent.operation == 'class_methods':
            query = ('MATCH (target:Class)-[:DEFINES_METHOD]->(n:Method) '
                'WHERE target.qualified_name STARTS WITH $project_prefix AND '+scope+' AND '+
                predicate.replace('n.','target.')+' '+result)
        else:
            edge = '(n)-[:CALLS]->(target)' if intent.operation == 'find_callers' else '(target)-[:CALLS]->(n)'
            query = (f'MATCH {edge} WHERE (n:Function OR n:Method) AND (target:Function OR target:Method) '
                'AND target.qualified_name STARTS WITH $project_prefix AND '+scope+' AND '+
                predicate.replace('n.','target.')+' '+result)
    return ScopedQuery(query,params)
