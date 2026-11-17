from pydantic import ValidationError
import pytest

from codebase_rag.query_intent import QueryIntent, compile_query


def test_scope_and_symbol_are_data_for_all_operations():
    hostile = "same_name' RETURN 1; MATCH (x) DETACH DELETE x //"
    for op in ['find_symbol','find_callers','find_callees','class_methods','count_symbols']:
        intent=QueryIntent(operation=op,symbol='' if op=='count_symbols' else hostile)
        bound=compile_query(intent,'chosen-project')
        assert bound.parameters['project_prefix']=='chosen-project.'
        assert 'chosen-project' not in bound.query and hostile not in bound.query
        assert 'n.qualified_name STARTS WITH $project_prefix' in bound.query
        if op in {'find_callers','find_callees','class_methods'}:
            assert 'target.qualified_name STARTS WITH $project_prefix' in bound.query
        if op=='find_callers':assert '(n)-[:CALLS]->(target)' in bound.query
        if op=='find_callees':assert '(target)-[:CALLS]->(n)' in bound.query
    for bad in [{'operation':'delete_all'}, {'operation':'find_symbol','symbol':'x','project':'other'}]:
        with pytest.raises(ValidationError):QueryIntent.model_validate(bad)
    with pytest.raises(ValueError):compile_query(QueryIntent(operation='unsupported'),'chosen-project')
    with pytest.raises(ValueError):compile_query(QueryIntent(operation='find_symbol'),'chosen-project')


@pytest.mark.parametrize('operation', ['find_symbol', 'find_callers', 'find_callees', 'class_methods'])
def test_multiple_symbols_are_bound_and_attributed(operation):
    symbols = ['first', "second' RETURN 1 //"]
    query = compile_query(QueryIntent(operation=operation, symbol=symbols), 'chosen-project')
    assert query.parameters == {'project_prefix': 'chosen-project.', 'symbols': symbols}
    assert query.query.startswith('UNWIND $symbols AS requested_symbol MATCH ')
    assert 'requested_symbol AS requested_symbol' in query.query
    assert 'n.qualified_name STARTS WITH $project_prefix' in query.query
    assert symbols[1] not in query.query
    if operation in {'find_callers', 'find_callees', 'class_methods'}:
        assert 'target.qualified_name STARTS WITH $project_prefix' in query.query
    if operation == 'find_callers':
        assert '(n)-[:CALLS]->(target)' in query.query
    if operation == 'find_callees':
        assert '(target)-[:CALLS]->(n)' in query.query


def test_symbol_batch_limits_and_empty_names():
    compile_query(QueryIntent(operation='find_symbol', symbol=['x'] * 8), 'demo')
    for symbols in [[], ['x'] * 9, ['x' * 513]]:
        with pytest.raises(ValidationError):
            QueryIntent(operation='find_symbol', symbol=symbols)
    for symbols in [['x', ' '], [''], ' ']:
        with pytest.raises(ValueError):
            compile_query(QueryIntent(operation='find_symbol', symbol=symbols), 'demo')
    with pytest.raises(ValueError):
        compile_query(QueryIntent(operation='count_symbols', symbol=['x']), 'demo')
