import json
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import langgraph_rag_backend as b
import rag_storage as storage


def upload(text, filename='notes.txt', thread='alice::one'):
    return b.ingest_file(text.encode(), thread, filename, thread.split('::')[0])


def test_multiple_files_are_searchable_without_overwriting():
    upload('Photosynthesis converts sunlight into energy.', 'biology.txt')
    upload('Database normalization reduces redundant data.', 'databases.txt')
    result = b.hybrid_context('alice::one', 'photosynthesis sunlight')
    assert result['sources'][0]['source'] == 'biology.txt'
    assert {s['source'] for s in result['sources']} == {'biology.txt', 'databases.txt'}
    assert b.thread_document_metadata('alice::one')['file_count'] == 2


def test_duplicate_and_same_name_replacement():
    upload('The exam is on Monday.')
    duplicate = upload('The exam is on Monday.')
    assert duplicate['duplicate']
    upload('The exam is on Friday.')
    result = b.hybrid_context('alice::one', 'exam date')
    assert 'Friday' in ' '.join(result['context'])
    assert 'Monday' not in ' '.join(result['context'])
    assert b.thread_document_metadata('alice::one')['file_count'] == 1


def test_failed_replacement_is_atomic(monkeypatch):
    upload('Keep the original text.')
    monkeypatch.setattr(b, '_build_indexes', Mock(side_effect=RuntimeError('embedding failure')))
    with pytest.raises(RuntimeError):
        upload('The replacement should not be committed.')
    assert 'original' in storage.documents('alice::one')[0]['chunks'][0]['page_content']
    assert 'original' in b.hybrid_context('alice::one', 'original')['context'][0]


def test_documents_are_isolated_and_restore_after_cache_loss():
    upload('Alice has confidential physics notes.')
    upload('Bob has chemistry notes.', thread='bob::one')
    b._THREAD_VS.clear()
    b._THREAD_BM25.clear()
    b._THREAD_CHUNKS.clear()
    result = b.hybrid_context('alice::one', 'notes')
    assert 'Alice' in result['context'][0]
    assert all('Bob' not in text for text in result['context'])
    assert b.retrieve_user_threads('alice') == ['alice::one']


def test_cross_user_ingestion_is_rejected():
    with pytest.raises(ValueError, match='current session'):
        b.ingest_file(b'secret', 'alice::one', 'notes.txt', 'bob')


def test_remove_and_delete_clear_persisted_state():
    upload('First source', 'first.txt')
    upload('Second source', 'second.txt')
    b.remove_thread_document('alice::one', 'first.txt', 'alice')
    assert {s['source'] for s in b.hybrid_context('alice::one', 'source')['sources']} == {'second.txt'}
    storage.append_message('alice::one', 'user', 'Hello')
    b.delete_conversation('alice::one', 'alice')
    assert not storage.messages('alice::one')
    assert not b.thread_has_document('alice::one')
    assert b.retrieve_user_threads('alice') == []


def test_rrf_does_not_merge_same_prefix_or_distinct_sources():
    docs = [Document(page_content='x' * 201 + suffix, metadata={'source': source})
            for suffix, source in [('A', 'a.txt'), ('B', 'a.txt'), ('A', 'b.txt')]]
    assert len(b._reciprocal_rank_fusion([docs, docs])) == 3


def test_pdf_chunks_have_correct_page_numbers():
    chunks = b._chunk_document('## Page 1\n\nFirst page.\n\n---\n\n## Page 2\n\nSecond page.', 'notes.pdf', 'pdf', 'hash')
    assert [(c.metadata['page'], 'First' in c.page_content) for c in chunks] == [(1, True), (2, False)]


def test_query_expansion_rejects_non_strings():
    llm = Mock()
    llm.invoke.return_value = AIMessage(content='[123, {"bad": "query"}]')
    assert b._query_transform('question', llm) == ['question']


def test_hybrid_search_keeps_working_if_one_retriever_fails(monkeypatch):
    upload('Robust retrieval survives one failed search path.')
    vs = b._THREAD_VS['alice::one']
    monkeypatch.setattr(vs, 'similarity_search', Mock(side_effect=RuntimeError('vector search unavailable')))
    result = b.hybrid_context('alice::one', 'retrieval')
    assert result['context']
    assert result['obs']['degraded']


def test_history_removes_orphan_tool_results_and_keeps_pairs():
    messages = [ToolMessage(content='orphan', tool_call_id='orphan'), HumanMessage(content='Calculate'),
                AIMessage(content='', tool_calls=[{'name': 'calculator', 'args': {}, 'id': 'ok'}]),
                ToolMessage(content='2', tool_call_id='ok'), AIMessage(content='Two'), HumanMessage(content='Next')]
    result = b._bounded_history(messages, max_chars=100)
    assert not any(isinstance(m, ToolMessage) and m.tool_call_id == 'orphan' for m in result)
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == 'ok' for m in result)


def test_graph_retrieves_before_answer_and_saves_sources(monkeypatch):
    upload('The laboratory opens at 9 AM.')
    seen = []
    class FakeModel:
        def bind_tools(self, tools):
            return self
        def invoke(self, messages, config=None):
            seen.append(messages)
            return AIMessage(content='The laboratory opens at 9 AM [1].')
    monkeypatch.setattr(b, 'get_llm', lambda user_id: FakeModel())
    bot = b.get_chatbot('alice')
    result = bot.invoke({'messages': [HumanMessage(content='When does the lab open?')]}, {'configurable': {'thread_id': 'alice::one'}})
    assert '9 AM' in seen[0][0].content
    assert 'untrusted data' in seen[0][0].content
    assert result['messages'][-1].additional_kwargs['sources'][0]['source'] == 'notes.txt'
    b.invalidate_graph('alice')
    restored = b.get_chatbot('alice').get_state({'configurable': {'thread_id': 'alice::one'}})
    assert '9 AM' in restored.values['messages'][-1].content


def test_graph_tool_roundtrip_has_valid_history(monkeypatch):
    seen = []
    class FakeModel:
        def bind_tools(self, tools):
            return self
        def invoke(self, messages, config=None):
            seen.append(messages)
            if len(seen) == 1:
                return AIMessage(content='', tool_calls=[{'name': 'calculator', 'args': {'first_num': 2, 'second_num': 3, 'operation': 'add'}, 'id': 'calc'}])
            assert isinstance(messages[-1], ToolMessage)
            return AIMessage(content='The answer is 5.')
    monkeypatch.setattr(b, 'get_llm', lambda user_id: FakeModel())
    result = b.get_chatbot('alice').invoke({'messages': [HumanMessage(content='2 + 3')]}, {'configurable': {'thread_id': 'alice::math'}, 'recursion_limit': 12})
    assert result['messages'][-1].content == 'The answer is 5.'
    assert len(seen) == 2


def test_transcript_includes_upload_only_events_and_exported_sources():
    storage.append_message('alice::one', 'assistant', 'Document ready')
    sources = [{'source': 'notes.pdf', 'page': 2, 'citation': 1, 'text': 'Source evidence'}]
    storage.append_message('alice::one', 'assistant', 'Answer [1]', sources)
    text = b.export_conversation_txt(storage.messages('alice::one'), 'Study notes')
    assert 'Document ready' in text
    assert '[1] notes.pdf - page 2' in text
    assert 'Source evidence' in text
