"""Tests use isolated SQLite databases and never load the user's credentials."""
import os
import shutil
import tempfile

_test_data = tempfile.mkdtemp(prefix='teamdino-tests-')
os.environ['RAG_DATA_DIR'] = _test_data
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ['RAG_SKIP_LEGACY_MIGRATION'] = '1'
os.environ['RAG_KEY_ENCRYPTION_KEY'] = 'test-only-encryption-key'

import pytest
import langgraph_rag_backend as backend
import rag_providers as providers
import key_vault as vault
import rag_storage as storage
from langchain_core.embeddings import Embeddings


class TestEmbeddings(Embeddings):
    def embed_documents(self, texts):
        import hashlib
        import math
        import re
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for token in re.findall(r'\w+', text.lower()):
                index = int(hashlib.sha256(token.encode()).hexdigest()[:4], 16) % 64
                vector[index] += 1
            norm = math.sqrt(sum(v * v for v in vector)) or 1
            vectors.append([v / norm for v in vector])
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    for name in (
        'OPENAI_API_KEY', 'OPENAI_API_KEYS', 'GROQ_API_KEY', 'GROQ_API_KEYS',
        'GOOGLE_API_KEY', 'GEMINI_API_KEY', 'GEMINI_API_KEYS',
        'ANTHROPIC_API_KEY', 'ANTHROPIC_API_KEYS', 'OPENROUTER_API_KEY',
        'OPENROUTER_API_KEYS', 'OX_ALPHA_API_KEY', 'OX_ALPHA_API_KEYS',
        'TAVILY_API_KEY', 'RAG_SYSTEM_KEYS_JSON', 'GROQ_TEXT_MODEL',
        'GROQ_VISION_MODEL', 'RAG_QUERY_EXPANSION',
    ):
        monkeypatch.delenv(name, raising=False)
    vault.reset_for_tests()
    backend.invalidate_all_graphs()
    backend._THREAD_VS.clear()
    backend._THREAD_BM25.clear()
    backend._THREAD_CHUNKS.clear()
    monkeypatch.setattr(backend, 'get_embeddings', lambda: TestEmbeddings())
    yield
    for tid in [r[0] for r in storage.CONN.execute('SELECT thread_id FROM threads')]:
        backend.checkpointer.delete_thread(tid)
        storage.delete_thread(tid)


def pytest_sessionfinish(session, exitstatus):
    backend.conn.close()
    vault._CONN.close()
    storage.CONN.close()
    shutil.rmtree(_test_data, ignore_errors=True)
