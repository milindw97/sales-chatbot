"""
Shared pytest fixtures for all test modules.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure src is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager
from src.storage.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Yield a fresh DatabaseManager backed by a temp SQLite file."""
    db_file = str(tmp_path / "test.db")
    db = DatabaseManager(db_file)
    yield db
    db.engine.dispose()


@pytest.fixture
def db_with_transcript(tmp_db):
    """A DatabaseManager that already has one transcript in it."""
    transcript = tmp_db.add_transcript("demo_call", "demo.txt")
    return tmp_db, transcript


# ---------------------------------------------------------------------------
# Vector-store fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_vector_store(tmp_path):
    """A fresh FAISSVectorStore that persists to a temp directory."""
    index_path = str(tmp_path / "test_index")
    return FAISSVectorStore(dimension=8, index_path=index_path)


# ---------------------------------------------------------------------------
# Mock embedding / LLM providers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_provider():
    """A mock EmbeddingProvider that returns deterministic 8-dim vectors."""
    provider = MagicMock()
    provider.get_dimension.return_value = 8
    provider.embed_text.side_effect = lambda text: np.ones(8, dtype="float32")
    provider.embed_batch.side_effect = lambda texts: np.ones(
        (len(texts), 8), dtype="float32"
    )
    return provider


@pytest.fixture
def mock_llm_provider():
    """A mock LLMProvider that returns a canned answer."""
    provider = MagicMock()
    provider.generate.return_value = "Mock LLM answer"
    provider.generate_with_system.return_value = "Mock LLM answer with system"
    return provider


# ---------------------------------------------------------------------------
# Sample transcript file
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = """\
[00:00] AE (Jordan): Hi Priya, thanks for joining the call today.
[00:10] Prospect (Priya): Good to meet you Jordan.
[00:20] AE (Jordan): Let me walk you through the pricing options.
[01:00] Prospect (Priya): What about security certifications?
[01:30] AE (Jordan): We are SOC2 and ISO 27001 certified.
[02:00] Prospect (Priya): The pricing seems high.
[02:30] AE (Jordan): We can offer a 10% discount for annual commitment.
[03:00] Prospect (Priya): Let me think about it and get back to you.
[03:15] AE (Jordan): Sounds good. I will send the proposal by EOD.
[03:30] *Call ends.*
"""


@pytest.fixture
def sample_transcript_file(tmp_path):
    """Write a minimal transcript to disk and return the path."""
    f = tmp_path / "1773333625507_1_demo_call.txt"
    f.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    return str(f)
