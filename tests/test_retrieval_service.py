"""
Tests for src/retrieval/service.py

Covers:
  RetrievalResult
    - all attributes stored correctly
    - repr format

  RetrievalService.retrieve
    - returns [] when vector store is empty
    - returns [] when FAISS finds no matching chunks in DB
    - returns RetrievalResult objects with correct fields
    - similarity score is in (0, 1]
    - respects top_k parameter

  RetrievalService.query
    - returns "no results" dict when retrieve returns empty
    - calls llm generate_with_system with context
    - result dict has required keys (answer, sources, query, num_sources)
    - LLM error falls back to raw sources
    - sources list length matches top-5 cap

  RetrievalService.summarize_call
    - returns error dict for unknown call_id
    - returns summary dict with expected keys
    - calls llm generate_with_system

  RetrievalService.list_calls
    - returns [] on empty DB
    - returns one entry per ingested call with correct keys
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.service import RetrievalResult, RetrievalService
from src.ingestion.service import IngestionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8


def _make_retrieval_service(
    tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
):
    return RetrievalService(
        db_manager=tmp_db,
        vector_store=tmp_vector_store,
        embedding_provider=mock_embedding_provider,
        llm_provider=mock_llm_provider,
        top_k=5,
    )


def _ingest_sample(
    tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
):
    """Helper to populate DB + vector store from the sample transcript."""
    svc = IngestionService(
        db_manager=tmp_db,
        vector_store=tmp_vector_store,
        embedding_provider=mock_embedding_provider,
        chunk_size=256,
        chunk_overlap=20,
    )
    return svc.ingest_transcript(sample_transcript_file)


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_attributes(self):
        r = RetrievalResult(
            text="hello",
            call_id="demo_call",
            timestamp_range="[00:00] - [00:30]",
            similarity_score=0.9,
            chunk_index=2,
        )
        assert r.text == "hello"
        assert r.call_id == "demo_call"
        assert r.timestamp_range == "[00:00] - [00:30]"
        assert r.similarity_score == pytest.approx(0.9)
        assert r.chunk_index == 2

    def test_repr_contains_call_id_and_score(self):
        r = RetrievalResult("t", "my_call", "N/A", 0.75, 0)
        assert "my_call" in repr(r)
        assert "0.750" in repr(r)


# ---------------------------------------------------------------------------
# RetrievalService.retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_empty_store_returns_empty_list(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
    ):
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        results = svc.retrieve("What happened?")
        assert results == []

    def test_returns_retrieval_results_after_ingest(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        results = svc.retrieve("pricing")
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_result_fields_populated(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        results = svc.retrieve("pricing", top_k=3)
        for r in results:
            assert r.call_id == "demo_call"
            assert r.text != ""
            assert 0 < r.similarity_score <= 1.0

    def test_top_k_limits_results(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        results = svc.retrieve("pricing", top_k=2)
        assert len(results) <= 2

    def test_uses_default_top_k_when_none(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        svc.top_k = 2
        results = svc.retrieve("pricing")
        assert len(results) <= 2

    def test_embed_text_called_with_query(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        svc.retrieve("security concerns")
        mock_embedding_provider.embed_text.assert_called_with("security concerns")


# ---------------------------------------------------------------------------
# RetrievalService.query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_no_results_returns_fallback_answer(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
    ):
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.query("anything")
        assert "answer" in result
        assert result["sources"] == []
        assert result["query"] == "anything"

    def test_result_has_required_keys(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.query("What were the pricing objections?")
        assert {"answer", "sources", "query", "num_sources"} <= result.keys()

    def test_llm_generate_with_system_called(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        svc.query("pricing")
        mock_llm_provider.generate_with_system.assert_called_once()

    def test_answer_comes_from_llm(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.query("pricing")
        assert result["answer"] == "Mock LLM answer with system"

    def test_llm_error_falls_back_gracefully(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        mock_llm_provider.generate_with_system.side_effect = RuntimeError("API down")
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.query("pricing")
        assert "Error generating answer" in result["answer"]
        assert len(result["sources"]) > 0

    def test_sources_capped_at_five(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.query("anything", top_k=20)
        assert len(result["sources"]) <= 5

    def test_source_has_required_fields(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.query("pricing")
        for src in result["sources"]:
            assert "call_id" in src
            assert "timestamp_range" in src
            assert "snippet" in src
            assert "similarity_score" in src

    def test_num_sources_matches_sources_length(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.query("pricing")
        assert result["num_sources"] == len(result["sources"])


# ---------------------------------------------------------------------------
# RetrievalService.summarize_call
# ---------------------------------------------------------------------------


class TestSummarizeCall:
    def test_unknown_call_returns_error_dict(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
    ):
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.summarize_call("nonexistent_call")
        assert "error" in result
        assert result["call_id"] == "nonexistent_call"

    def test_known_call_returns_summary_keys(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.summarize_call("demo_call")
        assert {"call_id", "summary", "total_chunks", "ingestion_date"} <= result.keys()

    def test_summary_comes_from_llm(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.summarize_call("demo_call")
        assert result["summary"] == "Mock LLM answer with system"

    def test_llm_generate_with_system_called_for_summary(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        svc.summarize_call("demo_call")
        mock_llm_provider.generate_with_system.assert_called_once()

    def test_ingestion_date_is_isoformat(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        result = svc.summarize_call("demo_call")
        # Should be parseable as ISO datetime
        from datetime import datetime

        datetime.fromisoformat(result["ingestion_date"])


# ---------------------------------------------------------------------------
# RetrievalService.list_calls
# ---------------------------------------------------------------------------


class TestListCalls:
    def test_empty_db_returns_empty_list(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
    ):
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        assert svc.list_calls() == []

    def test_returns_one_entry_per_call(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        calls = svc.list_calls()
        assert len(calls) == 1

    def test_call_entry_has_required_keys(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        calls = svc.list_calls()
        for c in calls:
            assert {"call_id", "filename", "total_chunks", "ingestion_date"} <= c.keys()

    def test_call_id_matches_ingested_name(
        self,
        tmp_db,
        tmp_vector_store,
        mock_embedding_provider,
        mock_llm_provider,
        sample_transcript_file,
    ):
        _ingest_sample(
            tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
        )
        svc = _make_retrieval_service(
            tmp_db, tmp_vector_store, mock_embedding_provider, mock_llm_provider
        )
        calls = svc.list_calls()
        assert calls[0]["call_id"] == "demo_call"
