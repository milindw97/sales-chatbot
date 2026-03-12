"""
Tests for src/ingestion/service.py

Covers:
  - ingest_transcript: success path (status, call_id, chunks key)
  - ingest_transcript: skip when already ingested
  - ingest_transcript: embedding and vector-store calls made correctly
  - ingest_directory:  empty directory returns []
  - ingest_directory:  nonexistent directory raises ValueError
  - ingest_directory:  summarises successes / skips / errors correctly
  - ingest_directory:  handles per-file errors gracefully

All external I/O (DB, embeddings, vector store) is isolated via mocks or
real lightweight in-memory fixtures from conftest.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.service import IngestionService


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_service(tmp_db, tmp_vector_store, mock_embedding_provider):
    return IngestionService(
        db_manager=tmp_db,
        vector_store=tmp_vector_store,
        embedding_provider=mock_embedding_provider,
        chunk_size=256,
        chunk_overlap=20,
    )


# ---------------------------------------------------------------------------
# ingest_transcript – success path
# ---------------------------------------------------------------------------


class TestIngestTranscriptSuccess:
    def test_status_is_success(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
    ):
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        result = svc.ingest_transcript(sample_transcript_file)
        assert result["status"] == "success"

    def test_call_id_extracted(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
    ):
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        result = svc.ingest_transcript(sample_transcript_file)
        assert result["call_id"] == "demo_call"

    def test_chunk_count_in_result(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
    ):
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        result = svc.ingest_transcript(sample_transcript_file)
        assert result["chunks"] > 0

    def test_vectors_added_to_store(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
    ):
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        svc.ingest_transcript(sample_transcript_file)
        assert tmp_vector_store.get_total_vectors() > 0

    def test_chunks_persisted_to_db(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
    ):
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        result = svc.ingest_transcript(sample_transcript_file)
        # The transcript should be retrievable from DB
        t = tmp_db.get_transcript_by_call_id("demo_call")
        assert t is not None
        assert t.total_chunks == result["chunks"]

    def test_embed_batch_called_once(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
    ):
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        svc.ingest_transcript(sample_transcript_file)
        mock_embedding_provider.embed_batch.assert_called_once()

    def test_participants_returned(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
    ):
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        result = svc.ingest_transcript(sample_transcript_file)
        assert isinstance(result["participants"], list)


# ---------------------------------------------------------------------------
# ingest_transcript – skip (already ingested)
# ---------------------------------------------------------------------------


class TestIngestTranscriptSkip:
    def test_second_ingest_returns_skipped(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
    ):
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        svc.ingest_transcript(sample_transcript_file)
        result = svc.ingest_transcript(sample_transcript_file)
        assert result["status"] == "skipped"
        assert result["reason"] == "already_exists"

    def test_skip_does_not_add_more_vectors(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, sample_transcript_file
    ):
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        svc.ingest_transcript(sample_transcript_file)
        count_after_first = tmp_vector_store.get_total_vectors()
        svc.ingest_transcript(sample_transcript_file)
        assert tmp_vector_store.get_total_vectors() == count_after_first


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------


class TestIngestDirectory:
    def test_nonexistent_directory_raises(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, tmp_path
    ):
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        with pytest.raises(ValueError, match="does not exist"):
            svc.ingest_directory(str(tmp_path / "ghost"))

    def test_empty_directory_returns_empty_list(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, tmp_path
    ):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        results = svc.ingest_directory(str(empty_dir))
        assert results == []

    def test_ingests_multiple_files(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, tmp_path
    ):
        from tests.conftest import SAMPLE_TRANSCRIPT

        dir_path = tmp_path / "transcripts"
        dir_path.mkdir()
        for i in range(3):
            f = dir_path / f"1_call{i}_transcript.txt"
            f.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        results = svc.ingest_directory(str(dir_path))
        assert len(results) == 3
        successful = [r for r in results if r["status"] == "success"]
        assert len(successful) == 3

    def test_error_in_one_file_does_not_stop_others(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, tmp_path
    ):
        from tests.conftest import SAMPLE_TRANSCRIPT

        dir_path = tmp_path / "mix"
        dir_path.mkdir()
        # Good file
        (dir_path / "1_good_call.txt").write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
        # Bad file (binary garbage that parse_file will choke on)
        (dir_path / "bad.txt").write_bytes(b"\xff\xfe broken")

        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        # patch parse_file to raise for the bad file only
        original_parse = svc.parser.parse_file

        def patched_parse(path):
            if "bad" in path:
                raise RuntimeError("Simulated parse error")
            return original_parse(path)

        svc.parser.parse_file = patched_parse
        results = svc.ingest_directory(str(dir_path))

        statuses = {r["status"] for r in results}
        assert "success" in statuses
        assert "error" in statuses

    def test_ignores_non_txt_files(
        self, tmp_db, tmp_vector_store, mock_embedding_provider, tmp_path
    ):
        from tests.conftest import SAMPLE_TRANSCRIPT

        dir_path = tmp_path / "mixed_files"
        dir_path.mkdir()
        (dir_path / "1_demo_call.txt").write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
        (dir_path / "readme.md").write_text("# nope", encoding="utf-8")
        (dir_path / "data.csv").write_text("a,b", encoding="utf-8")

        svc = make_service(tmp_db, tmp_vector_store, mock_embedding_provider)
        results = svc.ingest_directory(str(dir_path))
        assert len(results) == 1  # only the .txt file
