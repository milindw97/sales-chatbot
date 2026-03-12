"""
Tests for src/storage/database.py

Covers:
  - add_transcript         (return value, call_id, uniqueness)
  - add_chunk              (return value, fields)
  - get_transcript_by_call_id (found / not found)
  - get_all_transcripts    (empty / multiple)
  - get_chunk_by_faiss_index (found / not found)
  - update_transcript_chunk_count
  - DetachedInstanceError is NOT raised (objects usable after session close)
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# add_transcript
# ---------------------------------------------------------------------------


class TestAddTranscript:
    def test_returns_object_with_correct_call_id(self, tmp_db):
        t = tmp_db.add_transcript("call_001", "call_001.txt")
        assert t.call_id == "call_001"

    def test_returns_object_with_correct_filename(self, tmp_db):
        t = tmp_db.add_transcript("call_001", "call_001.txt")
        assert t.filename == "call_001.txt"

    def test_auto_assigns_id(self, tmp_db):
        t = tmp_db.add_transcript("call_002", "call_002.txt")
        assert t.id is not None
        assert isinstance(t.id, int)

    def test_id_attribute_accessible_after_session_close(self, tmp_db):
        """Regression: DetachedInstanceError should NOT occur."""
        t = tmp_db.add_transcript("call_003", "f.txt")
        # Accessing .id after session close must not raise
        _ = t.id
        _ = t.call_id

    def test_two_transcripts_get_different_ids(self, tmp_db):
        t1 = tmp_db.add_transcript("c1", "c1.txt")
        t2 = tmp_db.add_transcript("c2", "c2.txt")
        assert t1.id != t2.id

    def test_ingestion_date_is_set(self, tmp_db):
        t = tmp_db.add_transcript("c_date", "f.txt")
        assert t.ingestion_date is not None


# ---------------------------------------------------------------------------
# add_chunk
# ---------------------------------------------------------------------------


class TestAddChunk:
    def test_returns_object_with_correct_fields(self, db_with_transcript):
        db, transcript = db_with_transcript
        chunk = db.add_chunk(
            transcript_id=transcript.id,
            chunk_index=0,
            text="Hello world chunk",
            timestamp_range="[00:00] - [00:30]",
            faiss_index=0,
        )
        assert chunk.chunk_index == 0
        assert chunk.text == "Hello world chunk"
        assert chunk.timestamp_range == "[00:00] - [00:30]"
        assert chunk.faiss_index == 0

    def test_chunk_id_accessible_after_session_close(self, db_with_transcript):
        """Regression: DetachedInstanceError should NOT occur."""
        db, transcript = db_with_transcript
        chunk = db.add_chunk(
            transcript_id=transcript.id,
            chunk_index=0,
            text="data",
            timestamp_range="N/A",
            faiss_index=0,
        )
        assert isinstance(chunk.id, int)

    def test_multiple_chunks_get_sequential_faiss_indices(self, db_with_transcript):
        db, transcript = db_with_transcript
        ids = []
        for i in range(3):
            c = db.add_chunk(
                transcript_id=transcript.id,
                chunk_index=i,
                text=f"chunk {i}",
                timestamp_range="N/A",
                faiss_index=i,
            )
            ids.append(c.faiss_index)
        assert ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# get_transcript_by_call_id
# ---------------------------------------------------------------------------


class TestGetTranscriptByCallId:
    def test_found(self, tmp_db):
        tmp_db.add_transcript("find_me", "f.txt")
        result = tmp_db.get_transcript_by_call_id("find_me")
        assert result is not None
        assert result.call_id == "find_me"

    def test_not_found_returns_none(self, tmp_db):
        result = tmp_db.get_transcript_by_call_id("does_not_exist")
        assert result is None

    def test_attributes_accessible_after_return(self, tmp_db):
        tmp_db.add_transcript("accessible", "f.txt")
        t = tmp_db.get_transcript_by_call_id("accessible")
        # Must not raise DetachedInstanceError
        _ = t.id
        _ = t.filename


# ---------------------------------------------------------------------------
# get_all_transcripts
# ---------------------------------------------------------------------------


class TestGetAllTranscripts:
    def test_empty_db_returns_empty_list(self, tmp_db):
        assert tmp_db.get_all_transcripts() == []

    def test_returns_all_added_transcripts(self, tmp_db):
        tmp_db.add_transcript("a", "a.txt")
        tmp_db.add_transcript("b", "b.txt")
        transcripts = tmp_db.get_all_transcripts()
        call_ids = {t.call_id for t in transcripts}
        assert call_ids == {"a", "b"}

    def test_count_matches(self, tmp_db):
        for i in range(5):
            tmp_db.add_transcript(f"call_{i}", f"call_{i}.txt")
        assert len(tmp_db.get_all_transcripts()) == 5

    def test_attributes_accessible_after_return(self, tmp_db):
        tmp_db.add_transcript("z", "z.txt")
        transcripts = tmp_db.get_all_transcripts()
        for t in transcripts:
            _ = t.call_id
            _ = t.filename


# ---------------------------------------------------------------------------
# get_chunk_by_faiss_index
# ---------------------------------------------------------------------------


class TestGetChunkByFaissIndex:
    def test_found(self, db_with_transcript):
        db, transcript = db_with_transcript
        db.add_chunk(
            transcript_id=transcript.id,
            chunk_index=0,
            text="findable",
            timestamp_range="N/A",
            faiss_index=42,
        )
        chunk = db.get_chunk_by_faiss_index(42)
        assert chunk is not None
        assert chunk.text == "findable"

    def test_not_found_returns_none(self, tmp_db):
        assert tmp_db.get_chunk_by_faiss_index(9999) is None

    def test_attributes_accessible_after_return(self, db_with_transcript):
        db, transcript = db_with_transcript
        db.add_chunk(
            transcript_id=transcript.id,
            chunk_index=0,
            text="data",
            timestamp_range="N/A",
            faiss_index=7,
        )
        chunk = db.get_chunk_by_faiss_index(7)
        _ = chunk.id
        _ = chunk.chunk_index


# ---------------------------------------------------------------------------
# update_transcript_chunk_count
# ---------------------------------------------------------------------------


class TestUpdateTranscriptChunkCount:
    def test_updates_count(self, tmp_db):
        t = tmp_db.add_transcript("upd", "upd.txt")
        tmp_db.update_transcript_chunk_count(t.id, 15)
        refreshed = tmp_db.get_transcript_by_call_id("upd")
        assert refreshed.total_chunks == 15

    def test_update_nonexistent_id_is_noop(self, tmp_db):
        # Should not raise
        tmp_db.update_transcript_chunk_count(99999, 10)

    def test_default_chunk_count_is_zero(self, tmp_db):
        t = tmp_db.add_transcript("zero", "zero.txt")
        assert t.total_chunks == 0
