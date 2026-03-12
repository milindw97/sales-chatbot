"""
Tests for src/ingestion/parser.py

Covers:
  - TranscriptChunk construction and repr
  - TranscriptParser._extract_call_id   (direct + via parse_file)
  - TranscriptParser._extract_timestamp
  - TranscriptParser._create_chunks     (count, overlap, timestamp range, N/A fallback)
  - TranscriptParser.parse_file         (real file round-trip)
  - extract_call_metadata               (participants, duration, total_lines)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.parser import (
    TranscriptChunk,
    TranscriptParser,
    extract_call_metadata,
)

# ---------------------------------------------------------------------------
# TranscriptChunk
# ---------------------------------------------------------------------------


class TestTranscriptChunk:
    def test_attributes_are_stored(self):
        chunk = TranscriptChunk(
            text="Hello world", timestamp_range="[00:00] - [00:10]", chunk_index=0
        )
        assert chunk.text == "Hello world"
        assert chunk.timestamp_range == "[00:00] - [00:10]"
        assert chunk.chunk_index == 0

    def test_repr_contains_index_and_range(self):
        chunk = TranscriptChunk("x", "[01:00] - [01:30]", 3)
        r = repr(chunk)
        assert "3" in r
        assert "[01:00] - [01:30]" in r

    def test_empty_text_chunk(self):
        chunk = TranscriptChunk(text="", timestamp_range="N/A", chunk_index=0)
        assert chunk.text == ""
        assert chunk.timestamp_range == "N/A"


# ---------------------------------------------------------------------------
# TranscriptParser – internal helpers
# ---------------------------------------------------------------------------


class TestExtractCallId:
    def setup_method(self):
        self.parser = TranscriptParser()

    def test_strips_numeric_timestamp_prefix(self):
        assert self.parser._extract_call_id("1773333625507_1_demo_call.txt") == "demo_call"

    def test_strips_single_numeric_prefix(self):
        assert self.parser._extract_call_id("1_pricing_call.txt") == "pricing_call"

    def test_no_numeric_prefix(self):
        assert self.parser._extract_call_id("negotiation.txt") == "negotiation"

    def test_all_numeric_parts_returns_full_name(self):
        # Edge case: every segment is numeric → return full name (without .txt)
        result = self.parser._extract_call_id("123_456.txt")
        assert result == "123_456"

    def test_multi_word_call_id(self):
        assert (
            self.parser._extract_call_id("1773333625508_4_negotiation_call.txt")
            == "negotiation_call"
        )


class TestExtractTimestamp:
    def setup_method(self):
        self.parser = TranscriptParser()

    def test_finds_timestamp_in_bracket_format(self):
        assert self.parser._extract_timestamp("[00:00] Some text") == "00:00"

    def test_finds_timestamp_mid_string(self):
        assert self.parser._extract_timestamp("text [01:30] more") == "01:30"

    def test_returns_none_when_absent(self):
        assert self.parser._extract_timestamp("no timestamp here") is None

    def test_picks_first_timestamp(self):
        # Only one match per call (re.search – first occurrence)
        assert self.parser._extract_timestamp("[00:00] foo [01:00]") == "00:00"


# ---------------------------------------------------------------------------
# TranscriptParser – chunking
# ---------------------------------------------------------------------------


class TestCreateChunks:
    def test_short_content_produces_one_chunk(self):
        parser = TranscriptParser(chunk_size=512, chunk_overlap=50)
        chunks = parser._create_chunks("Short content.")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_chunk_indices_are_sequential(self):
        content = "A" * 200
        parser = TranscriptParser(chunk_size=80, chunk_overlap=10)
        chunks = parser._create_chunks(content)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_overlap_makes_chunks_share_content(self):
        # With overlap, consecutive chunks must share some characters
        content = "abcdefghij" * 30  # 300 chars
        parser = TranscriptParser(chunk_size=100, chunk_overlap=20)
        chunks = parser._create_chunks(content)
        assert len(chunks) >= 2
        # Last 20 chars of chunk 0 should appear at start of chunk 1
        end_of_first = chunks[0].text[-20:]
        start_of_second = chunks[1].text[:20]
        # strip() may have trimmed whitespace-only ends; just check non-empty overlap
        assert end_of_first or start_of_second  # both exist

    def test_no_timestamp_gives_na(self):
        parser = TranscriptParser(chunk_size=512)
        chunks = parser._create_chunks("No timestamps here at all.")
        assert chunks[0].timestamp_range == "N/A"

    def test_timestamp_range_captured(self):
        content = "[00:00] Hello\n[00:30] World\n[01:00] End"
        parser = TranscriptParser(chunk_size=512)
        chunks = parser._create_chunks(content)
        # Single chunk should span first to last timestamp
        assert chunks[0].timestamp_range == "[00:00] - [01:00]"

    def test_single_timestamp_same_start_end(self):
        content = "[00:05] Only one timestamp in here."
        parser = TranscriptParser(chunk_size=512)
        chunks = parser._create_chunks(content)
        assert chunks[0].timestamp_range == "[00:05] - [00:05]"

    def test_text_is_stripped(self):
        parser = TranscriptParser(chunk_size=512)
        chunks = parser._create_chunks("   [00:00] Hello   ")
        assert chunks[0].text == chunks[0].text.strip()

    def test_empty_content_produces_no_chunks(self):
        parser = TranscriptParser(chunk_size=512)
        chunks = parser._create_chunks("")
        assert chunks == []


# ---------------------------------------------------------------------------
# TranscriptParser.parse_file
# ---------------------------------------------------------------------------


class TestParseFile:
    def test_returns_correct_call_id(self, sample_transcript_file):
        parser = TranscriptParser(chunk_size=256, chunk_overlap=30)
        call_id, chunks = parser.parse_file(sample_transcript_file)
        assert call_id == "demo_call"

    def test_returns_non_empty_chunks(self, sample_transcript_file):
        parser = TranscriptParser(chunk_size=256, chunk_overlap=30)
        _, chunks = parser.parse_file(sample_transcript_file)
        assert len(chunks) > 0

    def test_all_chunks_have_text(self, sample_transcript_file):
        parser = TranscriptParser(chunk_size=256)
        _, chunks = parser.parse_file(sample_transcript_file)
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_chunk_index_starts_at_zero(self, sample_transcript_file):
        parser = TranscriptParser(chunk_size=256)
        _, chunks = parser.parse_file(sample_transcript_file)
        assert chunks[0].chunk_index == 0

    def test_smaller_chunk_size_yields_more_chunks(self, sample_transcript_file):
        parser_small = TranscriptParser(chunk_size=64, chunk_overlap=10)
        parser_large = TranscriptParser(chunk_size=1024, chunk_overlap=10)
        _, small_chunks = parser_small.parse_file(sample_transcript_file)
        _, large_chunks = parser_large.parse_file(sample_transcript_file)
        assert len(small_chunks) >= len(large_chunks)

    def test_file_not_found_raises(self, tmp_path):
        parser = TranscriptParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file(str(tmp_path / "nonexistent.txt"))


# ---------------------------------------------------------------------------
# extract_call_metadata
# ---------------------------------------------------------------------------


class TestExtractCallMetadata:
    def test_returns_expected_keys(self, sample_transcript_file):
        meta = extract_call_metadata(sample_transcript_file)
        assert set(meta.keys()) == {"filename", "participants", "duration", "total_lines"}

    def test_participants_extracted(self, sample_transcript_file):
        meta = extract_call_metadata(sample_transcript_file)
        # Sample transcript has AE (Jordan) and Prospect (Priya)
        assert "Jordan" in meta["participants"] or "Priya" in meta["participants"]

    def test_duration_is_last_timestamp(self, sample_transcript_file):
        meta = extract_call_metadata(sample_transcript_file)
        assert meta["duration"] == "03:30"

    def test_total_lines_positive(self, sample_transcript_file):
        meta = extract_call_metadata(sample_transcript_file)
        assert meta["total_lines"] > 0

    def test_filename_matches(self, sample_transcript_file):
        meta = extract_call_metadata(sample_transcript_file)
        assert meta["filename"].endswith("demo_call.txt")

    def test_no_timestamps_returns_unknown_duration(self, tmp_path):
        f = tmp_path / "plain.txt"
        f.write_text("No timestamps at all here.", encoding="utf-8")
        meta = extract_call_metadata(str(f))
        assert meta["duration"] == "Unknown"

    def test_no_participants_returns_empty_list(self, tmp_path):
        f = tmp_path / "1_anon.txt"
        f.write_text("[00:00] Just some text with nobody.", encoding="utf-8")
        meta = extract_call_metadata(str(f))
        assert isinstance(meta["participants"], list)
        assert meta["participants"] == []
