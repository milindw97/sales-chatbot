"""
Transcript parsing and chunking utilities
"""

import re
from typing import List, Tuple
from pathlib import Path


class TranscriptChunk:
    """Represents a chunk of transcript with metadata"""

    def __init__(self, text: str, timestamp_range: str, chunk_index: int):
        self.text = text
        self.timestamp_range = timestamp_range
        self.chunk_index = chunk_index

    def __repr__(self):
        return f"<Chunk {self.chunk_index}: {self.timestamp_range}>"


class TranscriptParser:
    """Parses call transcripts and creates chunks"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_file(self, file_path: str) -> Tuple[str, List[TranscriptChunk]]:
        """
        Parse a transcript file and create chunks

        Args:
            file_path: Path to the transcript file

        Returns:
            call_id: Extracted call ID from filename
            chunks: List of TranscriptChunk objects
        """
        path = Path(file_path)

        # Extract call_id from filename (e.g., "1_demo_call.txt" -> "demo_call")
        call_id = self._extract_call_id(path.name)

        # Read file content
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Create chunks
        chunks = self._create_chunks(content)

        return call_id, chunks

    def _extract_call_id(self, filename: str) -> str:
        """Extract a meaningful call ID from filename"""
        # Remove timestamp prefix and extension
        # e.g., "1773333625507_1_demo_call.txt" -> "demo_call"
        name = filename.replace(".txt", "")

        # Remove numeric prefixes
        parts = name.split("_")

        # Find the first non-numeric part
        call_parts = []
        for part in parts:
            if not part.isdigit():
                call_parts.append(part)

        return "_".join(call_parts) if call_parts else name

    def _extract_timestamp(self, text: str) -> str:
        """Extract timestamp from text (e.g., [00:00] or [01:30])"""
        match = re.search(r"\[(\d{2}:\d{2})\]", text)
        return match.group(1) if match else None

    def _create_chunks(self, content: str) -> List[TranscriptChunk]:
        """
        Create overlapping chunks from content

        Args:
            content: Full transcript text

        Returns:
            List of TranscriptChunk objects
        """
        chunks = []

        # Split into lines for timestamp extraction
        lines = content.split("\n")

        # Find all timestamps in the content
        timestamps = []
        for line in lines:
            ts = self._extract_timestamp(line)
            if ts:
                timestamps.append(ts)

        # Simple chunking: split by character count with overlap
        text = content
        start = 0
        chunk_index = 0

        while start < len(text):
            # Get chunk
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Find timestamps in this chunk
            chunk_timestamps = []
            for ts in timestamps:
                if f"[{ts}]" in chunk_text:
                    chunk_timestamps.append(ts)

            # Determine timestamp range
            if chunk_timestamps:
                timestamp_range = f"[{chunk_timestamps[0]}] - [{chunk_timestamps[-1]}]"
            else:
                timestamp_range = "N/A"

            # Create chunk
            chunk = TranscriptChunk(
                text=chunk_text.strip(),
                timestamp_range=timestamp_range,
                chunk_index=chunk_index,
            )
            chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            chunk_index += 1

            # Prevent infinite loop
            if end >= len(text):
                break

        return chunks


def extract_call_metadata(file_path: str) -> dict:
    """Extract basic metadata from a call transcript file"""
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract participants
    participants = set()
    for line in content.split("\n"):
        # Match patterns like "AE (Jordan):" or "Prospect (Priya):"
        match = re.search(r"\b(AE|SE|Prospect|CISO|VP)\s*\(([^)]+)\)", line)
        if match:
            participants.add(match.group(2))

    # Extract all timestamps
    timestamps = re.findall(r"\[(\d{2}:\d{2})\]", content)
    duration = timestamps[-1] if timestamps else "Unknown"

    return {
        "filename": path.name,
        "participants": list(participants),
        "duration": duration,
        "total_lines": len(content.split("\n")),
    }
