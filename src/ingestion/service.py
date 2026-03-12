"""
Ingestion service for processing and storing call transcripts
"""

from pathlib import Path
from typing import List
from src.ingestion.parser import TranscriptParser, extract_call_metadata
from src.storage.database import DatabaseManager
from src.storage.embeddings import EmbeddingProvider
from src.storage.vector_store import FAISSVectorStore


class IngestionService:
    """Service for ingesting call transcripts into the system"""

    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_store: FAISSVectorStore,
        embedding_provider: EmbeddingProvider,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.parser = TranscriptParser(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def ingest_transcript(self, file_path: str) -> dict:
        """
        Ingest a single transcript file

        Args:
            file_path: Path to the transcript file

        Returns:
            dict with ingestion statistics
        """
        print(f"\n📄 Processing: {file_path}")

        # Parse the transcript
        call_id, chunks = self.parser.parse_file(file_path)

        # Check if already ingested
        existing = self.db_manager.get_transcript_by_call_id(call_id)
        if existing:
            print(f"⚠️  Call '{call_id}' already exists in database. Skipping.")
            return {"status": "skipped", "call_id": call_id, "reason": "already_exists"}

        # Extract metadata
        metadata = extract_call_metadata(file_path)

        # Add transcript to database
        transcript = self.db_manager.add_transcript(
            call_id=call_id, filename=Path(file_path).name
        )

        print(f"✓ Created transcript record: {call_id}")
        print(f"  Participants: {', '.join(metadata['participants'])}")
        print(f"  Duration: {metadata['duration']}")
        print(f"  Total chunks: {len(chunks)}")

        # Process chunks
        chunk_texts = [chunk.text for chunk in chunks]

        print(f"🔢 Generating embeddings for {len(chunk_texts)} chunks...")
        embeddings = self.embedding_provider.embed_batch(chunk_texts)

        print("💾 Adding to vector store...")
        faiss_indices = self.vector_store.add_vectors(embeddings)

        # Add chunks to database
        for chunk, faiss_idx in zip(chunks, faiss_indices):
            self.db_manager.add_chunk(
                transcript_id=transcript.id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                timestamp_range=chunk.timestamp_range,
                faiss_index=faiss_idx,
            )

        # Update chunk count
        self.db_manager.update_transcript_chunk_count(transcript.id, len(chunks))

        # Save vector store
        self.vector_store.save()

        print(f"✅ Successfully ingested '{call_id}' with {len(chunks)} chunks\n")

        return {
            "status": "success",
            "call_id": call_id,
            "chunks": len(chunks),
            "participants": metadata["participants"],
        }

    def ingest_directory(self, directory_path: str) -> List[dict]:
        """
        Ingest all transcript files from a directory

        Args:
            directory_path: Path to directory containing transcript files

        Returns:
            List of ingestion results
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        # Find all .txt files
        transcript_files = list(directory.glob("*.txt"))

        if not transcript_files:
            print(f"⚠️  No .txt files found in {directory_path}")
            return []

        print(f"\n🚀 Starting ingestion of {len(transcript_files)} files...")
        print("=" * 60)

        results = []
        for file_path in transcript_files:
            try:
                result = self.ingest_transcript(str(file_path))
                results.append(result)
            except Exception as e:
                print(f"❌ Error ingesting {file_path.name}: {str(e)}")
                results.append(
                    {"status": "error", "file": file_path.name, "error": str(e)}
                )

        # Print summary
        successful = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errors = sum(1 for r in results if r["status"] == "error")

        print("=" * 60)
        print("\n📊 Ingestion Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   ❌ Errors: {errors}")
        print(f"   📦 Total vectors in store: {self.vector_store.get_total_vectors()}")

        return results
