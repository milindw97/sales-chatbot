"""
Retrieval service for querying call transcripts using RAG
"""

from src.llm.prompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT_TEMPLATE,
    SUMMARIZATION_SYSTEM_PROMPT,
    SUMMARIZATION_USER_PROMPT_TEMPLATE,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_TEMPLATE,
)
from typing import List, Dict
from src.storage.database import DatabaseManager, CallChunk, CallTranscript
from src.storage.embeddings import EmbeddingProvider
from src.storage.vector_store import FAISSVectorStore
from src.llm.providers import LLMProvider


class RetrievalResult:
    """Represents a retrieved chunk with metadata"""

    def __init__(
        self,
        text: str,
        call_id: str,
        timestamp_range: str,
        similarity_score: float,
        chunk_index: int,
    ):
        self.text = text
        self.call_id = call_id
        self.timestamp_range = timestamp_range
        self.similarity_score = similarity_score
        self.chunk_index = chunk_index

    def __repr__(self):
        return f"<Result from '{self.call_id}' {self.timestamp_range} (score: {self.similarity_score:.3f})>"


class RetrievalService:
    """Service for retrieving relevant information from call transcripts"""

    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_store: FAISSVectorStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        top_k: int = 5,
    ):
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query

        Args:
            query: User's question
            top_k: Number of results to retrieve (default: self.top_k)

        Returns:
            List of RetrievalResult objects
        """
        k = top_k if top_k is not None else self.top_k

        # Check if we have any vectors
        if self.vector_store.get_total_vectors() == 0:
            print("⚠️  No vectors in store. Please ingest transcripts first.")
            return []

        # Embed the query
        query_embedding = self.embedding_provider.embed_text(query)

        # Search FAISS
        distances, indices = self.vector_store.search(query_embedding, k=k)

        # Debug: Check if search returned results
        if len(indices) == 0:
            print("⚠️  No results from vector search.")
            return []

        # Get chunks from database - need to keep session open
        results = []
        session = self.db_manager.get_session()
        try:
            for distance, faiss_idx in zip(distances, indices):
                # Query within the session

                chunk = (
                    session.query(CallChunk)
                    .filter_by(faiss_index=int(faiss_idx))
                    .first()
                )

                if chunk:
                    # Access transcript within session
                    transcript = chunk.transcript

                    # Convert distance to similarity score (inverse)
                    similarity = 1 / (1 + distance)

                    result = RetrievalResult(
                        text=chunk.text,
                        call_id=transcript.call_id,
                        timestamp_range=chunk.timestamp_range,
                        similarity_score=similarity,
                        chunk_index=chunk.chunk_index,
                    )
                    results.append(result)
                else:
                    print(f"⚠️  Chunk not found for FAISS index {faiss_idx}")
        finally:
            session.close()

        return results

    def _rewrite_query(self, question: str, history: List[Dict]) -> str:
        """
        Rewrite a follow-up question to be a standalone query based on conversation history.

        Args:
            question: The follow-up question
            history: List of previous conversation turns

        Returns:
            Rewritten standalone query
        """
        if not history:
            return question

        # Format the history for the prompt
        history_text = ""
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

        user_prompt = QUERY_REWRITE_USER_TEMPLATE.format(
            history=history_text.strip(), question=question
        )

        try:
            rewritten_query = self.llm_provider.generate_with_system(
                QUERY_REWRITE_SYSTEM_PROMPT, user_prompt
            )
            # Remove any surrounding quotes
            rewritten_query = rewritten_query.strip().strip("\"'")
            return rewritten_query
        except Exception as e:
            print(f"⚠️  Error rewriting query, using original: {e}")
            return question

    def query(self, question: str, top_k: int = None, history: List[Dict] = None) -> Dict:
        """
        Query the system and generate an answer using RAG

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            history: Optional list of previous conversation turns

        Returns:
            Dict with answer and source information
        """
        history = history or []

        # Rewrite the query if we have history
        search_query = self._rewrite_query(question, history)
        
        # Retrieve relevant chunks using the rewritten query
        results = self.retrieve(search_query, top_k=top_k or 10)  # Retrieve more chunks

        if not results:
            return {
                "answer": "I couldn't find any relevant information in the call transcripts. Try:\n- Listing calls with 'list calls'\n- Asking about specific topics like pricing, objections, or security\n- Summarizing a specific call",
                "sources": [],
                "query": search_query,
            }

        # Build context from retrieved chunks - use top 5 most relevant
        top_results = results[:5]
        context_parts = []
        for i, result in enumerate(top_results, 1):
            context_parts.append(
                f"[Source {i} - Call: {result.call_id}, Time: {result.timestamp_range}]\n{result.text}\n"
            )

        context = "\n".join(context_parts)

        # Format history block for the QA prompt
        history_block = ""
        if history:
            history_block = "Conversation History:\n"
            for turn in history:
                role = "User" if turn["role"] == "user" else "Assistant"
                history_block += f"{role}: {turn['content']}\n"
            history_block += "\n"

        # Build prompts from centralised templates
        user_prompt = QA_USER_PROMPT_TEMPLATE.format(
            context=context, history=history_block, question=question
        )

        # Generate answer
        try:
            answer = self.llm_provider.generate_with_system(
                QA_SYSTEM_PROMPT, user_prompt
            )
        except Exception as e:
            answer = f"Error generating answer: {str(e)}\n\nHere are the relevant sources I found:\n"
            for i, result in enumerate(top_results, 1):
                answer += f"\n{i}. From {result.call_id} at {result.timestamp_range}:\n{result.text[:200]}...\n"

        # Format sources
        sources = [
            {
                "call_id": r.call_id,
                "timestamp_range": r.timestamp_range,
                "snippet": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                "similarity_score": round(r.similarity_score, 3),
            }
            for r in top_results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "query": search_query,  # Return the search query
            "original_question": question,
            "num_sources": len(sources),
        }

    def summarize_call(self, call_id: str) -> Dict:
        """
        Generate a summary of a specific call

        Args:
            call_id: ID of the call to summarize

        Returns:
            Dict with summary and metadata
        """
        # Get transcript from database with session management
        session = self.db_manager.get_session()
        try:
            transcript = (
                session.query(CallTranscript).filter_by(call_id=call_id).first()
            )

            if not transcript:
                return {"error": f"Call '{call_id}' not found", "call_id": call_id}

            # Get all chunks for this call
            chunks = (
                session.query(CallChunk)
                .filter_by(transcript_id=transcript.id)
                .order_by(CallChunk.chunk_index)
                .all()
            )

            # Extract data while session is open
            total_chunks = transcript.total_chunks
            ingestion_date = transcript.ingestion_date.isoformat()

            # Combine all chunks
            full_text = "\n\n".join([chunk.text for chunk in chunks])

        finally:
            session.close()

        # Limit context if too long (take first ~3000 chars)
        if len(full_text) > 3000:
            full_text = full_text[:3000] + "\n\n[... transcript continues ...]"

        # Build prompts from centralised templates
        user_prompt = SUMMARIZATION_USER_PROMPT_TEMPLATE.format(
            call_id=call_id, full_text=full_text
        )

        # Generate summary
        summary = self.llm_provider.generate_with_system(
            SUMMARIZATION_SYSTEM_PROMPT, user_prompt
        )

        return {
            "call_id": call_id,
            "summary": summary,
            "total_chunks": total_chunks,
            "ingestion_date": ingestion_date,
        }

    def list_calls(self) -> List[Dict]:
        """
        List all available calls

        Returns:
            List of call metadata
        """
        transcripts = self.db_manager.get_all_transcripts()

        return [
            {
                "call_id": t.call_id,
                "filename": t.filename,
                "total_chunks": t.total_chunks,
                "ingestion_date": t.ingestion_date.isoformat(),
            }
            for t in transcripts
        ]
