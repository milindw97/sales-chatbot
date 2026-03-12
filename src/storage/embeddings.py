"""
Embedding providers for converting text to vectors
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings"""
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Embedding provider using Sentence Transformers (local)"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings"""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimension


class GeminiEmbedding(EmbeddingProvider):
    """Embedding provider using Google Gemini API"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_name = "models/embedding-001"
        self.dimension = 768  # Gemini embedding dimension
        print(f"Initialized Gemini Embedding with model: {self.model_name}")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string"""

        result = genai.embed_content(
            model=self.model_name, content=text, task_type="retrieval_document"
        )
        return np.array(result["embedding"], dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings"""

        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name, content=text, task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return np.array(embeddings, dtype=np.float32)

    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimension


def get_embedding_provider(provider_type: str, **kwargs) -> EmbeddingProvider:
    """Factory function to get the appropriate embedding provider"""
    if provider_type == "sentence-transformers":
        model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
        return SentenceTransformerEmbedding(model_name)
    if provider_type == "gemini":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("Gemini API key is required for Gemini embeddings")
        return GeminiEmbedding(api_key)
    raise ValueError(f"Unknown embedding provider: {provider_type}")
