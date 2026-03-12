"""
FAISS vector store for storing and retrieving embeddings
"""

import pickle
from pathlib import Path
from typing import List, Tuple
import faiss
import numpy as np


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""

    def __init__(self, dimension: int, index_path: str = None):
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None

        # Initialize FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(dimension)
        self.document_count = 0

        # Load existing index if path provided and the .index file exists
        if self.index_path and Path(str(self.index_path) + ".index").exists():
            self.load()

    def add_vectors(self, vectors: np.ndarray) -> List[int]:
        """
        Add vectors to the index

        Args:
            vectors: numpy array of shape (n, dimension)

        Returns:
            List of indices where vectors were added
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Ensure correct dtype and shape
        vectors = vectors.astype("float32")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}"
            )

        # Add to FAISS
        start_idx = self.document_count
        self.index.add(vectors)
        self.document_count += len(vectors)

        return list(range(start_idx, self.document_count))

    def search(
        self, query_vector: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors

        Args:
            query_vector: numpy array of shape (dimension,) or (1, dimension)
            k: number of nearest neighbors to return

        Returns:
            distances: numpy array of shape (k,)
            indices: numpy array of shape (k,)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype("float32")

        # Ensure we don't ask for more results than we have
        k = min(k, self.document_count)

        if k == 0:
            return np.array([]), np.array([])

        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]

    def save(self):
        """Save the FAISS index to disk"""
        if not self.index_path:
            raise ValueError("No index path specified")

        # Create directory if it doesn't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path) + ".index")

        # Save metadata
        metadata = {"dimension": self.dimension, "document_count": self.document_count}
        with open(str(self.index_path) + ".meta", "wb") as f:
            pickle.dump(metadata, f)

        print(
            f"Saved FAISS index with {self.document_count} vectors to {self.index_path}"
        )

    def load(self):
        """Load the FAISS index from disk"""
        if not self.index_path or not self.index_path.parent.exists():
            return

        index_file = str(self.index_path) + ".index"
        meta_file = str(self.index_path) + ".meta"

        if not Path(index_file).exists():
            return

        # Load FAISS index
        self.index = faiss.read_index(index_file)

        # Load metadata
        if Path(meta_file).exists():
            with open(meta_file, "rb") as f:
                metadata = pickle.load(f)
                self.dimension = metadata["dimension"]
                self.document_count = metadata["document_count"]

        print(
            f"Loaded FAISS index with {self.document_count} vectors from {self.index_path}"
        )

    def get_total_vectors(self) -> int:
        """Get total number of vectors in the index"""
        return self.document_count
