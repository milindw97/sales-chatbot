"""
Tests for src/storage/vector_store.py

Covers:
  - Fresh store starts empty
  - add_vectors: shape, returned indices, document_count increment
  - search: correct k, empty-store early-return, distance ordering
  - save/load round-trip (persists vectors and count)
  - load skipped when .index file absent (no crash)
  - get_total_vectors reflects current state
  - dimension mismatch raises ValueError
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

DIM = 8


def random_vectors(n: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype("float32")


def random_query(dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(dim).astype("float32")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInitialisation:
    def test_starts_empty(self, tmp_vector_store):
        assert tmp_vector_store.get_total_vectors() == 0

    def test_dimension_stored(self, tmp_vector_store):
        assert tmp_vector_store.dimension == DIM

    def test_load_skipped_when_no_index_file(self, tmp_path):
        """Constructing a store with a non-existent index should not crash."""
        store = FAISSVectorStore(dimension=DIM, index_path=str(tmp_path / "missing"))
        assert store.get_total_vectors() == 0

    def test_no_index_path_ok(self):
        store = FAISSVectorStore(dimension=DIM)
        assert store.get_total_vectors() == 0


# ---------------------------------------------------------------------------
# add_vectors
# ---------------------------------------------------------------------------


class TestAddVectors:
    def test_returns_correct_indices(self, tmp_vector_store):
        vectors = random_vectors(5)
        indices = tmp_vector_store.add_vectors(vectors)
        assert indices == list(range(5))

    def test_count_increases(self, tmp_vector_store):
        tmp_vector_store.add_vectors(random_vectors(3))
        assert tmp_vector_store.get_total_vectors() == 3

    def test_sequential_adds_accumulate(self, tmp_vector_store):
        tmp_vector_store.add_vectors(random_vectors(2))
        tmp_vector_store.add_vectors(random_vectors(3))
        assert tmp_vector_store.get_total_vectors() == 5

    def test_second_batch_indices_continue_from_first(self, tmp_vector_store):
        tmp_vector_store.add_vectors(random_vectors(3))
        indices = tmp_vector_store.add_vectors(random_vectors(2))
        assert indices == [3, 4]

    def test_1d_vector_accepted(self, tmp_vector_store):
        vec = random_vectors(1)[0]  # shape (DIM,)
        indices = tmp_vector_store.add_vectors(vec)
        assert len(indices) == 1

    def test_wrong_dimension_raises(self, tmp_vector_store):
        wrong = np.ones((3, DIM + 1), dtype="float32")
        with pytest.raises(ValueError, match="dimension"):
            tmp_vector_store.add_vectors(wrong)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_empty_store_returns_empty(self, tmp_vector_store):
        distances, indices = tmp_vector_store.search(random_query(), k=5)
        assert len(distances) == 0
        assert len(indices) == 0

    def test_search_k_capped_to_available(self, tmp_vector_store):
        tmp_vector_store.add_vectors(random_vectors(3))
        _, indices = tmp_vector_store.search(random_query(), k=10)
        assert len(indices) == 3

    def test_search_returns_k_results(self, tmp_vector_store):
        tmp_vector_store.add_vectors(random_vectors(10))
        _, indices = tmp_vector_store.search(random_query(), k=4)
        assert len(indices) == 4

    def test_exact_vector_has_zero_distance(self, tmp_vector_store):
        vectors = random_vectors(5)
        tmp_vector_store.add_vectors(vectors)
        # Search with the exact first vector
        distances, indices = tmp_vector_store.search(vectors[0], k=1)
        assert distances[0] == pytest.approx(0.0, abs=1e-4)
        assert int(indices[0]) == 0

    def test_distances_non_negative(self, tmp_vector_store):
        tmp_vector_store.add_vectors(random_vectors(5))
        distances, _ = tmp_vector_store.search(random_query(), k=3)
        assert all(d >= 0 for d in distances)

    def test_results_sorted_by_distance(self, tmp_vector_store):
        tmp_vector_store.add_vectors(random_vectors(10))
        distances, _ = tmp_vector_store.search(random_query(), k=5)
        assert list(distances) == sorted(distances)

    def test_1d_query_accepted(self, tmp_vector_store):
        tmp_vector_store.add_vectors(random_vectors(3))
        q = random_query()  # shape (DIM,)
        _, indices = tmp_vector_store.search(q, k=2)
        assert len(indices) == 2


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_index_and_meta_files(self, tmp_vector_store):
        tmp_vector_store.add_vectors(random_vectors(4))
        tmp_vector_store.save()
        base = str(tmp_vector_store.index_path)
        assert Path(base + ".index").exists()
        assert Path(base + ".meta").exists()

    def test_loaded_store_has_same_count(self, tmp_path):
        index_path = str(tmp_path / "idx")
        store = FAISSVectorStore(dimension=DIM, index_path=index_path)
        store.add_vectors(random_vectors(7))
        store.save()

        loaded = FAISSVectorStore(dimension=DIM, index_path=index_path)
        assert loaded.get_total_vectors() == 7

    def test_loaded_store_searchable(self, tmp_path):
        index_path = str(tmp_path / "idx")
        vectors = random_vectors(5)
        store = FAISSVectorStore(dimension=DIM, index_path=index_path)
        store.add_vectors(vectors)
        store.save()

        loaded = FAISSVectorStore(dimension=DIM, index_path=index_path)
        distances, indices = loaded.search(vectors[0], k=1)
        assert int(indices[0]) == 0
        assert distances[0] == pytest.approx(0.0, abs=1e-4)

    def test_save_without_path_raises(self):
        store = FAISSVectorStore(dimension=DIM)  # no path
        store.add_vectors(random_vectors(2))
        with pytest.raises(ValueError):
            store.save()

    def test_load_idempotent_on_nonexistent_file(self, tmp_path):
        """load() when .index is absent should be a no-op."""
        store = FAISSVectorStore(dimension=DIM, index_path=str(tmp_path / "ghost"))
        store.load()  # should not raise
        assert store.get_total_vectors() == 0
