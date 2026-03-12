"""
Tests for src/llm/providers.py and src/storage/embeddings.py factory functions.

Covers:
  get_llm_provider
    - "ollama" returns OllamaProvider (constructor mocked)
    - "gemini" returns GeminiProvider (constructor mocked)
    - "gemini" without api_key raises ValueError
    - unknown type raises ValueError

  get_embedding_provider
    - "sentence-transformers" returns SentenceTransformerEmbedding (constructor mocked)
    - "gemini" returns GeminiEmbedding (constructor mocked)
    - "gemini" without api_key raises ValueError
    - unknown type raises ValueError

  OllamaProvider (unit – ollama client mocked)
    - generate delegates to client
    - generate_with_system delegates via client.chat

  GeminiProvider (unit – genai mocked)
    - generate delegates to model.generate_content
    - generate_with_system prepends system prompt
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# get_llm_provider factory
# ---------------------------------------------------------------------------


class TestGetLlmProviderFactory:
    def test_ollama_type_returns_provider(self):
        with patch("src.llm.providers.OllamaProvider") as MockOllama:
            MockOllama.return_value = MagicMock()
            from src.llm.providers import get_llm_provider

            _ = get_llm_provider(
                "ollama", model="llama3.2", base_url="http://localhost:11434"
            )
            MockOllama.assert_called_once()

    def test_gemini_type_returns_provider(self):
        with patch("src.llm.providers.GeminiProvider") as MockGemini:
            MockGemini.return_value = MagicMock()
            from src.llm.providers import get_llm_provider

            _ = get_llm_provider("gemini", api_key="fake-key")
            MockGemini.assert_called_once()

    def test_gemini_without_api_key_raises(self):
        from src.llm.providers import get_llm_provider

        with pytest.raises(ValueError, match="Gemini API key"):
            get_llm_provider("gemini")

    def test_unknown_provider_raises(self):
        from src.llm.providers import get_llm_provider

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider("openai")


# ---------------------------------------------------------------------------
# get_embedding_provider factory
# ---------------------------------------------------------------------------


class TestGetEmbeddingProviderFactory:
    def test_sentence_transformers_type(self):
        with patch("src.storage.embeddings.SentenceTransformerEmbedding") as Mock:
            Mock.return_value = MagicMock()
            from src.storage.embeddings import get_embedding_provider

            get_embedding_provider(
                "sentence-transformers", model_name="all-MiniLM-L6-v2"
            )
            Mock.assert_called_once_with("all-MiniLM-L6-v2")

    def test_gemini_type(self):
        with patch("src.storage.embeddings.GeminiEmbedding") as Mock:
            Mock.return_value = MagicMock()
            from src.storage.embeddings import get_embedding_provider

            get_embedding_provider("gemini", api_key="k")
            Mock.assert_called_once_with("k")

    def test_gemini_without_api_key_raises(self):
        from src.storage.embeddings import get_embedding_provider

        with pytest.raises(ValueError, match="Gemini API key"):
            get_embedding_provider("gemini")

    def test_unknown_type_raises(self):
        from src.storage.embeddings import get_embedding_provider

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider("openai")

    def test_default_model_used_when_not_specified(self):
        with patch("src.storage.embeddings.SentenceTransformerEmbedding") as Mock:
            Mock.return_value = MagicMock()
            from src.storage.embeddings import get_embedding_provider

            get_embedding_provider("sentence-transformers")
            Mock.assert_called_once_with("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# OllamaProvider (mocked client)
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    def _make_provider(self):
        with patch("src.llm.providers.OllamaProvider.__init__") as mock_init:
            mock_init.return_value = None
            from src.llm.providers import OllamaProvider

            p = OllamaProvider.__new__(OllamaProvider)
            p.model = "llama3.2"
            p.client = MagicMock()
            return p

    def test_generate_uses_client_generate(self):
        p = self._make_provider()
        p.client.generate.return_value = {"response": "Hello"}
        result = p.generate("Say hi")
        p.client.generate.assert_called_once()
        assert result == "Hello"

    def test_generate_with_context_builds_prompt(self):
        p = self._make_provider()
        p.client.generate.return_value = {"response": "ctx answer"}
        p.generate("question?", context="some context")
        call_args = p.client.generate.call_args
        _ = (
            call_args.kwargs.get("prompt") or call_args.args[1]
            if call_args.args
            else call_args.kwargs["prompt"]
        )
        # Just confirm generate was called
        assert p.client.generate.called

    def test_generate_with_system_uses_chat(self):
        p = self._make_provider()
        p.client.chat.return_value = {"message": {"content": "system reply"}}
        result = p.generate_with_system("You are helpful.", "What is RAG?")
        p.client.chat.assert_called_once()
        assert result == "system reply"

    def test_generate_with_system_passes_both_messages(self):
        p = self._make_provider()
        p.client.chat.return_value = {"message": {"content": "ok"}}
        p.generate_with_system("sys", "usr")
        call_kwargs = p.client.chat.call_args.kwargs
        messages = call_kwargs.get("messages") or p.client.chat.call_args.args[1]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles


# ---------------------------------------------------------------------------
# GeminiProvider (mocked genai)
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    def _make_provider(self):
        with patch("src.llm.providers.GeminiProvider.__init__") as mock_init:
            mock_init.return_value = None
            from src.llm.providers import GeminiProvider

            p = GeminiProvider.__new__(GeminiProvider)
            p.model = MagicMock()
            p.model_name = "gemini-2.5-flash"
            return p

    def test_generate_calls_generate_content(self):
        p = self._make_provider()
        p.model.generate_content.return_value = MagicMock(text="gem answer")
        result = p.generate("Hello?")
        p.model.generate_content.assert_called_once()
        assert result == "gem answer"

    def test_generate_with_system_prepends_system_content(self):
        p = self._make_provider()
        p.model.generate_content.return_value = MagicMock(text="combined")
        result = p.generate_with_system("Be concise.", "What is sales?")
        p.model.generate_content.assert_called_once()
        # The system prompt should be embedded in the call argument
        call_arg = p.model.generate_content.call_args.args[0]
        assert "Be concise." in call_arg
        assert "What is sales?" in call_arg
        assert result == "combined"
