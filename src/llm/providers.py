"""
LLM providers for generating responses
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, context: str = None) -> str:
        """Generate a response given a prompt and optional context"""
        raise NotImplementedError

    @abstractmethod
    def generate_with_system(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response with a system prompt"""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """LLM provider using Ollama (local)"""

    def __init__(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434"
    ):
        import ollama

        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        print(f"Initialized Ollama with model: {model}")

    def generate(self, prompt: str, context: str = None) -> str:
        """Generate a response given a prompt and optional context"""
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt

        response = self.client.generate(model=self.model, prompt=full_prompt)
        return response["response"]

    def generate_with_system(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response with a system prompt"""
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]


class GeminiProvider(LLMProvider):
    """LLM provider using Google Gemini"""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        print(f"Initialized Gemini with model: {model}")

    def generate(self, prompt: str, context: str = None) -> str:
        """Generate a response given a prompt and optional context"""
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt

        response = self.model.generate_content(full_prompt)
        return response.text

    def generate_with_system(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response with a system prompt"""
        # Gemini doesn't have explicit system prompts, so we prepend it
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.model.generate_content(full_prompt)
        return response.text


def get_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factory function to get the appropriate LLM provider"""
    if provider_type == "ollama":
        model = kwargs.get("model", "llama3.2")
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaProvider(model=model, base_url=base_url)
    if provider_type == "gemini":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")
        model = kwargs.get("model", "gemini-2.5-flash")
        return GeminiProvider(api_key=api_key, model=model)
    raise ValueError(f"Unknown LLM provider: {provider_type}")
