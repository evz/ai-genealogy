"""Ollama API utilities for model management and querying"""

import logging
import os

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, host: str | None = None, port: int | None = None, timeout: int = 120):
        self.host = host or os.getenv("OLLAMA_HOST", "localhost")
        self.port = port or int(os.getenv("OLLAMA_PORT", "11434"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            logger.error(f"Failed to fetch models: {response.status_code}")
            return []
        except requests.RequestException as e:
            logger.exception(f"Error connecting to Ollama: {e}")
            return []

    def get_model_choices(self) -> list[tuple[str, str]]:
        """Get model choices formatted for Django choice field"""
        models = self.list_models()
        choices = []

        for model in models:
            name = model.get("name", "")
            if name:
                # Create display name with size info if available
                size = model.get("size", 0)
                if size:
                    size_gb = size / (1024**3)
                    display_name = f"{name} ({size_gb:.1f}GB)"
                else:
                    display_name = name
                choices.append((name, display_name))

        # Sort by name
        choices.sort(key=lambda x: x[0])
        return choices

    def get_embedding_models(self) -> list[tuple[str, str]]:
        """Get models suitable for embeddings"""
        all_models = self.get_model_choices()

        # Filter for embedding models (common patterns)
        embedding_patterns = [
            "embed",
            "embedding",
            "e5",
            "nomic",
            "bge",
            "sentence",
            "multilingual",
            "all-minilm",
            "all-mpnet",
        ]

        embedding_models = []
        for name, display in all_models:
            name_lower = name.lower()
            if any(pattern in name_lower for pattern in embedding_patterns):
                embedding_models.append((name, display))

        return embedding_models

    def get_llm_models(self) -> list[tuple[str, str]]:
        """Get models suitable for text generation"""
        all_models = self.get_model_choices()

        # Filter out embedding models
        embedding_patterns = [
            "embed",
            "embedding",
            "e5",
            "nomic-embed",
            "bge",
            "sentence",
        ]

        llm_models = []
        for name, display in all_models:
            name_lower = name.lower()
            if not any(pattern in name_lower for pattern in embedding_patterns):
                llm_models.append((name, display))

        return llm_models

    def generate(self, model: str, prompt: str, **kwargs) -> str | None:
        """Generate text using specified model"""
        try:
            payload = {"model": model, "prompt": prompt, "stream": False, **kwargs}

            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            logger.error(f"Generation failed: {response.status_code} - {response.text}")
            return None

        except requests.RequestException as e:
            logger.exception(f"Error generating text: {e}")
            return None

    def embed(self, model: str, input_text: str) -> list[float] | None:
        """Generate embeddings using specified model"""
        try:
            payload = {"model": model, "input": input_text}

            response = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                return result.get("embeddings", [None])[0]
            logger.error(f"Embedding failed: {response.status_code} - {response.text}")
            return None

        except requests.RequestException as e:
            logger.exception(f"Error generating embedding: {e}")
            return None


def get_default_models() -> dict[str, str]:
    """Get default model configuration from environment"""
    return {
        "llm_model": os.getenv("OLLAMA_LLM_MODEL", "aya:35b-23"),
        "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", "zylonai/multilingual-e5-large:latest"),
    }
