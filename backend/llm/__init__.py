from .base import LLMProvider, ProviderEvent
from .claude import ClaudeProvider
from .gemini import GeminiProvider

__all__ = ["LLMProvider", "ProviderEvent", "ClaudeProvider", "GeminiProvider"]


def get_provider(name: str) -> LLMProvider:
    name = (name or "claude").lower()
    if name == "claude":
        return ClaudeProvider()
    if name == "gemini":
        return GeminiProvider()
    raise ValueError(f"Unknown provider: {name}")
