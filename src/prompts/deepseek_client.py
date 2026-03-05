"""
Cliente mínimo para chamadas ao DeepSeek (usado por signal_reevaluator e stop_adjuster).
Usa DEEPSEEK_API_KEY e agno.
"""
import os
from typing import Optional

try:
    from agno.models.deepseek import DeepSeek
    from agno.agent import Agent
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False

from src.core.logger import get_logger
logger = get_logger(__name__)


class DeepSeekClient:
    """Cliente para chat com DeepSeek (API key via env)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY não configurada")
        if not AGNO_AVAILABLE:
            raise ImportError("agno não instalado")

    async def __aenter__(self):
        self._model = DeepSeek(id="deepseek-chat", api_key=self.api_key, temperature=0.3, max_tokens=1000)
        self._agent = Agent(model=self._model)
        return self

    async def __aexit__(self, *args):
        pass

    async def chat(self, prompt: str) -> str:
        """Envia prompt e retorna resposta em texto."""
        try:
            response = await self._agent.arun(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.exception("DeepSeek chat error: %s", e)
            return ""
