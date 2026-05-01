"""
llm_service.py — Ollama LLM integration — replaces Gemini/OpenAI entirely
Default model: phi3 (configurable via OLLAMA_MODEL in .env)
Note: First call is slow on CPU (~2-5 min cold start for phi3). Be patient.
"""

import requests
from core.config import settings


def ask_llm(context: str, query: str) -> str:
    """
    Send prompt to local Ollama instance and return the answer.
    Falls back gracefully if Ollama is not running.
    """
    prompt = (
        "Use ONLY the given context to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer clearly and cite sources."
    )

    payload = {
        "model":  settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 512,
            "temperature": 0.1,
            "num_ctx": 2048
        }
    }

    try:
        url  = f"{settings.OLLAMA_BASE_URL}/api/generate"
        resp = requests.post(url, json=payload, timeout=300)  # 5 min — phi3 cold-start on CPU is slow
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "⚠️  Ollama is not running. Start it with `ollama serve` and "
            f"pull the model: `ollama pull {settings.OLLAMA_MODEL}`"
        )
    except Exception as e:
        return f"⚠️  LLM error: {e}"
