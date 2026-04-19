# frontend/client.py
"""
TacoLLM — HTTP client for the FastAPI backend.

Functions:
    health_check(base_url) -> dict
    generate_taco(message, session_id, model, base_url) -> dict
"""

from typing import Any, Dict

import requests

BACKEND_URL = "http://localhost:8000"


def health_check(base_url: str = BACKEND_URL) -> Dict[str, Any]:
    """GET /health — raises on non-2xx."""
    resp = requests.get(f"{base_url}/health", timeout=5)
    resp.raise_for_status()
    return resp.json()


def generate_taco(
    message: str,
    session_id: str,
    model: str = "base",
    base_url: str = BACKEND_URL,
) -> Dict[str, Any]:
    """POST /generate-taco — raises on non-2xx or connection error."""
    payload = {"message": message, "session_id": session_id, "model": model}
    resp = requests.post(f"{base_url}/generate-taco", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()
