# frontend/tests/test_client.py
from unittest.mock import MagicMock, patch

import pytest
import requests

from client import generate_taco, health_check


def test_health_check_returns_dict():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "ok", "model_loaded": True, "active_model": "base"}
    mock_resp.raise_for_status.return_value = None
    with patch("client.requests.get", return_value=mock_resp) as mock_get:
        result = health_check()
    assert result["status"] == "ok"
    mock_get.assert_called_once_with("http://localhost:8000/health", timeout=5)


def test_health_check_raises_on_http_error():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
    with patch("client.requests.get", return_value=mock_resp):
        with pytest.raises(requests.HTTPError):
            health_check()


def test_generate_taco_returns_dict():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": {"name": "Test Taco"}, "metadata": {}}
    mock_resp.raise_for_status.return_value = None
    with patch("client.requests.post", return_value=mock_resp) as mock_post:
        result = generate_taco("high protein taco", "sess-123", model="base")
    assert result["data"]["name"] == "Test Taco"
    mock_post.assert_called_once_with(
        "http://localhost:8000/generate-taco",
        json={"message": "high protein taco", "session_id": "sess-123", "model": "base"},
        timeout=30,
    )


def test_generate_taco_raises_on_422():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("422 Unprocessable")
    with patch("client.requests.post", return_value=mock_resp):
        with pytest.raises(requests.HTTPError):
            generate_taco("bad prompt", "sess-123")


def test_generate_taco_raises_on_connection_error():
    with patch("client.requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(requests.ConnectionError):
            generate_taco("any prompt", "sess-123")
