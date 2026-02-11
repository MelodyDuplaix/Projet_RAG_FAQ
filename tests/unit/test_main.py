from fastapi.testclient import TestClient
import pytest

from src.main import app

def test_health_check(client): 
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "ok"
    assert "timestamp" in json_response
    assert "version" in json_response
    assert "faq_count" in json_response
