from fastapi.testclient import TestClient
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from fastapi import HTTPException

from src.main import app
from src.models import AnswerResponse, FAQ
from src.routes import get_rag_service


def test_get_answer(client, mock_rag_service_routes, mock_data_loader_df):
    response = client.post(
        "/api/v1/answer", json={"question": "Test question?"}
    )
    assert response.status_code == 200
    answer_response = AnswerResponse(**response.json())
    assert answer_response.answer == "Mocked Answer"
    assert answer_response.confidence == 0.95
    assert "doc_mock" in answer_response.sources
    mock_rag_service_routes.return_value.answer_question.assert_called_once_with(
        "Test question?"
    )

def test_get_answer_internal_error(client, mock_data_loader_df):
    mock_rag_service_instance = MagicMock()
    mock_rag_service_instance.answer_question.side_effect = HTTPException(status_code=500, detail="Internal error")

    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service_instance
    try:
        response = client.post(
            "/api/v1/answer", json={"question": "Test question?"}
        )
        assert response.status_code == 500
        assert response.json() == {"detail": "500: Internal error"}
    finally:
        app.dependency_overrides = {}

def test_list_faqs(client, mock_data_loader_df):
   
    response = client.get("/api/v1/faq")
    assert response.status_code == 200
    faqs = [FAQ(**item) for item in response.json()]
    assert len(faqs) == 2
    assert faqs[0].id == "1"
    assert faqs[1].question == "Q2"

def test_list_faqs_empty(client, mock_data_loader_df):
    mock_data_loader_df.return_value = pd.DataFrame()
    response = client.get("/api/v1/faq")
    assert response.status_code == 200
    assert response.json() == []

def test_get_faq_by_id_success(client, mock_data_loader_df):
    response = client.get("/api/v1/faq/1")
    assert response.status_code == 200
    faq_item = FAQ(**response.json())
    assert faq_item.id == "1"
    assert faq_item.question == "Q1"

def test_get_faq_by_id_not_found(client, mock_data_loader_df):
    response = client.get("/api/v1/faq/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "FAQ with id '999' not found."}

def test_get_faq_by_id_data_not_available(client, mock_data_loader_df):
    mock_data_loader_df.return_value = pd.DataFrame()
    response = client.get("/api/v1/faq/1")
    assert response.status_code == 404
    assert response.json() == {"detail": "FAQ data not available."}
