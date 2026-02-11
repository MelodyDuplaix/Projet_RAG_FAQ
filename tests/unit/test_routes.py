from fastapi.testclient import TestClient
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from fastapi import HTTPException

from src.main import app
from src.models import AnswerResponse, FAQ
from src.routes.api_router import get_rag_service, get_faq_df
from src.services.rag_service import RAGService

MOCK_FAQ_DATA_FOR_TESTS_DF = pd.DataFrame([
    {"id": "1", "question": "Q1", "answer": "A1", "category": "Cat1", "theme": "Theme1", "tags": ["tag1", "tag2"]},
    {"id": "2", "question": "Q2", "answer": "A2", "category": "Cat2", "theme": "Theme2", "tags": ["tag3"]},
])

@pytest.fixture()
def mock_get_faq_df_dependency():
    app.dependency_overrides[get_faq_df] = lambda: MOCK_FAQ_DATA_FOR_TESTS_DF
    yield
    app.dependency_overrides = {}

def test_get_answer(client, mock_data_loader_df, mock_get_faq_df_dependency):
    mock_rag_service = MagicMock(spec=RAGService)
    mock_rag_service.answer_question.return_value = {
        "answer": "Mocked LLM Answer",
        "confidence": 0.95,
        "sources": ["doc_mock"],
        "latency_ms": 100.0,
    }
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    try:
        response = client.post(
            "/api/v1/answer", json={"question": "Test question?"}
        )
        assert response.status_code == 200
        answer_response = AnswerResponse(**response.json())
        assert answer_response.answer == "Mocked LLM Answer"
        assert answer_response.confidence == 0.95
        assert "doc_mock" in answer_response.sources
        mock_rag_service.answer_question.assert_called_once_with( 
            "Test question?"
        )
    finally:
        app.dependency_overrides = {} 

def test_get_answer_internal_error(client, mock_data_loader_df, mock_get_faq_df_dependency):
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

def test_list_faqs(client, mock_data_loader_df, mock_get_faq_df_dependency):
   
    response = client.get("/api/v1/faq")
    assert response.status_code == 200
    faqs = [FAQ(**item) for item in response.json()]
    assert len(faqs) == 2
    assert faqs[0].id == "1"
    assert faqs[1].question == "Q2"

def test_list_faqs_empty(client, mock_data_loader_df):
    app.dependency_overrides[get_faq_df] = lambda: pd.DataFrame() 
    try:
        response = client.get("/api/v1/faq")
        assert response.status_code == 200
        assert response.json() == []
    finally:
        app.dependency_overrides = {}


def test_get_faq_by_id_success(client, mock_data_loader_df, mock_get_faq_df_dependency):
    response = client.get("/api/v1/faq/1")
    assert response.status_code == 200
    faq_item = FAQ(**response.json())
    assert faq_item.id == "1"
    assert faq_item.question == "Q1"

def test_get_faq_by_id_not_found(client, mock_data_loader_df, mock_get_faq_df_dependency):
    response = client.get("/api/v1/faq/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "FAQ with id '999' not found."}

def test_get_faq_by_id_data_not_available(client, mock_data_loader_df):
    app.dependency_overrides[get_faq_df] = lambda: pd.DataFrame() 
    try:
        response = client.get("/api/v1/faq/1")
        assert response.status_code == 404
        assert response.json() == {"detail": "FAQ data not available."}
    finally:
        app.dependency_overrides = {}
