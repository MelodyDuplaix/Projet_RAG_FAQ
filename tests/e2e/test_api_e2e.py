import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
import json

from src.main import app
from src.models import AnswerResponse, FAQ
from src.routes import get_rag_service, get_faq_df
from src.services.rag_service import RAGService
from src.services.data_loader import load_faq_data

from tests.conftest import MOCK_FAQ_DATA_RAG_SERVICE_FUNCTIONAL 

@pytest.fixture(autouse=True)
def setup_e2e_mocks(mock_functional_llm_client, mock_functional_sentence_transformer, mock_faq_data_for_rag_service, functional_mock_hf_token_env):
    """
    Sets up common mocks required for E2E tests, ensuring RAGService uses mocked LLM/embeddings
    and FAQ data while allowing the API routes to be tested end-to-end.
    """
    app.dependency_overrides[get_faq_df] = lambda: pd.DataFrame(MOCK_FAQ_DATA_RAG_SERVICE_FUNCTIONAL)
    yield
    app.dependency_overrides = {}

class TestApiE2E:

    def test_e2e_health_check(self, client):
        """
        Tests the /health endpoint to ensure the API is running.
        """
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_e2e_answer_endpoint(self, client, mock_functional_llm_client):
        """
        Tests the /api/v1/answer endpoint end-to-end.
        RAGService should be initialized with mocked SentenceTransformer and LLM client.
        """
        question = "Comment obtenir un acte de naissance ?"
        response = client.post(
            "/api/v1/answer",
            json={"question": question}
        )

        assert response.status_code == 200
        answer_response = AnswerResponse(**response.json())

        assert answer_response.answer == "Mocked LLM Answer based on context."
        assert answer_response.confidence > 0.0
        assert "EC001" in answer_response.sources 
        assert answer_response.latency_ms > 0.0
       
        mock_functional_llm_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_functional_llm_client.chat.completions.create.call_args
        messages = kwargs["messages"] 

        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        assert "Comment obtenir un acte de naissance ?" in system_prompt
        assert "Pour obtenir un acte de naissance, vous pouvez faire la demande en ligne sur le site service-public.fr." in system_prompt

        user_prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert user_prompt == question

    def test_e2e_list_faqs_endpoint(self, client):
        """
        Tests the /api/v1/faq endpoint to list all FAQs.
        """
        response = client.get("/api/v1/faq")
        assert response.status_code == 200
        faqs = [FAQ(**item) for item in response.json()]

        assert len(faqs) == 3
        assert faqs[0].id == "EC001"
        assert faqs[1].question == "Comment déposer un permis de construire ?"
        assert faqs[2].category == "etat_civil"

    def test_e2e_get_faq_by_id_endpoint(self, client):
        """
        Tests the /api/v1/faq/{faq_id} endpoint to retrieve a specific FAQ by ID.
        """
        faq_id = "URB001"
        response = client.get(f"/api/v1/faq/{faq_id}")
        assert response.status_code == 200
        faq_item = FAQ(**response.json())

        assert faq_item.id == faq_id
        assert faq_item.question == "Comment déposer un permis de construire ?"
        assert faq_item.answer == "Le permis de construire se dépose à la mairie."

    def test_e2e_get_faq_by_id_not_found(self, client):
        """
        Tests the /api/v1/faq/{faq_id} endpoint for a non-existent ID.
        """
        faq_id = "NON_EXISTENT"
        response = client.get(f"/api/v1/faq/{faq_id}")
        assert response.status_code == 404
        assert response.json() == {"detail": f"FAQ with id '{faq_id}' not found."}

    def test_e2e_answer_endpoint_no_context(self, client, mock_functional_llm_client):
        """
        Tests the /api/v1/answer endpoint when no relevant context is found by RAGService.
        """
        question = "What is the meaning of life?"
        
       
        with patch.object(RAGService, '_find_context', return_value=("", [], 0.0)):
            response = client.post(
                "/api/v1/answer",
                json={"question": question}
            )

            assert response.status_code == 200
            answer_response = AnswerResponse(**response.json())

            assert "Je suis désolé" in answer_response.answer
            assert answer_response.confidence == 0.0
            assert answer_response.sources == []
            assert answer_response.latency_ms > 0.0
            
            mock_functional_llm_client.chat.completions.create.assert_not_called()