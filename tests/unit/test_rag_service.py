import os
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import torch
from src.services.rag_service import RAGService, get_llm_client

def test_rag_service_initialization(mock_load_faq_data_rag, mock_sentence_transformer):
    service = RAGService()
    assert not service.faq_df.empty
    assert service.embed_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert service.top_k == 6
    mock_load_faq_data_rag.assert_called_once()
    mock_sentence_transformer.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
    assert len(service._faq_corpus) == 3
    assert service._faq_embeddings is not None

def test_rag_service_initialization_empty_faq_data(mock_sentence_transformer):
    with patch("src.services.rag_service.load_faq_data") as mock_data_loader:
        mock_data_loader.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="FAQ data is empty or could not be loaded."):
            RAGService()

def test_find_context(mock_load_faq_data_rag, mock_sentence_transformer):
    service = RAGService()
    service._embed_model.encode.return_value = torch.tensor([[0.7, 0.8, 0.9]])

    context, sources, confidence = service._find_context("User question 3")

    assert "Catégorie: Cat A" in context
    assert "Mots-clés: kw3" in context
    assert "Q: Q3" in context
    assert "R: A3" in context
    assert "3" in sources
    assert confidence > 0.0 

def test_answer_question_with_context(mock_load_faq_data_rag, mock_sentence_transformer, mock_inference_client):
    service = RAGService()
    service._embed_model.encode.return_value = torch.tensor([[0.7, 0.8, 0.9]])
    
    with patch("src.services.rag_service.get_llm_client", return_value=mock_inference_client.return_value):
        response = service.answer_question("User question with context")

        assert response["answer"] == "Mocked LLM Answer"
        assert response["confidence"] > 0.0
        assert "3" in response["sources"]
        assert "latency_ms" in response

def test_answer_question_no_context(mock_load_faq_data_rag, mock_sentence_transformer, mock_inference_client):
    service = RAGService()
    with patch.object(service, '_find_context', return_value=("", [], 0.0)):
        response = service.answer_question("Completely unrelated question")

        assert "Je suis désolé" in response["answer"]
        assert response["confidence"] == 0.0
        assert response["sources"] == []
        assert "latency_ms" in response

def test_get_llm_client_no_token(mock_hf_token): 
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="HF_TOKEN missing from .env"):
            get_llm_client.cache_clear()
            get_llm_client()

def test_get_llm_client_with_token(mock_inference_client, mock_hf_token):
    with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
        get_llm_client.cache_clear()
        client = get_llm_client()
        assert client is not None
        mock_inference_client.assert_called_once_with(token="test_token")