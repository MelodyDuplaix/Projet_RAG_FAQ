import json
import os
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock, patch, mock_open
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.main import app
from src.services.rag_service import RAGService, get_llm_client
from src.services.data_loader import load_faq_data
from sentence_transformers import SentenceTransformer, util

@pytest.fixture(scope="function")
def client():
    return TestClient(app)

MOCK_FAQ_DATA_DF = pd.DataFrame([
    {"id": "1", "question": "Q1", "answer": "A1", "category": "Cat1", "theme": "Theme1", "tags": ["tag1", "tag2"]},
    {"id": "2", "question": "Q2", "answer": "A2", "category": "Cat2", "theme": "Theme2", "tags": ["tag3"]},
])

MOCK_FAQ_DATA = {
    "faq": [
        {"id": "1", "question": "Q1", "answer": "A1", "category": "Cat A", "keywords": ["kw1"]},
        {"id": "2", "question": "Q2", "answer": "A2", "category": "Specific", "keywords": ["kw2"]},
        {"id": "3", "question": "Q3", "answer": "A3", "category": "Another", "keywords": ["kw3"]},
    ]
}

MOCK_FAQ_DATA_NO_ID = {
    "faq": [
        {"question": "Q1", "answer": "A1"},
    ]
}

MOCK_FAQ_DATA_RAG_SERVICE_FUNCTIONAL = [
    {"id": "EC001", "question": "Comment obtenir un acte de naissance ?", "answer": "Pour obtenir un acte de naissance, vous pouvez faire la demande en ligne sur le site service-public.fr.", "category": "etat_civil", "keywords": ["naissance", "acte"]},
    {"id": "URB001", "question": "Comment déposer un permis de construire ?", "answer": "Le permis de construire se dépose à la mairie.", "category": "urbanisme", "keywords": ["permis", "construire"]},
    {"id": "DIV001", "question": "Comment obtenir un extrait de casier judiciaire ?", "answer": "Le bulletin n°3 du casier judiciaire se demande gratuitement en ligne.", "category": "etat_civil", "keywords": ["casier judiciaire", "extrait"]},
]

@pytest.fixture
def mock_data_loader_df():
    with patch("src.services.data_loader.load_faq_data") as mock_dl:
        mock_dl.return_value = MOCK_FAQ_DATA_DF
        yield mock_dl

@pytest.fixture
def mock_load_faq_data_success():
    with patch("builtins.open", new_callable=mock_open, read_data=json.dumps(MOCK_FAQ_DATA)) as mock_file:
        with patch("os.path.exists", return_value=True) as mock_exists:
            yield mock_file, mock_exists

@pytest.fixture
def mock_load_faq_data_invalid_json():
    with patch("builtins.open", new_callable=mock_open, read_data="invalid json") as mock_file:
        yield mock_file

@pytest.fixture
def mock_load_faq_data_no_id():
    with patch("builtins.open", new_callable=mock_open, read_data=json.dumps(MOCK_FAQ_DATA_NO_ID)) as mock_file:
        yield mock_file

@pytest.fixture
def mock_load_faq_data_rag():
    with patch("src.services.rag_service.load_faq_data") as mock_data_loader:
        df = pd.DataFrame(MOCK_FAQ_DATA["faq"])
        mock_data_loader.return_value = df
        yield mock_data_loader

@pytest.fixture
def mock_faq_data_for_rag_service():
    """Provides a consistent mock for load_faq_data for RAGService functional tests."""
    with patch("src.services.rag_service.load_faq_data") as mock_load_faq:
        mock_load_faq.return_value = pd.DataFrame(MOCK_FAQ_DATA_RAG_SERVICE_FUNCTIONAL)
        yield mock_load_faq

@pytest.fixture
def mock_sentence_transformer():
    """
    Mocks SentenceTransformer for unit tests.
    Provides dummy embeddings for 3 FAQ items and a user question.
    """
    with patch("src.services.rag_service.SentenceTransformer") as mock_st:
        mock_instance = MagicMock(spec=SentenceTransformer)
        mock_instance.encode.return_value = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])
        mock_st.return_value = mock_instance
        yield mock_st

@pytest.fixture
def mock_functional_sentence_transformer():
    """
    Mocks SentenceTransformer for functional tests.
    Provides dummy embeddings for MOCK_FAQ_DATA_RAG_SERVICE_FUNCTIONAL and user questions.
    The side_effect is a list of return values for successive calls to encode.
    """
    with patch("src.services.rag_service.SentenceTransformer") as MockSentenceTransformer:
        mock_embed_model = MagicMock(spec=SentenceTransformer)
        
        mock_embed_model.encode.side_effect = [
            util.normalize_embeddings(torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.9, 0.9, 0.9]])),
            util.normalize_embeddings(torch.tensor([[0.8, 0.8, 0.8]])),
            util.normalize_embeddings(torch.tensor([[0.15, 0.15, 0.15]])),
            util.normalize_embeddings(torch.tensor([[-1.0, -1.0, -1.0]])),
            util.normalize_embeddings(torch.tensor([[0.15, 0.15, 0.15]]))
        ]
        MockSentenceTransformer.return_value = mock_embed_model
        yield MockSentenceTransformer

@pytest.fixture
def mock_inference_client():
    with patch("src.services.rag_service.InferenceClient") as mock_client:
        mock_instance = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Mocked LLM Answer"))]
        mock_instance.chat.completions.create.return_value = mock_completion
        mock_client.return_value = mock_instance
        yield mock_client

@pytest.fixture
def mock_functional_llm_client():
    with patch("src.services.rag_service.get_llm_client") as mock_get_llm_client:
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Mocked LLM Answer based on context."))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_get_llm_client.return_value = mock_client
        yield mock_client



@pytest.fixture(autouse=True)
def reset_rag_service_instance_routes():
    from src.routes.api_router import _rag_service_instance
    _rag_service_instance = None
    yield

@pytest.fixture
def functional_rag_service_instance(mock_faq_data_for_rag_service, mock_functional_sentence_transformer):
    service = RAGService()
    yield service
