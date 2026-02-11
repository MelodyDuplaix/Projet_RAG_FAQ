import pytest
import os
import pandas as pd
from unittest.mock import MagicMock, patch
import torch
from src.services.rag_service import RAGService
from sentence_transformers import util

class TestRAGServiceFunctional:

    def test_rag_service_initialization_functional(self, functional_rag_service_instance):
        """
        Test that RAGService initializes correctly with mocked FAQ data and SentenceTransformer for functional tests.
        """
        service = functional_rag_service_instance
        assert not service.faq_df.empty
        assert len(service._faq_corpus) == len(service.faq_df)
        
        functional_rag_service_instance._embed_model.encode.assert_called_with(
            service._faq_corpus, convert_to_tensor=True
        )

    def test_find_context_functional(self, functional_rag_service_instance):
        """
        Test that _find_context returns relevant context and sources for a matching question.
        """
        service = functional_rag_service_instance
        question = "où puis-je avoir mon casier judiciaire ?"
        context, sources, confidence = service._find_context(question)

        assert "DIV001" in sources
        assert "casier judiciaire" in context
        assert "bulletin n°3" in context
        assert confidence > 0.0

        call_args_list = service._embed_model.encode.call_args_list
        assert len(call_args_list) >= 2
        assert call_args_list[1][0][0] == question


    def test_find_context_no_match_functional(self, functional_rag_service_instance):
        """
        Test that _find_context returns a low confidence for a question with no strong matching context.
        """
        service = functional_rag_service_instance
       
        question = "Question completement hors sujet."

        with patch("src.services.rag_service.util.cos_sim", return_value=torch.tensor([[0.0, 0.0, 0.0]])) as mock_cos_sim:
            context, sources, confidence = service._find_context(question)
            assert confidence < 0.1
            mock_cos_sim.assert_called_once()
        call_args_list = service._embed_model.encode.call_args_list
       
        user_question_encoded = False
        for args, kwargs in call_args_list:
            if args and args[0] == question:
                user_question_encoded = True
                break
        assert user_question_encoded


    def test_answer_question_functional(self, functional_rag_service_instance, mock_functional_llm_client, functional_mock_hf_token_env):
        """
        Test the end-to-end answer generation with mocked LLM but real context retrieval.
        """
        service = functional_rag_service_instance
        question = "Comment obtenir un acte de naissance ?"
        response = service.answer_question(question)
        assert response["answer"] == "Mocked LLM Answer based on context."
        assert "latency_ms" in response
        assert response["confidence"] > 0.0
        assert "EC001" in response["sources"]
        mock_functional_llm_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_functional_llm_client.chat.completions.create.call_args
        messages = kwargs["messages"]
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        assert "Tu es un assistant municipal expert" in system_prompt
        assert "CONTRAT DE NAISSANCE" not in system_prompt
       
        assert "Comment obtenir un acte de naissance ?" in system_prompt
        assert "Pour obtenir un acte de naissance, vous pouvez faire la demande en ligne sur le site service-public.fr." in system_prompt
        user_prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert user_prompt == question
