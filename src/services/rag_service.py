import os
import time
from functools import lru_cache
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util
import logging

from .data_loader import load_faq_data
from src.routes.metrics import REQUEST_COUNT, RESPONSE_TIME, CONFIDENCE_SCORE

load_dotenv()

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

logger = logging.getLogger("faq_api")

@lru_cache(maxsize=1)
def get_llm_client():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing from .env")
    return InferenceClient(token=token)

class RAGService:
    def __init__(
        self,
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=6, # Aligned with benchmark
        model_id=MODEL_ID,
    ):
        self.faq_df = load_faq_data()
        if self.faq_df.empty:
            raise ValueError("FAQ data is empty or could not be loaded.")
            
        self.embed_model_name = embed_model_name
        self.top_k = top_k
        self.model_id = model_id

        self._embed_model = SentenceTransformer(self.embed_model_name)

        self._faq_questions = self.faq_df["question"].fillna("").tolist()
        self._faq_answers = self.faq_df["answer"].fillna("").tolist()
        self._faq_ids = self.faq_df["id"].tolist()
        self._faq_categories = self.faq_df.get("category", pd.Series([""] * len(self.faq_df))).fillna("").tolist()
        self._faq_keywords = self.faq_df.get("keywords", pd.Series([[] for _ in range(len(self.faq_df))])).tolist()


        # Create a combined corpus for embedding, similar to the benchmark runner
        self._faq_corpus = []
        for q, a, c, kws in zip(
            self._faq_questions,
            self._faq_answers,
            self._faq_categories,
            self._faq_keywords,
        ):
            if isinstance(kws, list):
                kws_str = ", ".join(map(str, kws))
            else:
                kws_str = str(kws) if kws is not None else ""
            
            corpus_text = (
                f"Question: {q}\n"
                f"Réponse: {a}\n"
                f"Catégorie: {c}\n"
                f"Mots-clés: {kws_str}"
            )
            self._faq_corpus.append(corpus_text)

        self._faq_embeddings = self._embed_model.encode(
            self._faq_corpus,
            convert_to_tensor=True,
        )

    def _find_context(self, user_question):
        q_emb = self._embed_model.encode(user_question, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, self._faq_embeddings)[0]
        top_results = cos_scores.topk(k=min(self.top_k, len(self._faq_corpus)))

        context_chunks = []
        sources = []
        confidence_score = 0.0

        if top_results.values.numel() > 0:
            confidence_score = top_results.values[0].item()

        for score, idx in zip(top_results.values, top_results.indices):
            idx = int(idx)
            q = self._faq_questions[idx]
            a = self._faq_answers[idx]
            c = self._faq_categories[idx]
            kws = self._faq_keywords[idx]
            if isinstance(kws, list):
                kws_str = ", ".join(map(str, kws))
            else:
                kws_str = str(kws) if kws is not None else ""
            
            context_chunks.append(
                f"- Catégorie: {c}\n  Mots-clés: {kws_str}\n  Q: {q}\n  R: {a}"
            )
            sources.append(self._faq_ids[idx])

        context = "\n\n".join(context_chunks)
        return context, sources, confidence_score

    def answer_question(self, question):
        logger.info(f"Question received: '{question}'")
        start_time = time.perf_counter()
        
        try:
            context, sources, confidence = self._find_context(question)
            
            if not context:
                logger.warning(f"No context found for question: '{question}'")
                REQUEST_COUNT.labels(endpoint="/answer", status="no_context").inc()
                duration = time.perf_counter() - start_time
                RESPONSE_TIME.labels(strategy="default").observe(duration)
                CONFIDENCE_SCORE.observe(0.0) # No confidence if no context
                return {
                    "answer": "Je suis désolé, mais je n'ai pas trouvé d'informations pertinentes pour répondre à votre question.",
                    "confidence": 0.0,
                    "sources": [],
                    "latency_ms": duration * 1000,
                }

            client = get_llm_client()
            
            base_system_prompt = (
                "Tu es un assistant municipal expert de la communauté de communes Val de Loire Numérique.\n"
                "Ton but est de répondre en français EXCLUSIVEMENT aux questions sur les sujets de la FAQ fournie.\n"
                "Règles OBLIGATOIRES :\n"
                "- Si tu n'as pas suffisamment d'informations pour répondre, utilise la phrase: 'Bonjour, je suis désolé mais je ne suis pas en mesure de répondre à cette question.'\n"
                "- Sinon, commence toujours par 'Bonjour'.\n"
                "- Tu dois t'appuyer STRICTEMENT sur la FAQ fournie en contexte pour répondre. Ne mentionne JAMAIS la FAQ dans ta réponse."
            )

            final_system_prompt = (
                base_system_prompt
                + "\n\n--- CONTEXTE FAQ ---\n"
                + context
                + "\n--- FIN DU CONTEXTE ---"
            )
            
            messages = [
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": question},
            ]

            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
            )
            
            answer_text = completion.choices[0].message.content

            duration = time.perf_counter() - start_time
            latency_ms = duration * 1000

            REQUEST_COUNT.labels(endpoint="/answer", status="success").inc()
            RESPONSE_TIME.labels(strategy="default").observe(duration)
            CONFIDENCE_SCORE.observe(confidence)

            if confidence < 0.3:
                logger.warning(f"Low confidence detected for question: '{question}'. Confidence: {confidence:.2f}")

            logger.info(f"Answer generated in {latency_ms:.0f}ms with confidence {confidence:.2f}. Sources: {sources}")

            return {
                "answer": answer_text,
                "confidence": confidence,
                "sources": sources,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            duration = time.perf_counter() - start_time
            REQUEST_COUNT.labels(endpoint="/answer", status="error").inc()
            RESPONSE_TIME.labels(strategy="default").observe(duration)
            logger.error(f"Error answering question '{question}': {e}", exc_info=True)
            raise
