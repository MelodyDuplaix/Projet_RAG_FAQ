import time
from abc import ABC, abstractmethod
import os
from functools import lru_cache

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import pandas as pd
from sentence_transformers import SentenceTransformer, util  # pour RAG / QA extractif

load_dotenv()

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
EXTRACTIVE_QA_MODEL_ID = "AgentPublic/camembert-base-squadFR-fquad-piaf"


@lru_cache(maxsize=1)
def get_client():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN manquant dans .env")
    return InferenceClient()


@lru_cache(maxsize=1)
def get_hf_inference_client():
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("HF_TOKEN manquant dans .env pour Inference Providers")
    return InferenceClient(provider="hf-inference", api_key=api_key)


class BaseMethodRunner(ABC):
    def __init__(
        self,
        question_col="question",
        answer_col="answer_model",
        latency_col="latency_seconds",
    ):
        self.question_col = question_col
        self.answer_col = answer_col
        self.latency_col = latency_col

    @abstractmethod
    def answer_one(self, question, system_prompt=None):
        raise NotImplementedError

    def run_on_dataframe(self, df, system_prompt=None, delay_seconds=0.0):
        answers = []
        latencies = []

        for idx, row in df.iterrows():
            question = row[self.question_col]
            print(f"[{idx}] Inférence sur la question : {question!r}")

            start = time.perf_counter()
            try:
                response = self.answer_one(question, system_prompt=system_prompt)
            except Exception as e:
                print(f"Erreur pendant l'inférence sur la ligne {idx}: {e}")
                response = None
            end = time.perf_counter()

            answers.append(response)
            latencies.append(end - start)

            if delay_seconds and delay_seconds > 0:
                time.sleep(delay_seconds)

        df = df.copy()
        df[self.answer_col] = answers
        df[self.latency_col] = latencies
        return df


class LLMOnlyRunner(BaseMethodRunner):
    def __init__(self, model_id=MODEL_ID, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id

    def answer_one(self, question, system_prompt=None):
        client = get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        completion = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9,
        )

        return completion.choices[0].message.content


class RAGRunner(BaseMethodRunner):
    def __init__(
        self,
        faq_df,
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=6,
        model_id=MODEL_ID,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.faq_df = faq_df
        self.embed_model_name = embed_model_name
        self.top_k = top_k
        self.model_id = model_id

        self._embed_model = SentenceTransformer(self.embed_model_name)

        self._faq_questions = faq_df["question"].fillna("").tolist()
        self._faq_answers = faq_df["answer"].fillna("").tolist()
        self._faq_categories = faq_df.get("category", "").fillna("").tolist()
        self._faq_keywords = faq_df.get("keywords", "").tolist()

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

    def _build_context(self, user_question):
        q_emb = self._embed_model.encode(user_question, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, self._faq_embeddings)[0]
        top_results = cos_scores.topk(k=min(self.top_k, len(self._faq_corpus)))

        context_chunks = []
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

        context = "\n\n".join(context_chunks)
        return context

    def answer_one(self, question, system_prompt=None):
        client = get_client()

        context = self._build_context(question)

        rag_system_prompt = (
            (system_prompt or "")
            + "\n\nTu disposes de la FAQ suivante comme contexte :\n"
            + context
            + "\n\nUtilise uniquement ces informations pour répondre à la question."
        )

        messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": question},
        ]

        completion = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9,
        )

        return completion.choices[0].message.content


class ExtractiveQARunner(BaseMethodRunner):
    def __init__(
        self,
        faq_df,
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=10,
        qa_model_id=EXTRACTIVE_QA_MODEL_ID,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.faq_df = faq_df
        self.embed_model_name = embed_model_name
        self.top_k = top_k
        self.qa_model_id = qa_model_id

        self._embed_model = SentenceTransformer(self.embed_model_name)

        self._faq_questions = faq_df["question"].fillna("").tolist()
        self._faq_answers = faq_df["answer"].fillna("").tolist()
        self._faq_categories = faq_df.get("category", "").fillna("").tolist()
        self._faq_keywords = faq_df.get("keywords", "").tolist()

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

        self._hf_inference_client = get_hf_inference_client()

    def _build_context(self, user_question):
        q_emb = self._embed_model.encode(user_question, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, self._faq_embeddings)[0]
        top_results = cos_scores.topk(k=min(self.top_k, len(self._faq_corpus)))

        context_chunks = []
        for score, idx in zip(top_results.values, top_results.indices):
            idx = int(idx)
            a = self._faq_answers[idx]
            context_chunks.append(a)

        context = "\n\n".join(context_chunks)
        return context

    def answer_one(self, question, system_prompt=None):
        context = self._build_context(question)

        if not context.strip():
            return "Je ne dispose pas d'éléments suffisants dans la base pour répondre précisément."

        answer_obj = self._hf_inference_client.question_answering(
            question=question,
            context=context,
            model=self.qa_model_id,
            align_to_words=True,
            handle_impossible_answer=True,
            max_answer_len=512
        )

        if isinstance(answer_obj, dict):
            answer = answer_obj.get("answer") or "Je ne suis pas en mesure de répondre à cette question."
        else:
            answer = str(answer_obj)

        return answer.strip()
