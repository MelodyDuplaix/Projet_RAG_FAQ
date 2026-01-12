# 📚 Ressources Pédagogiques - Projet FAQ Intelligent

Ce document recense les ressources essentielles pour mener à bien le projet. Consultez-les selon vos besoins et le planning proposé.

---

## 🎯 Parcours d'Apprentissage Recommandé

| Phase | Jours | Focus | Ressources prioritaires |
|-------|-------|-------|------------------------|
| Cadrage & Veille | J1-J2 | Comprendre RAG, embeddings, Q&A | Sections 1, 2, 3 |
| Implémentation | J3-J5 | Coder les 3 stratégies | Sections 2, 3, 4 |
| API | J7 | Développer l'API REST | Section 5 |
| Tests & CI/CD | J8-J9 | Industrialiser | Section 6 |

---

## 1. 🤗 HuggingFace - API Inference

L'API Inference de HuggingFace permet d'utiliser des modèles de ML sans infrastructure locale.

### Documentation officielle

| Ressource | Description | Lien |
|-----------|-------------|------|
| **Serverless Inference API** | Documentation principale de l'API gratuite | [huggingface.co/docs/api-inference](https://huggingface.co/docs/api-inference/index) |
| **Getting Started** | Guide de démarrage rapide | [huggingface.co/inference/get-started](https://huggingface.co/inference/get-started) |
| **Inference Providers** | Guide des providers d'inférence unifiés | [huggingface.co/docs/inference-providers](https://huggingface.co/docs/inference-providers/en/index) |
| **API Reference - Tasks** | Référence des tâches (chat, embeddings, Q&A...) | [huggingface.co/docs/inference-providers/tasks](https://huggingface.co/docs/inference-providers/en/tasks/index) |

### Tutoriels pratiques

| Ressource | Description | Lien |
|-----------|-------------|------|
| **Serverless Inference Cookbook** | Notebook avec exemples complets | [huggingface.co/learn/cookbook](https://huggingface.co/learn/cookbook/en/enterprise_hub_serverless_inference_api) |
| **How to Use HuggingFace API** | Tutoriel pas à pas (GeeksforGeeks) | [geeksforgeeks.org](https://www.geeksforgeeks.org/deep-learning/how-to-use-hugging-face-api/) |

### ⚠️ Important : API Chat vs Text Generation

Les modèles comme **Mistral** utilisent l'API **chat** (conversational) et non `text_generation`. Utilisez `chat_completion()` :

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="votre_token")

# ✅ Bonne méthode : chat_completion (pour Mistral, Zephyr, etc.)
messages = [
    {"role": "system", "content": "Tu es un assistant utile."},
    {"role": "user", "content": "Bonjour, comment puis-je vous aider ?"}
]

response = client.chat_completion(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=messages,
    max_tokens=100
)

print(response.choices[0].message.content)
```

> ⚠️ **Ne pas utiliser** `text_generation()` avec les modèles Mistral - cela provoquera une erreur "Model not supported for task text-generation".

---

## 2. 🔍 Sentence-Transformers et Recherche Sémantique

Les sentence-transformers convertissent du texte en vecteurs (embeddings) pour la recherche sémantique.

### Documentation officielle

| Ressource | Description | Lien |
|-----------|-------------|------|
| **GitHub officiel** | Repository avec documentation complète | [github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) |
| **Semantic Search Guide** | Guide officiel de la recherche sémantique | [sbert.net/semantic-search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) |
| **HuggingFace Hub** | Tous les modèles sentence-transformers | [huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers) |

### Modèles recommandés

| Modèle | Dimensions | Vitesse | Usage |
|--------|------------|---------|-------|
| **all-MiniLM-L6-v2** | 384 | ⚡ Rapide | Recommandé pour ce projet |
| all-mpnet-base-v2 | 768 | Moyen | Plus précis, plus lent |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | Moyen | Multilingue |

**Model Card all-MiniLM-L6-v2** : [huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Exemple de code minimal

```python
from sentence_transformers import SentenceTransformer, util

# Charger le modèle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encoder des phrases
corpus = ["Comment obtenir un acte de naissance ?", "Où déposer mes déchets verts ?"]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Recherche sémantique
query = "Je voudrais un extrait de naissance"
query_embedding = model.encode(query, convert_to_tensor=True)

# Calculer les similarités
similarities = util.cos_sim(query_embedding, corpus_embeddings)
print(similarities)  # La première phrase sera la plus similaire
```

### Tutoriels complémentaires

| Ressource | Description | Lien |
|-----------|-------------|------|
| **Semantic Search Tutorial** | Tutoriel pratique FAQ (DZone) | [dzone.com/articles/sentence-transformers-semantic-search-tutorial](https://dzone.com/articles/sentence-transformers-semantic-search-tutorial) |
| **Step-by-Step Guide** | Guide détaillé avec visualisations (Medium) | [medium.com/@hassanqureshi700](https://medium.com/@hassanqureshi700/a-step-by-step-guide-to-similarity-and-semantic-search-using-sentence-transformers-7091723a7bf9) |

---

## 3. 🧠 RAG (Retrieval-Augmented Generation)

Le RAG combine recherche d'information et génération pour des réponses plus précises.

### Comprendre le concept

| Ressource | Description | Lien |
|-----------|-------------|------|
| **What is RAG? (AWS)** | Introduction claire et complète | [aws.amazon.com/what-is/retrieval-augmented-generation](https://aws.amazon.com/what-is/retrieval-augmented-generation/) |
| **Introduction to RAG (Weaviate)** | Article détaillé avec architecture | [weaviate.io/blog/introduction-to-rag](https://weaviate.io/blog/introduction-to-rag) |
| **RAG Explained (Pinecone)** | Guide pratique avec exemples de prompts | [pinecone.io/learn/retrieval-augmented-generation](https://www.pinecone.io/learn/retrieval-augmented-generation/) |

### Architecture RAG simplifiée

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Question   │────▶│  Recherche   │────▶│  Contexte   │
│ utilisateur │     │  sémantique  │     │  récupéré   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐            │
                    │   Réponse    │◀───────────┘
                    │   générée    │     ┌──────────────┐
                    └──────────────┘◀────│     LLM      │
                                         └──────────────┘
```

### Exemple de code RAG simplifié

```python
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient

# 1. Recherche sémantique
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faq_embeddings = embedding_model.encode(faq_texts, convert_to_tensor=True)

question = "Comment obtenir un acte de naissance ?"
q_emb = embedding_model.encode(question, convert_to_tensor=True)
similarities = util.cos_sim(q_emb, faq_embeddings)[0]
top_indices = similarities.argsort(descending=True)[:3]

# 2. Construction du contexte
context = "\n".join([faq_list[i]["answer"] for i in top_indices])

# 3. Génération avec LLM
client = InferenceClient(token="votre_token")
messages = [
    {"role": "system", "content": "Réponds en te basant sur le contexte fourni."},
    {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {question}"}
]
response = client.chat_completion(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=messages,
    max_tokens=300
)
print(response.choices[0].message.content)
```

### Ressources avancées

| Ressource | Description | Lien |
|-----------|-------------|------|
| **RAG for LLMs** | Techniques avancées (Prompt Engineering Guide) | [promptingguide.ai/research/rag](https://www.promptingguide.ai/research/rag) |
| **Introduction to RAG (Coursera)** | Projet guidé 2h | [coursera.org/projects/introduction-to-rag](https://www.coursera.org/projects/introduction-to-rag) |

---

## 4. ❓ Question-Answering Extractif

Le Q&A extractif extrait directement la réponse d'un contexte donné.

### Documentation HuggingFace

| Ressource | Description | Lien |
|-----------|-------------|------|
| **Task: Question Answering** | Page de tâche avec démo interactive | [huggingface.co/tasks/question-answering](https://huggingface.co/tasks/question-answering) |
| **Q&A Tutorial** | Tutoriel complet fine-tuning sur SQuAD | [huggingface.co/docs/transformers/tasks/question_answering](https://huggingface.co/docs/transformers/tasks/question_answering) |
| **HuggingFace Course - Chapter 7** | Chapitre gratuit sur le Q&A | [huggingface.co/course/chapter7/7](https://huggingface.co/course/chapter7/7) |

### Modèles recommandés

| Modèle | Langue | Performance | Lien |
|--------|--------|-------------|------|
| **deepset/roberta-base-squad2** | Anglais | F1: 82.9% | [huggingface.co/deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2) |
| etalab-ia/camembert-base-squadFR-fquad-piaf | Français | Bon | [huggingface.co/etalab-ia/camembert-base-squadFR-fquad-piaf](https://huggingface.co/etalab-ia/camembert-base-squadFR-fquad-piaf) |

### Exemple de code minimal

```python
from transformers import pipeline

# Charger le pipeline Q&A
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Poser une question sur un contexte
result = qa_pipeline(
    question="Quel est le délai pour obtenir un acte de naissance ?",
    context="Pour obtenir un acte de naissance, le délai est de 5 à 10 jours ouvrés."
)

print(result)
# {'answer': '5 à 10 jours ouvrés', 'score': 0.95, 'start': 52, 'end': 70}
```

---

## 5. 🚀 FastAPI - Développement d'API REST

FastAPI est un framework Python moderne pour créer des APIs performantes.

### Documentation officielle

| Ressource | Description | Lien |
|-----------|-------------|------|
| **FastAPI Documentation** | Documentation complète officielle | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |
| **Tutorial - User Guide** | Tutoriel pas à pas | [fastapi.tiangolo.com/tutorial](https://fastapi.tiangolo.com/tutorial/) |
| **First Steps** | Premiers pas avec FastAPI | [fastapi.tiangolo.com/tutorial/first-steps](https://fastapi.tiangolo.com/tutorial/first-steps/) |

### Tutoriels complémentaires

| Ressource | Description | Lien |
|-----------|-------------|------|
| **FastAPI Tutorial (GeeksforGeeks)** | Tutoriel complet | [geeksforgeeks.org/fastapi-tutorial](https://www.geeksforgeeks.org/python/fastapi-tutorial/) |
| **Python REST APIs (Real Python)** | Tutoriel approfondi | [realpython.com/fastapi-python-web-apis](https://realpython.com/fastapi-python-web-apis/) |
| **FastAPI for ML (KDnuggets)** | Orienté Data Science | [kdnuggets.com/fastapi-tutorial](https://www.kdnuggets.com/fastapi-tutorial-build-apis-with-python-in-minutes) |

### Exemple de code minimal

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    confidence: float

@app.post("/ask", response_model=Answer)
async def ask_question(q: Question):
    # Logique de réponse ici
    return Answer(answer="Réponse générée", confidence=0.85)

# Lancer avec : uvicorn main:app --reload
```

### Points clés FastAPI

- **Documentation automatique** : Swagger UI sur `/docs`, ReDoc sur `/redoc`
- **Validation automatique** : Via Pydantic models
- **Async/await** : Support natif de l'asynchrone
- **Type hints** : Autocomplétion IDE et validation

---

## 6. 🧪 Tests et CI/CD

### pytest

| Ressource | Description | Lien |
|-----------|-------------|------|
| **pytest Documentation** | Documentation officielle | [docs.pytest.org](https://docs.pytest.org/) |
| **Testing FastAPI** | Tests d'API avec TestClient | [fastapi.tiangolo.com/tutorial/testing](https://fastapi.tiangolo.com/tutorial/testing/) |

### Exemple de test

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_ask_question():
    response = client.post(
        "/ask",
        json={"question": "Comment obtenir un acte de naissance ?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
```

### GitHub Actions

| Ressource | Description | Lien |
|-----------|-------------|------|
| **GitHub Actions Docs** | Documentation officielle | [docs.github.com/actions](https://docs.github.com/en/actions) |
| **Python CI** | Guide spécifique Python | [docs.github.com/actions/python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python) |

### Exemple de workflow CI

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
```

---

## 7. 📖 Concepts Complémentaires

### Similarité cosinus

La similarité cosinus mesure l'angle entre deux vecteurs. Plus l'angle est petit, plus les vecteurs sont similaires.

**Formule** : `cos(θ) = (A · B) / (||A|| × ||B||)`

- Résultat entre -1 et 1
- 1 = identiques
- 0 = orthogonaux (pas de relation)
- -1 = opposés

**Ressource** : [Wikipedia - Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

### OpenAPI / Swagger

Standard de documentation pour APIs REST. FastAPI génère automatiquement la documentation OpenAPI.

**Ressource** : [swagger.io/specification](https://swagger.io/specification/)

### Embeddings et NLP

**Ressource** : [HuggingFace NLP Course - Chapter 1](https://huggingface.co/learn/nlp-course/chapter1/1)

---

## 🔗 Récapitulatif des liens essentiels

### Incontournables (à consulter en priorité)

1. [AWS - What is RAG?](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
2. [SBERT - Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
3. [HuggingFace - Question Answering](https://huggingface.co/tasks/question-answering)
4. [FastAPI - Tutorial](https://fastapi.tiangolo.com/tutorial/)
5. [HuggingFace - Inference API](https://huggingface.co/docs/api-inference/index)

### Modèles à utiliser

| Usage | Modèle | Lien |
|-------|--------|------|
| Embeddings | all-MiniLM-L6-v2 | [Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| LLM | Mistral-7B-Instruct-v0.2 | [Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| Q&A Extractif | roberta-base-squad2 | [Model Card](https://huggingface.co/deepset/roberta-base-squad2) |

---

## 💡 Conseils de lecture

1. **Ne lisez pas tout d'un coup** : Consultez les ressources au fur et à mesure de votre avancement
2. **Testez le code** : Chaque exemple de code est fonctionnel, testez-le !
3. **Utilisez les démos HuggingFace** : Les pages de modèles ont des interfaces de test
4. **Documentez vos découvertes** : Notez ce qui vous a aidé pour votre rapport de veille

---

*Dernière mise à jour : Janvier 2026*