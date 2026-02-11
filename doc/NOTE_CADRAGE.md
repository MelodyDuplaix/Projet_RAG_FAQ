# Note de Cadrage - Projet FAQ Intelligent

**Étudiant(s)** : Melody Duplaix

**Date** : 12-01-2026

**Version** : 1.0

---

## 1. Contexte et objectifs

### 1.1 Contexte du projet

Le client est une collectivitée territoriale (la Communauté de Communes Val de Loire Numérique) qui souhaite moderniser son service d'accueil citoyen en mettant en place un assistant FAQ intelligent. Cet assistant doit être capable de répondre automatiquement aux questions récurrentes des citoyens concernant les démarches administratives, afin de libérer du temps pour les agents. L'assistant sera accessible via une API REST intégrée au site web de la collectivité.

### 1.2 Objectifs du projet

**Objectif principal** :
Développer et déployer une API d'assistance FAQ intégrant un LLM, en suivant une démarche de benchmark pour sélectionner la meilleure approche technique parmi les 3 stratégies proposées.

**Objectifs secondaires** :
- [x] Préparer un protocole de benchmark
- [x] Développer les 3 stratégies de réponse aux questions
- [x] Réaliser un benchmark comparatif des 3 stratégies
- [x] Implémenter l'API avec la stratégie retenue
- [x] Mettre en place des tests automatisés et un pipeline CI/CD

### 1.3 Périmètre

**Dans le périmètre** :
- Développement et benchmark des 3 stratégies de réponse
- Mise en place de l'API REST
- Tests unitaires et de non-régression
- Documentation technique

**Hors périmètre** :
- LLM non open-source ou nécessitant une API payante
- Intégration front-end sur le site web de la collectivité
- 

---

## 2. Compréhension des 3 stratégies

### 2.1 Stratégie A - LLM seul

**Principe** :
Cette stratégie consiste à utiliser un modèle de langage large (LLM) pour générer des réponses aux questions posées par les utilisateurs. Le LLM est alimenté par un prompt système qui fournit le contexte nécessaire pour répondre de manière pertinente aux questions sur les démarches administratives. Le modèle traite directement la question et génère une réponse sans recourir à des étapes intermédiaires

**Avantages attendus** :
- Réponse rapide et directe
- Moins de complexité technique

**Inconvénients attendus** :
- Imprécisions et hallucinations dûes au manque d'information contextuelle

**Schéma simplifié** :
```
Question → LLM → Réponse
```

### 2.2 Stratégie B - Recherche sémantique + LLM

**Principe** :
Cette stratégie combine une étape de recherche sémantique avec l'utilisation d'un LLM. Lorsqu'une question est posée, un moteur de recherche sémantique utilise des embeddings pour identifier les FAQ les plus pertinentes dans une base de données. Ces FAQ sont ensuite utilisées pour contextualiser la question avant de la transmettre au LLM, qui génère une réponse basée sur ces informations.


**Avantages attendus** :
- Réponses plus précises grâce à la recherche sémantique
- Meilleure gestion des questions complexes

**Inconvénients attendus** :
- Complexité technique accrue
- Coût potentiellement plus élevé en ressources

**Schéma simplifié** :
```
Question → Recherche sémantique → LLM → Réponse
```

### 2.3 Stratégie C - Q&A extractif

**Principe** :
Cette stratégie utilise une combinaison de recherche sémantique et d'un modèle extractif de questions-réponses. Lorsqu'une question est posée, un moteur de recherche sémantique identifie les documents ou FAQ les plus pertinents. Ensuite, un modèle extractif analyse ces documents pour extraire précisément la réponse à la question posée, en se basant sur le contenu existant plutôt que de générer une nouvelle réponse.


**Avantages attendus** :
- Réponses extraites directement du contenu existant
- Réponses plus fiables et précises
- Moins de risque d'incohérence ou d'erreurs dans les réponses générées

**Inconvénients attendus** :
- Moins de flexibilité dans la génération de réponses
- Dépendance forte au contenu existant

**Schéma simplifié** :
```
Question → Recherche sémantique → Q&A extractif → Réponse
```

---

## 3. Stack technique envisagée

### 3.1 Composants principaux

| Composant | Technologie choisie | Justification |
|-----------|---------------------|---------------|
| Langage | Python 3.x | Language de programmation le plus adaptée pour le développement de modèles IA |
| Framework API | FastAPI | Facilité d'utilisation et performance pour créer des APIs REST |
| LLM |  |  |
| Embeddings | Sentence Transformers | Librairie efficace pour générer des embeddings sémantiques avec des modèles légers |
| Tests | pytest | Framework de tests unitaires populaire et facile à utiliser |
| CI/CD | GitHub Actions | Automatisation du pipeline CI/CD intégrée à GitHub

### 3.2 Modèles IA identifiés

| Usage | Modèle | Source | Raison du choix |
|-------|--------|--------|-----------------|
| LLM (génération) | Mistral-7B-Instruct-v0.2 | HuggingFace | Modèle LLM open-source performant et léger |
| Embeddings | all-MiniLM-L6-v2 | HuggingFace | Légèreté et performance, basé sur un ancien benchmark pour un précédent projet |
| Q&A extractif | camembert-base-squadFR-fquad-piaf | HuggingFace | Modèle d'extractif de questions-réponses performant spécialisé en francais |

---

## 4. Planning prévisionnel

| Jour | Phase | Objectifs | Livrables |
|------|-------|-----------|-----------|
| J1 | Veille technique | Compréhension du projet et des techniques et dentifier les modèles IA appropriés | Rapport de veille |
| J2 | Préparation du protocole de benchmark | Définir les critères d'évaluation et les scénarios de test | Protocole de benchmark |
| J3 | Développement des scripts d'évaluation | Création des scripts pour évaluer les performances des différentes stratégies | Scripts d'évaluation stratégie A |
| J4 | Développement des scripts d'évaluation | Création des scripts pour évaluer les performances des différentes stratégies | Scripts d'évaluation stratégie B et C |
| J5 | Benchmark | Évaluation comparative des performances des différentes stratégies | Rapport de benchmark |
| J6 | Développement API | Développement de l'API REST pour servir les réponses | API REST fonctionelle |
| J7 | Tests et CI/CD | Tests unitaires et de non-regression et configuration du pipeline CI/CD | Tests unitaires et CI/CD configurés |
| J8 | Monitoring | Configuration du monitoring de l'API | Monitoring configuré |
| J9 | Documentation | Rédaction de la documentation technique | Documentation technique rédigée |
| J10 | Finalisation | Finalisation du projet et livraison | Projet finalisé et livré |

---

## 5. Risques identifiés

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| API HuggingFace indisponible | Faible | Élevé | Utiliser des modèles locaux comme alternative |
| Modèles IA non performants | Moyenne | Élevé | Effectuer une veille approfondie et des tests préliminaires |


---

## 6. Questions en suspens



---

## 7. Ressources consultées (Veille J1) triés par pertinence

[Rapport de veille technique](RAPPORT_VEILLE_TECHNIQUE.md)

### 7.1 Sentence Transformers

- [huggingface/sentence-transformers: State-of-the-Art Text Embeddings](https://github.com/huggingface/sentence-transformers)
  - repo du framework Sentence Transformers pour la génération d'embeddings via Hugging Face en local
- [Semantic Search — Sentence Transformers documentation](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
  - documentation spécifique sur la rechercher sémantique Sentence Transformers
- MODEL: [sentence-transformers/all-MiniLM-L6-v2 · Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 
  - modèle de base pour les embeddings sémantiques
- [A Tutorial With Sentence-Transformers for Semantic Search](https://dzone.com/articles/sentence-transformers-semantic-search-tutorial)
  - tutoriel sur la recherche sémantique
- [A Step-by-Step Guide to Similarity and Semantic Search Using Sentence Transformers | by Hassanmustafa | Medium](https://medium.com/@hassanqureshi700/a-step-by-step-guide-to-similarity-and-semantic-search-using-sentence-transformers-7091723a7bf9)
  - Tuto avec exemple de code pour la construction d'une rechercher sémantique sur une faq
- [Cosine similarity - Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
  - Page d'explications sur la similarité cosinus utilisée en recherche sémantique

### 7.2 RAG (Retrieval Augmented Generation)

- [Retrieval-Augmented Generation (RAG) | Pinecone](https://www.pinecone.io/learn/retrieval-augmented-generation/)
  - explications techniques sur le RAG
- [Retrieval Augmented Generation (RAG) for LLMs | Prompt Engineering Guide](https://www.promptingguide.ai/research/rag)
  - explications techniques sur le RAG et les différentes architectures possibles

### 7.3 LLM

- [Inference Providers](https://huggingface.co/docs/inference-providers/index)
  - documentation de base sur l'inférence sur les modèles hébergés par Hugging Face
- [Feature Extraction](https://huggingface.co/docs/inference-providers/en/tasks/feature-extraction)
  - documentation sur l'embedding avec les modèles Hugging Face
- [Chat Completion](https://huggingface.co/docs/inference-providers/en/tasks/chat-completion)
  - documentation sur l'utilisation des modèles de type chat (LLM)
- [Serverless Inference API - Hugging Face Open-Source AI Cookbook](https://huggingface.co/learn/cookbook/en/enterprise_hub_serverless_inference_api)
  - documentation sur l'utilisation de l'API serverless de Hugging Face
- MODEL: [mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  - modèle LLM open-source performant et léger

### 7.4 Q&A extractif

- [API Reference](https://huggingface.co/docs/inference-providers/en/tasks/question-answering)
  - documentation sur l'utilisation des modèles de Q&A extractif
- [What is Question Answering? - Hugging Face](https://huggingface.co/tasks/question-answering)
  - documentation huggingface sur le Q&A extractif
- [Question answering](https://huggingface.co/docs/transformers/tasks/question_answering)
  - Tutoriel huggingface sur le Q&A extractif pas à pas
- [Question answering - Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter7/7)
  - Cours huggingface sur le Q&A extractif
- MODEL: [deepset/roberta-base-squad2 · Hugging Face](https://huggingface.co/deepset/roberta-base-squad2)
  - modèle fréquemment utilisé pour le Q&A extractif
- MODEL: [AgentPublic/camembert-base-squadFR-fquad-piaf · Hugging Face](https://huggingface.co/AgentPublic/camembert-base-squadFR-fquad-piaf)
  - modèle de Q&A extractif en français basé sur CamemBERT




---
