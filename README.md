# Projet : Assistant FAQ Intelligent pour Collectivité Territoriale

## Contexte professionnel

L'objectif de ce projet est de concevoir, développer et déployer une API d'assistance FAQ intégrant un modèle de langage large (LLM), en suivant une démarche de benchmark pour sélectionner la meilleure approche technique parmi trois stratégies proposées :
1. LLM seul
2. Recherche sémantique + LLM
3. Q&A extractif

# Documentations

- [Brief du projet](doc/BRIEF_PROJET.md)
- [Note de cadrage](doc/NOTE_CADRAGE.md)
- [Rapport de benchmark](doc/RAPPORT_BENCHMARK.md)

# Documentation Technique

## Mon Comparatif de Méthodes

Pour sélectionner l'approche technique la plus adaptée, j'ai réalisé un benchmark de trois stratégies distinctes :

1.  **LLM seul**: Le LLM est utilisé en "zero-shot", répondant aux questions en se basant uniquement sur ses connaissances intrinsèques.
2.  **Recherche + LLM (RAG)**: Le système récupère les questions similaires dans la FAQ via une recherche sémantique (vectorisation des questions), puis ces questions et leurs réponses sont injectées comme contexte dans le prompt d'un LLM qui génère la réponse finale.
3.  **Q&A Extractif**: Un modèle spécialisé analyse le corpus de documents pour en extraire directement le segment de texte contenant la réponse, sans génération de nouveau contenu.

J'ai évalué chaque méthode sur la base de la pertinence et de la précision des réponses. Les résultats détaillés de cette analyse comparative sont disponibles dans le [Rapport de benchmark](doc/RAPPORT_BENCHMARK.md).

## La Méthode Choisie : Recherche + LLM (RAG)

Suite à l'évaluation, j'ai retenu la méthode **Recherche + LLM (RAG)**. Elle offre le meilleur compromis entre la flexibilité de génération du LLM et la fidélité aux données de la collectivité. En ancrant les réponses sur des informations factuelles issues de la FAQ, cette approche minimise les risques d'hallucination et garantit une haute précision.

## Architecture de l'API

L'API REST, développée avec FastAPI, sert de point d'entrée pour l'application. Son architecture est modulaire et découplée en plusieurs services :

*   **Endpoint API (FastAPI)**: Gère les requêtes HTTP, la validation des données d'entrée (via Pydantic) et l'orchestration des appels aux services internes.
*   **Service RAG**: Le cœur de l'application. Il prend en charge :
    *   Le pré-traitement et l'indexation des données de la FAQ dans une base de données vectorielle.
    *   La vectorisation de la question utilisateur (embedding) pour effectuer une recherche de similarité sémantique.
    *   La récupération des N documents les plus pertinents depuis la base vectorielle.
    *   La construction d'un prompt enrichi avec le contexte récupéré et l'interrogation du LLM pour la génération de la réponse.
*   **Service de Chargement de Données**: Module responsable du chargement et de la sérialisation des données sources de la FAQ.
