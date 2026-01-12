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
- [ ] Préparer un protocole de benchmark
- [ ] Développer les 3 stratégies de réponse aux questions
- [ ] Réaliser un benchmark comparatif des 3 stratégies
- [ ] Implémenter l'API avec la stratégie retenue
- [ ] Mettre en place des tests automatisés et un pipeline CI/CD

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
- Possible manque de précision si le LLM n'est pas suffisamment spécialisé
- Plus de cout en ressources pour des questions complexes

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
| LLM (génération) | | HuggingFace | |
| Embeddings | | | |
| Q&A extractif | | | |

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

## 7. Ressources consultées (Veille J1)

| Source | URL | Pertinence | Notes |
|--------|-----|------------|-------|
| | | | |
| | | | |

---
