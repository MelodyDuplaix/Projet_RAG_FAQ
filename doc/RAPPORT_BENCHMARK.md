# Rapport de Benchmark - Stratégies FAQ Intelligent

**Étudiant(s)** : Melody Duplaix

**Date** : 2026-01-13

**Version** : 1.0

---

## Résumé exécutif

[2-3 phrases résumant les résultats et la recommandation finale]

**Recommandation** : Stratégie [A/B/C] - [Nom de la stratégie]

---

## 1. Protocole d'évaluation

### Cadre du benchmark

- Objectif : comparer 3 stratégies (LLM seul, Recherche sémantique + LLM, Q&A extractif) pour une FAQ citoyenne.
- Jeux de données :  
  - `faq_base.json` (≈70 QA) pour construire le corpus / index.
  - `golden_set.json` (≈30 questions : faciles, moyennes, hors scope) pour l’évaluation.
- Critères et poids : Exactitude 30%, Pertinence 20%, Hallucinations 20%, Latence 15%, Complexité 15%.

### Conditions de test

- Environnement : Python 3.10+, même machine, mêmes versions de libs, même conditions réseau.
- Modèles :  
  - LLM : Mistral‑7B‑Instruct‑v0.2 (stratégies A et B).
  - Embeddings : all‑MiniLM‑L6‑v2 (stratégies B et C).
  - Q&A extractif : camembert‑base‑squadFR‑fquad‑piaf (stratégie C).
- Paramètres : température, top‑k, max_tokens fixés et documentés pour toute la campagne.

### Méthode de mesure

- Pour chaque question du golden set : exécuter A, B, C et logguer question, réponse, temps, docs récupérés (B/C).
- Mesures automatiques :  
  - Exactitude : comparaison à la réponse de référence et aux keywords.
  - Latence : temps moyen par stratégie (start/end timestamp).
- Mesures manuelles :  
  - Pertinence : note 0–2 par réponse.
  - Hallucination : booléen « info inventée ».
  - Complexité : note 1–3 (implémentation, infra, maintenance).

### Agrégation et scoring

- Calculer, par stratégie : moyennes des métriques + taux d’hallucination.
- Appliquer la pondération pour obtenir un score global par stratégie (tableau comparatif).
- Conserver les résultats bruts dans un fichier CSV/JSON + quelques exemples de réponses typiques (bonnes, limites, hors scope).

### Interprétation et recommandation

- Analyser forces/faiblesses de chaque stratégie (précision, robustesse hors scope, risque d’hallucination, latence, complexité).
- Choisir une stratégie recommandée (A/B/C) + 3–5 arguments alignés avec les contraintes client (open source, hébergement interne, simplicité).
- Documenter les limites du benchmark (taille du golden set, types de questions peu représentés) et proposer 2–3 pistes d’amélioration (plus de données, tuning, optimisation index).

### 1.1 Critères d'évaluation

| Critère | Description | Méthode de mesure | Poids |
|---------|-------------|-------------------|-------|
| Exactitude | % de réponses correctes | Évaluation sur golden set | 30% |
| Pertinence | Qualité de la réponse (0-2) | Notation manuelle | 20% |
| Hallucinations | % de réponses avec infos inventées | Vérification manuelle | 20% |
| Latence | Temps de réponse moyen | Mesure automatique | 15% |
| Complexité | Facilité de maintenance | Évaluation qualitative | 15% |

### 1.2 Jeu d'entrainement (Faq Base)

- **Nombre de questions** : 70 questions

### 1.3 Jeu de test (Golden Set)

- **Nombre de questions** : 30 questions
- **Répartition** :
  - Questions faciles : 17
  - Questions moyennes : 8
  - Questions hors scope : 5

### 1.4 Conditions de test

- **Date des tests** : [Date]
- **Environnement** : [Local / Cloud]
- **Modèle LLM utilisé** : [Nom du modèle]
- **Modèle d'embeddings** : [Nom du modèle]
- **Nombre d'exécutions par question** : [X]

---

## 2. Résultats par stratégie

### 2.1 Stratégie A - LLM seul

**Configuration** :
- Modèle : mistral-7B-Instruct-v0.2
- Paramètres : temperature=0.7, max_tokens=1024

**Résultats** :

| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| Exactitude | 27.4% | |
| Pertinence moyenne | 0.667/2 | |
| Taux d'hallucinations | 46.7% | |
| Latence moyenne | 7.193s | |
| Complexité | Faible | |

**Observations qualitatives** :
-  Le modèle répond en se basant uniquement sur internet, ce qui entraîne des réponses hors sujet.
-  Le modèle vire parfois sur de l'anglais.
-  Il ne se base pas toujours en france.

---

### 2.2 Stratégie B - Recherche sémantique + LLM

**Configuration** :
- Modèle LLM : mistral-7B-Instruct-v0.2
- Modèle embeddings : all-MiniLM-L6-v2
- Top-K documents : [X]

**Résultats** :

| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| Exactitude | [X]% | |
| Pertinence moyenne | [X]/2 | |
| Taux d'hallucinations | [X]% | |
| Latence moyenne | [X]s | |
| Complexité | [Faible/Moyenne/Élevée] | |

**Observations qualitatives** :
- [Observation 1]
- [Observation 2]

**Exemples de réponses** :

| Question | Documents récupérés | Réponse | Évaluation |
|----------|---------------------|---------|------------|
| [Question 1] | [X docs] | [Réponse...] | ✅/⚠️/❌ |
| [Question 2] | [X docs] | [Réponse...] | ✅/⚠️/❌ |

---

### 2.3 Stratégie C - Q&A extractif

**Configuration** :
- Modèle Q&A : [X]
- Modèle embeddings : [X]
- Top-K documents : [X]

**Résultats** :

| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| Exactitude | [X]% | |
| Pertinence moyenne | [X]/2 | |
| Taux d'hallucinations | [X]% | |
| Latence moyenne | [X]s | |
| Complexité | [Faible/Moyenne/Élevée] | |

**Observations qualitatives** :
- [Observation 1]
- [Observation 2]

---

## 3. Analyse comparative

### 3.1 Tableau récapitulatif

| Critère | Poids | Stratégie A | Stratégie B | Stratégie C |
|---------|-------|-------------|-------------|-------------|
| Exactitude | 30% | [X]% | [X]% | [X]% |
| Pertinence | 20% | [X]/2 | [X]/2 | [X]/2 |
| Hallucinations | 20% | [X]% | [X]% | [X]% |
| Latence | 15% | [X]s | [X]s | [X]s |
| Complexité | 15% | [1-3] | [1-3] | [1-3] |
| **Score pondéré** | 100% | **[X]** | **[X]** | **[X]** |

### 3.2 Graphique comparatif

[Insérer un graphique radar ou histogramme comparant les 3 stratégies]

### 3.3 Analyse des forces et faiblesses

**Stratégie A** :
- ✅ Forces : [...]
- ❌ Faiblesses : [...]

**Stratégie B** :
- ✅ Forces : [...]
- ❌ Faiblesses : [...]

**Stratégie C** :
- ✅ Forces : [...]
- ❌ Faiblesses : [...]

---

## 4. Recommandation

### 4.1 Stratégie recommandée

**Choix : Stratégie [X] - [Nom]**

### 4.2 Justification

[Argumenter le choix en 3-5 points]

1. [Argument 1]
2. [Argument 2]
3. [Argument 3]

### 4.3 Limites de la recommandation

[Identifier les cas où cette stratégie pourrait ne pas être optimale]

### 4.4 Axes d'amélioration possibles

[Suggérer des pistes d'optimisation pour la stratégie retenue]

---

## 5. Annexes

### 5.1 Détail des résultats bruts

[Lien vers le fichier CSV/JSON des résultats complets]

### 5.2 Code du benchmark

[Lien vers le script de benchmark]

### 5.3 Grille d'évaluation complète

[Lien vers la grille d'évaluation remplie]

---
