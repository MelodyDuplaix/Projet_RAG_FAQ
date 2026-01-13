# Veille technique

## Sentence Transformers & Similarité Cosinus

### [huggingface/sentence-transformers: State-of-the-Art Text Embeddings](https://github.com/huggingface/sentence-transformers)
**Auteur :** Nils Reimers et contributeurs , **Source :** github.com , Projet officiel largement utilisé dans l’écosystème NLP

Le framework Sentence-Transformers fournit une API Python pour calculer des embeddings de phrases et de documents, avec des modèles préentraînés et des outils pour l’entraînement, la recherche, le reranking et le clustering.  
L’écosystème s’appuie sur les modèles hébergés sur Hugging Face et couvre à la fois les embeddings denses, les embeddings clairsemés et les cross-encoders (rerankers).

- Chargement rapide d’un modèle via SentenceTransformer("all-MiniLM-L6-v2") puis encodage de listes de textes avec model.encode.
- Support de nombreuses architectures de Transformers (BERT, RoBERTa, DistilBERT, XLM-R, BART, etc.).
- Tâches principales : semantic search, semantic textual similarity, paraphrase mining, clustering, retrieve & re-rank.
- Possibilité de fine-tuning avec de nombreuses fonctions de perte (contrastive, triplet, margin, etc.) pour adapter les embeddings à un cas métier.

### [Semantic Search — Sentence Transformers documentation](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
**Auteur :** Équipe Sentence-Transformers , **Source :** sbert.net , Documentation officielle du framework Sentence-Transformers

La recherche sémantique consiste à représenter les documents et les requêtes sous forme de vecteurs et à retrouver les éléments les plus proches en termes de sens plutôt qu’en matching de mots-clés.  
Le pipeline standard encode le corpus une fois, puis encode chaque requête et applique une mesure de similarité cosinus pour récupérer les meilleurs candidats.

- Étapes : encodage du corpus, encodage de la requête, calcul de similarités, tri des résultats.
- Usage de la similarité cosinus ou du produit scalaire pour mesurer la proximité entre vecteurs.
- Utilisation possible d’index de nearest neighbors approximatifs (ANN) pour scalabilité sur de grands volumes.
- Intégration avec des bases vectorielles ou des moteurs de recherche (Faiss, Elastic, etc.).
- Impact fort du choix de modèle (taille, domaine, multilingue) sur la qualité des résultats de recherche.

### [sentence-transformers/all-MiniLM-L6-v2 · Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 
**Auteur :** Sentence-Transformers Team , **Source :** huggingface.co , Fiche officielle décrivant le modèle et son entraînement

Le modèle all-MiniLM-L6-v2 est un encodeur de phrases compact qui produit des embeddings de dimension 384, optimisés pour des tâches de similarité sémantique généralistes.  
Il offre un compromis vitesse/qualité très adapté aux systèmes temps réel et aux gros volumes de requêtes.

- Architecture MiniLM à 6 couches, entraînée sur des jeux de données NLI et paraphrases.
- Bon niveau sur les benchmarks de similarité (STS) tout en restant léger.
- Embeddings adaptés à la recherche sémantique, au clustering, à la détection de doublons et au reranking.
- Utilisation possible via la librairie `sentence-transformers` ou directement via `transformers` en feature extraction.
- Fonctionne bien sur CPU ou sur GPU peu puissant, ce qui facilite le déploiement en production.

### [A Tutorial With Sentence-Transformers for Semantic Search](https://dzone.com/articles/sentence-transformers-semantic-search-tutorial)
**Auteur :** Rédaction DZone , **Source :** dzone.com , Article pédagogique publié sur une plateforme d’ingénierie logicielle

- Installation de `sentence-transformers`, chargement d’un modèle (par exemple all-MiniLM-L6-v2), encodage du corpus puis encodage des requêtes.
- Utilisation de la similarité cosinus pour classer les documents selon leur proximité avec la requête.
- Affichage des top-k résultats avec leur score pour inspection qualitative.
- Discussion des limites de la recherche purement lexicale (TF-IDF/BM25) et des gains de la sémantique.
- Suggestions pour gérer des corpus plus volumineux via des structures ANN ou des bases vectorielles.

### [A Step-by-Step Guide to Similarity and Semantic Search Using Sentence Transformers | by Hassanmustafa | Medium](https://medium.com/@hassanqureshi700/a-step-by-step-guide-to-similarity-and-semantic-search-using-sentence-transformers-7091723a7bf9)
**Auteur :** Hassan Mustafa , **Source :** medium.com , Article de blog technique décrivant un cas d’usage concret

- Préparation des données de FAQ (questions, réponses, éventuellement catégories).
- Encodage des questions en vecteurs et stockage dans une structure de recherche (liste, DataFrame, vectordb).
- Encodage d’une nouvelle question, calcul de la similarité cosinus avec toutes les questions existantes, tri par score.
- Mise en place de seuils pour décider si la similarité est suffisante ou si la question doit être remontée à un humain.
- Possibilité d’améliorer encore les performances par fine-tuning sur des paires question–réponse annotées.

### [Cosine similarity - Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
**Auteur :** Contributeurs Wikipédia , **Source :** wikipedia.org , Article encyclopédique décrivant la définition mathématique et les usages

La similarité cosinus mesure la proximité angulaire entre deux vecteurs, avec une valeur entre -1 et 1, 1 correspondant à des vecteurs parfaitement alignés.  
Cette mesure est particulièrement adaptée aux embeddings normalisés, car elle capture la proximité de direction indépendamment de la norme.

- Définition : produit scalaire des vecteurs divisé par le produit de leurs normes.
- Utilisation fréquente sur des embeddings textuels normalisés pour comparer des phrases ou documents.
- Lien entre similarité cosinus et distance cosinus : la distance cosinus est souvent définie comme 1 moins la similarité cosinus, ce qui permet de l’utiliser comme métrique dans des algorithmes de clustering ou de nearest neighbors.
- Représentations géométriques en 2D ou 3D : deux vecteurs peuvent être éloignés en norme mais proches en angle, ce qui motive la normalisation.
- Usage large en recherche d’information, analyse de texte et systèmes de recommandation.


### Synthèse globale Sentence Transformers

Sentence-Transformers fournit un cadre pour représenter des textes sous forme de vecteurs, avec une API Python simple, une grande variété de modèles.
La brique mathématique centrale est la similarité cosinus, utilisée pour comparer ces vecteurs et retrouver des éléments proches en sens.

- Le framework propose des modèles préentraînés simples à charger et à utiliser, ainsi que des outils de fine-tuning pour adapter les embeddings à un domaine spécifique.
- Le modèle all-MiniLM-L6-v2 représente un bon compromis entre coût de calcul et qualité, bien adapté à des services temps réel.
- Les tutoriels autour de la recherche sémantique et des FAQ montrent comment construire rapidement des systèmes de question–réponse ou de recherche documentaire exploitant ces embeddings.
- Il y a un lien direct entre la qualité du modèle choisi, la métrique de similarité cosinus et la pertinence des résultats obtenus.

## Retrieval Augmented Generation (RAG)

### [Retrieval-Augmented Generation (RAG) | Pinecone](https://www.pinecone.io/learn/retrieval-augmented-generation/)
**Auteur :** Équipe Pinecone , **Source :** pinecone.io , Article technique publié par l’éditeur d’une base vectorielle

RAG combine un modèle génératif avec un système de recherche qui apporte du contexte à la demande de l’utilisateur.  
Le flux typique suit trois étapes : ingestion des données, récupération de passages pertinents, puis génération conditionnée par ces passages.

- Ingestion : nettoyage des données (textes, PDFs, wikis internes…), éventuel découpage en chunks, calcul des embeddings et indexation dans une base vectorielle.
- Récupération : encodage de la requête en vecteur, recherche des voisins les plus proches (dense, sparse ou hybride), reranking éventuel.
- Augmentation : construction d’un prompt qui combine question utilisateur et passages retrouvés.
- Génération : appel au LLM avec ce prompt enrichi pour produire une réponse ancrée dans les documents.
- Objectif principal : réduire les hallucinations et permettre l’accès à des données privées ou fréquemment mises à jour.

### [Retrieval Augmented Generation (RAG) for LLMs | Prompt Engineering Guide](https://www.promptingguide.ai/research/rag)
**Auteur :** Équipe Prompt Engineering Guide , **Source :** promptingguide.ai , Guide communautaire spécialisé sur le prompt engineering

Les architectures RAG se déclinent en variantes plus ou moins sophistiquées selon la complexité de l’application.  
Le cœur reste la boucle requête → retrieval → génération, mais on peut enrichir la pipeline avec des modules supplémentaires.

- RAG naïf : indexation simple, retrieval par similarité, concaténation des passages dans le prompt et génération.
- RAG modulaire : séparation explicite des modules (search, mémoire, fusion, routage, tâches spécifiques), permettant d’ajuster ou remplacer chaque brique.
- RAG avancé : ajout de mémoire de conversation, fusion de plusieurs sources, routage dynamique vers différents modèles ou index.
- Importance du prompt : bien séparer le rôle du « system prompt » (règles, contraintes) et du « user prompt » (question, consignes locales).
- Possibilité d’utiliser du n-shot prompting pour montrer au modèle des exemples de bonnes réponses pilotées par le contexte récupéré.

### Synthèse Global RAG

Le RAG fournit un schéma général pour brancher un LLM sur une base de connaissances externe et ainsi produire des réponses factuelles, à jour et spécifiques à un domaine.  
La qualité d’un système RAG dépend autant du retrieval (chunking, embeddings, index, reranking) que du prompt et de l’architecture globale.

- La pipeline minimale comprend ingestion, retrieval et génération, mais des systèmes robustes ajoutent mémoire, routage et modules de contrôle.
- Le choix de la base vectorielle, du modèle d’embeddings et de la stratégie de chunking influence directement le signal que reçoit le LLM.
- Le prompt engineering est une surface de contrôle critique pour réduire les hallucinations et garantir que le modèle s’appuie réellement sur les passages retournés.
- Pour un usage en entreprise, RAG est le pattern privilégié pour exposer des données internes via des assistants type chatbot ou moteur de recherche conversationnel.

## Inference & modèles Hugging Face

### [Inference Providers](https://huggingface.co/docs/inference-providers/index)
**Auteur :** Équipe Hugging Face , **Source :** huggingface.co , Documentation officielle des fournisseurs d’inférence

Les Inference Providers Hugging Face proposent une manière standardisée d’appeler des modèles hébergés (open-source ou propriétaires) sans gérer l’infrastructure. 

- Abstraction unique pour différents backends (Hugging Face Inference, cloud providers, etc.).
- Configuration du provider (clés d’API, région, modèle cible) au niveau du client.
- Gestion de la scalabilité et de la latence côté provider (serverless, autoscaling).
- Intégration facilitée dans des applications Python.

### [Feature Extraction](https://huggingface.co/docs/inference-providers/en/tasks/feature-extraction)
**Auteur :** Équipe Hugging Face , **Source :** huggingface.co , Page de documentation sur la tâche de feature extraction

La tâche de feature extraction permet d’obtenir des représentations vectorielles à partir de textes ou de tokens.  
Cette fonctionnalité est utilisée pour les embeddings nécessaires à la recherche sémantique, au clustering ou aux systèmes RAG.

- Envoi d’un ou plusieurs textes à l’API qui renvoie les embeddings correspondants.
- Utilisation en amont de bases vectorielles pour indexer des documents.
- Réutilisation d’un même modèle pour plusieurs tâches de similarité et de classification.
- Intégration naturelle avec les modèles de type Sentence-Transformers ou d’autres encodeurs Hugging Face.

### [Chat Completion](https://huggingface.co/docs/inference-providers/en/tasks/chat-completion)
**Auteur :** Équipe Hugging Face , **Source :** huggingface.co , Page de documentation décrivant l’API de chat

La tâche chat completion expose les modèles de type assistant conversationnel via un format de messages (system, user, assistant).  
Elle permet de construire des chats, des agents ou des assistants spécialisés en contrôlant le contexte et les consignes.

- Envoi d’une liste de messages structurés (rôle + contenu).
- Paramétrage de la génération : température, top-p, max tokens, etc.
- Injection de contexte (résultats de retrieval, règles métier) dans le system prompt ou dans les messages précédents.

### 4. [Serverless Inference API - Hugging Face Open-Source AI Cookbook](https://huggingface.co/learn/cookbook/en/enterprise_hub_serverless_inference_api)
**Auteur :** Équipe Hugging Face , **Source :** huggingface.co , Article de cookbook expliquant l’API serverless

L’API serverless de Hugging Face fournit un moyen de déployer et consommer des modèles sans gérer de serveurs ni de scaling manuel.  
Elle est pensée pour brancher rapidement des modèles sur des applications de production ou des prototypes avancés.

- Déploiement d’un modèle hébergé (ou custom) en endpoint serverless.
- Gestion automatique du scaling en fonction de la charge.
- Possibilité de combiner plusieurs endpoints (embeddings, chat, Q&A) dans une même application.
- Intéressant pour séparer responsabilités : data/ML d’un côté, intégration produit de l’autre.

### [mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
**Auteur :** Mistral AI , **Source :** huggingface.co , Fiche officielle décrivant le modèle d’instruction Mistral-7B

Mistral-7B-Instruct-v0.2 est un LLM optimisé pour des usages instruct (chat, assistants, Q&A).  

- Modèle open-source, fine-tuné pour suivre des instructions, générer du texte cohérent et dialoguer.
- Bon comportement sur des tâches de raisonnement léger, de résumé, de traduction et de Q&A généraliste.
- Taille raisonnable permettant des temps de réponse acceptables en contexte interactif.
- Utilisable via l’API Hugging Face (chat completion) ou en self-hosted (transformers + accélération).
- Candidat naturel pour la brique « génération » dans un pipeline RAG ou un assistant spécialisé.

### Synthèse global LLM Inference

Les Inference Providers Hugging Face et l’API serverless offrent une couche d’abstraction pour appeler des LLM et des modèles d’embeddings sans se soucier de l’infrastructure. 

- La séparation claire entre tâches (chat, embeddings, Q&A) permet de combiner plusieurs modèles dans une même application.
- Les formats standard (chat messages, embeddings) simplifient l’intégration en Python, JS ou autres langages.
- L’API serverless crée un pont simple entre les besoins produit (endpoints HTTP) et les modèles ML (hébergés, versionnés).
- Pour un projet de dev IA, maîtriser ces interfaces permet de passer rapidement du prototype local au service exposé en production.

## Q&A extractif

### [API Reference](https://huggingface.co/docs/inference-providers/en/tasks/question-answering)
**Auteur :** Équipe Hugging Face , **Source :** huggingface.co , Référence API officielle pour la tâche de question answering

L’API de question answering permet d’envoyer un couple (question, contexte texte) et de récupérer une réponse extraite sous forme de span dans le contexte.  
Le modèle renvoie généralement la réponse, sa position dans le texte et parfois un score de confiance.

- Input : question en langage naturel et contexte (paragraphe, document).
- Output : texte réponse, indices de début/fin dans le contexte, score.
- Utilisation pour rechercher une information précise dans un document plutôt que générer un texte libre.
- Compatible avec différents modèles préentraînés (anglais, multilingue, français, etc.).

### [What is Question Answering? - Hugging Face](https://huggingface.co/tasks/question-answering)
**Auteur :** Équipe Hugging Face , **Source :** huggingface.co , Page de description de la tâche de question answering

Le Q&A extractif repose sur un modèle qui prédit la position du début et de la fin de la réponse dans un passage donné.  
Cette approche convient quand la réponse est explicitement présente dans le texte, ce qui permet de garder un comportement déterministe et contrôlable.

- Le modèle encode la question et le contexte, puis produit des scores pour chaque token comme début/fin possible.
- La réponse est formée par le span optimisant la combinaison des scores début/fin.
- Le Q&A extractif permet de limiter fortement les hallucinations par rapport aux modèles génératifs.
- Utilisation typique : recherche d’un fait, d’une définition, d’une date, d’un argument précis dans un document.
- Combinaison fréquente avec un moteur de retrieval (d’abord retrouver un paragraphe pertinent, puis appliquer le modèle extractif).

### [Question answering](https://huggingface.co/docs/transformers/tasks/question_answering)
**Auteur :** Équipe Hugging Face , **Source :** huggingface.co , Tutoriel officiel sur l’usage de Transformers pour le Q&A

Le tutoriel montre comment utiliser Transformers pour charger un modèle de Q&A, tokeniser (question + contexte) et obtenir les indices de la réponse. 

- Usage du pipeline question-answering pour une interface simplifiée.
- Explication de la tokenisation conjointe (question + contexte) et du format attendu par le modèle.
- Détails sur le post-traitement : décodage du span, gestion des scores, filtrage des réponses peu probables.
- Exemple de code complet depuis l’import des librairies jusqu’à l’affichage de la réponse.
- Possibilité de remplacer le modèle par un autre plus adapté (multilingue, spécialisé domaine).

### [Question answering - Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter7/7)
**Auteur :** Équipe Hugging Face , **Source :** huggingface.co , Chapitre de cours dédié au question answering dans le LLM Course

Comment adapter un modèle à un nouveau dataset de Q&A et comment mesurer la qualité avec des métriques dédiées.

- Présentation des datasets typiques (SQuAD, etc.) et de leur format.
- Exemple de fine-tuning d’un modèle de base sur un dataset Q&A.
- Mise en place d’une boucle d’évaluation (EM, F1) pour suivre la progression.
- Discussion des limites du Q&A extractif (réponses dispersées, paraphrasées, implicites).
- Ouverture vers des approches plus avancées (Q&A génératif, RAG) pour des cas moins structurés.

### [deepset/roberta-base-squad2 · Hugging Face](https://huggingface.co/deepset/roberta-base-squad2)
**Auteur :** deepset.ai , **Source :** huggingface.co , Fiche décrivant un modèle RoBERTa fine-tuné sur SQuAD2

Le modèle deepset/roberta-base-squad2 est un RoBERTa base fine-tuné pour le Q&A extractif en anglais.  

- Gère la présence de questions sans réponse possible en prédisant un span « vide » ou un seuil de non-réponse.
- Entraînement sur des paires question–contexte avec réponses parfois absentes.
- Utilisation fréquente comme baseline pour des systèmes de Q&A extractif en anglais.
- Compatible avec Transformers et les pipelines question-answering.
- Base robuste pour un système de QA documentaire ou un prototype de moteur de recherche de faits.

### [AgentPublic/camembert-base-squadFR-fquad-piaf · Hugging Face](https://huggingface.co/AgentPublic/camembert-base-squadFR-fquad-piaf)
**Auteur :** AgentPublic , **Source :** huggingface.co , Fiche décrivant un modèle CamemBERT adapté au Q&A en français

Ce modèle est un CamemBERT base fine-tuné sur plusieurs datasets français de Q&A. 

- Spécialisation sur des corpus et questions en français, incluant des textes journalistiques ou encyclopédiques.
- Architecture CamemBERT adaptée au français (pré-entraînement sur corpus francophones).
- Utilisation possible dans un pipeline question-answering simplement en changeant le nom de modèle.
- Pertinent pour des bases de connaissances, FAQ ou documents internes rédigés en français.

### Synthèse Global Q&A extractif

Le Q&A extractif permet d’identifier un span de texte précis répondant à une question dans un document donné, ce qui offre un comportement beaucoup plus contrôlable que la génération libre.  

- Le schéma typique combine un moteur de retrieval (retrouver le bon paragraphe) et un modèle extractif (trouver la réponse dans ce paragraphe).
- Les pipelines Transformers et les APIs Hugging Face simplifient fortement l’intégration dans une application réelle.
- L’adaptation par fine-tuning sur un domaine (docs internes, support client, juridique) permet de spécialiser le comportement.
