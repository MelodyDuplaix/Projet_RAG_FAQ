import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    return mo, torch


@app.cell
def _(torch):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    # Embedding models
    """)
    return


@app.cell
def _():
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, util


@app.cell
def _(model):
    model.max_seq_length
    return


@app.cell
def _():
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    return (sentences,)


@app.cell
def _(model, sentences):
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_tensor=True)
    print(embeddings.shape)
    return (embeddings,)


@app.cell
def _(model, query):
    query_embeddings = model.encode(query, show_progress_bar=True, convert_to_tensor=True)
    query_embeddings.shape
    return (query_embeddings,)


@app.cell
def _(embeddings, query_embeddings, util):
    # Cosinus Similarity between the query and first sentence 
    util.cos_sim(query_embeddings, embeddings[2])
    return


@app.cell
def _(embeddings, model):
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Find the most similar sentences
    """)
    return


@app.cell
def _(embeddings, query_embeddings, util):
    result = util.semantic_search(query_embeddings, embeddings)[0]
    result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reranker model
    """)
    return


@app.cell
def _():
    from sentence_transformers import CrossEncoder

    # 1. Load a pretrained CrossEncoder model
    model_cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return (model_cross,)


@app.cell
def _():
    # The texts for which to predict similarity scores
    query = "How many people live in Berlin?"
    passages = [
        "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
        "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
    ]

    return passages, query


@app.cell
def _(model_cross, passages, query):

    # 2a. predict scores for pairs of texts
    scores = model_cross.predict([(query, passage) for passage in passages])
    print(scores)
    # => [8.607139 5.506266 6.352977]
    return


@app.cell
def _(model_cross, passages, query):
    # 2b. Rank a list of passages for a query
    ranks = model_cross.rank(query, passages, return_documents=True)

    print("Query:", query)
    for rank in ranks:
        print(f"- #{rank['corpus_id']} ({rank['score']:.2f}): {rank['text']}")
    """
    Query: How many people live in Berlin?
    - #0 (8.61): Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
    - #2 (6.35): In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.
    - #1 (5.51): Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sparse Encoder model
    """)
    return


@app.cell
def _():
    from sentence_transformers import SparseEncoder
    return (SparseEncoder,)


@app.cell
def calculate_sparces_imilarity(SparseEncoder):
    def calculate_sparce_embedding_similarity(sentences):
        # 1. Load a pretrained SparseEncoder model
        model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

        # 2. Calculate sparse embeddings by calling model.encode()
        embeddings = model.encode(sentences)
        print(embeddings.shape)

        # 3. Calculate the embedding similarities
        similarities = model.similarity(embeddings, embeddings)
        print(similarities)

        # 4. Check sparsity stats
        stats = SparseEncoder.sparsity(embeddings)
        return stats

    return (calculate_sparce_embedding_similarity,)


@app.cell
def _(calculate_sparce_embedding_similarity, sentences):
    stats = calculate_sparce_embedding_similarity(sentences)
    print(f"Sparsity: {stats['sparsity_ratio']:.2%}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
