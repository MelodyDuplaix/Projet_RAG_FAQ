import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import json
    from collections import Counter
    return Counter, json


@app.cell
def _(json):
    with open("data/golden-set.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data,)


@app.cell
def _(data):
    categories = data["metadata"]["categories"]
    categories
    return


@app.cell(hide_code=True)
def _(data):
    print(f"Nombre total de question: {data['metadata']['total_questions']}")
    return


@app.cell(hide_code=True)
def _(Counter, data):
    dif_counts = Counter(item["difficulty"] for item in data["golden_set"])
    print("Questions par difficulté :")
    for dif, n in dif_counts.items():
        print(f"  {dif}: {n}")
    return


if __name__ == "__main__":
    app.run()
