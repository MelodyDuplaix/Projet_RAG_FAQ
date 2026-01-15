import json
import pandas as pd


def extract_questions(path="data/golden-set.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["golden_set"]
    questions_df = pd.DataFrame(questions)
    return questions_df

def load_faq_base(path="data/faq-base.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    faq = data["faq"]
    faq_df = pd.DataFrame(faq)
    return faq_df