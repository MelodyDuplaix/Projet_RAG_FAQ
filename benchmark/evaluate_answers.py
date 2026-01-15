import pandas as pd
from sentence_transformers import SentenceTransformer, util


def _normalize_text(text):
    if text is None:
        return ""
    return str(text).strip()


def _compute_keywords_proportion(answer, expected_keywords):
    if expected_keywords is None:
        return 0.0

    if isinstance(expected_keywords, str):
        try:
            import ast
            expected_keywords = ast.literal_eval(expected_keywords)
        except Exception:
            expected_keywords = [k.strip() for k in expected_keywords.split(",") if k.strip()]

    if not isinstance(expected_keywords, (list, tuple)) or len(expected_keywords) == 0:
        return 0.0

    answer_lower = answer.lower()
    count_present = 0

    for kw in expected_keywords:
        kw_str = str(kw).strip().lower()
        if kw_str and kw_str in answer_lower:
            count_present += 1

    return count_present / len(expected_keywords)


class GoldenSetEvaluator:
    def __init__(
        self,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        col_answer_model="answer_model",
        col_expected_keywords="expected_keywords",
        col_expected_summary="expected_answer_summary",
        col_question="question",
        col_expected_answer="expected_answer_summary",
        col_latency="latency_seconds",
    ):
        self.embedding_model_name = embedding_model_name
        self.col_answer_model = col_answer_model
        self.col_expected_keywords = col_expected_keywords
        self.col_expected_summary = col_expected_summary
        self.col_question = col_question
        self.col_expected_answer = col_expected_answer
        self.col_latency = col_latency

        self._st_model = SentenceTransformer(self.embedding_model_name)

    def _evaluate_keywords_and_similarity(self, df):
        answers = df[self.col_answer_model].fillna("").apply(_normalize_text).tolist()
        summaries = df[self.col_expected_summary].fillna("").apply(_normalize_text).tolist()

        emb_answers = self._st_model.encode(answers, convert_to_tensor=True)
        emb_summaries = self._st_model.encode(summaries, convert_to_tensor=True)

        cos_sim_matrix = util.cos_sim(emb_answers, emb_summaries)
        similarity_scores = cos_sim_matrix.diag().cpu().tolist()

        keywords_props = []
        for answer, kws in zip(answers, df[self.col_expected_keywords].tolist()):
            prop = _compute_keywords_proportion(answer, kws)
            keywords_props.append(prop)

        df = df.copy()
        df["keywords_proportion"] = keywords_props
        df["similarity_answer"] = similarity_scores

        return df

    def _collect_manual_scores(self, df):
        pertinences = []
        hallucinations = []

        print("\n=== ÉVALUATION MANUELLE ===")
        print(" - pertinence : 0, 1 ou 2")
        print(" - hallucination : x (oui) ou v (non)\n")

        for idx, row in df.iterrows():
            question = row[self.col_question]
            expected_answer = row[self.col_expected_answer]
            predicted_answer = row[self.col_answer_model]

            print("=" * 80)
            print(f"[{idx}] Question :\n{question}\n")
            print("Réponse attendue :")
            print(expected_answer)
            print("\nRéponse prédite :")
            print(predicted_answer)
            print("-" * 80)

            while True:
                raw = input("Note (format 0/1/2 x/v) : ").strip().lower()
                parts = raw.split()

                if len(parts) == 2:
                    p_str, h_str = parts
                    try:
                        p = int(p_str)
                    except ValueError:
                        p = None

                    if p in (0, 1, 2) and h_str in ("x", "v"):
                        pertinences.append(p)
                        hallucinations.append(True if h_str == "x" else False)
                        break

                print("Entrée invalide. Exemple attendu : '0 x' ou '2 v'.")

            print()

        df = df.copy()
        df["manual_pertinence"] = pertinences
        df["manual_hallucination"] = hallucinations

        return df

    def _compute_summary_scores(self, df):
        col_keywords_prop = "keywords_proportion"
        col_similarity = "similarity_answer"
        col_pertinence = "manual_pertinence"
        col_hallucination = "manual_hallucination"
        col_latency = self.col_latency

        keywords_mean = df[col_keywords_prop].mean() if col_keywords_prop in df.columns else 0.0
        similarity_mean = df[col_similarity].mean() if col_similarity in df.columns else 0.0

        exactitude_score = (keywords_mean + similarity_mean) / 2.0

        pertinence_mean = df[col_pertinence].mean() if col_pertinence in df.columns else 0.0

        if col_hallucination in df.columns:
            hallucinations_rate = df[col_hallucination].mean()
        else:
            hallucinations_rate = 0.0

        if col_latency in df.columns:
            latence_mean = df[col_latency].mean()
        else:
            latence_mean = 0.0

        latence_score = 1.0 / (1.0 + latence_mean) if latence_mean > 0 else 1.0

        exactitude_weight = 0.30
        pertinence_weight = 0.20
        hallucinations_weight = 0.20
        latence_weight = 0.15
        complexite_weight = 0.15

        hallucinations_score = 1.0 - hallucinations_rate
        complexite_score = 1.0

        global_score = (
            exactitude_score * exactitude_weight
            + (pertinence_mean / 2.0) * pertinence_weight
            + hallucinations_score * hallucinations_weight
            + latence_score * latence_weight
            + complexite_score * complexite_weight
        )

        summary = {
            "keywords_mean": keywords_mean,
            "similarity_mean": similarity_mean,
            "exactitude_score": exactitude_score,
            "pertinence_mean": pertinence_mean,
            "hallucinations_rate": hallucinations_rate,
            "latence_mean": latence_mean,
            "latence_score": latence_score,
            "complexite_score": complexite_score,
            "global_score": global_score,
        }

        return summary

    def evaluate(self, df):
        df_eval = self._evaluate_keywords_and_similarity(df)
        df_eval = self._collect_manual_scores(df_eval)

        if self.col_latency not in df_eval.columns:
            df_eval[self.col_latency] = None

        summary = self._compute_summary_scores(df_eval)

        return df_eval, summary

    @staticmethod
    def append_summary_to_csv(method_name, summary, path="data/methods_scores_summary.csv"):
        row = {"method": method_name}
        row.update(summary)

        try:
            existing = pd.read_csv(path)
            df_out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        except FileNotFoundError:
            df_out = pd.DataFrame([row])

        df_out.to_csv(path, index=False, encoding="utf-8")
