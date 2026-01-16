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
        complexite_score_method=1.0,
    ):
        self.embedding_model_name = embedding_model_name
        self.col_answer_model = col_answer_model
        self.col_expected_keywords = col_expected_keywords
        self.col_expected_summary = col_expected_summary
        self.col_question = col_question
        self.col_expected_answer = col_expected_answer
        self.col_latency = col_latency
        self.complexite_score_method = float(complexite_score_method)

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

    def _compute_per_question_scores(self, df):
        col_keywords_prop = "keywords_proportion"
        col_similarity = "similarity_answer"
        col_pertinence = "manual_pertinence"
        col_hallucination = "manual_hallucination"
        col_latency = self.col_latency

        exactitude_weight = 0.30
        pertinence_weight = 0.20
        hallucinations_weight = 0.20
        latence_weight = 0.15
        complexite_weight = 0.15

        df = df.copy()

        df["exactitude_score"] = (
            df[col_keywords_prop].fillna(0.0) + df[col_similarity].fillna(0.0)
        ) / 2.0

        df["hallucination_flag"] = df[col_hallucination].fillna(False).astype(bool)
        df["pertinence_norm"] = df[col_pertinence].fillna(0.0) / 2.0

        lat = df[col_latency].fillna(0.0)
        df["latence_score"] = lat.apply(lambda x: 1.0 / (1.0 + x) if x > 0 else 1.0)

        df["hallucinations_score"] = 1.0 - df["hallucination_flag"].astype(float)

        df["complexite_score"] = self.complexite_score_method

        df["global_score"] = (
            df["exactitude_score"] * exactitude_weight
            + df["pertinence_norm"] * pertinence_weight
            + df["hallucinations_score"] * hallucinations_weight
            + df["latence_score"] * latence_weight
            + df["complexite_score"] * complexite_weight
        )

        return df

    def _compute_summary_scores(self, df):
        keywords_mean = df["keywords_proportion"].mean() if "keywords_proportion" in df.columns else 0.0
        similarity_mean = df["similarity_answer"].mean() if "similarity_answer" in df.columns else 0.0
        exactitude_score = df["exactitude_score"].mean() if "exactitude_score" in df.columns else 0.0
        pertinence_mean = df["manual_pertinence"].mean() if "manual_pertinence" in df.columns else 0.0
        hallucinations_rate = df["hallucination_flag"].mean() if "hallucination_flag" in df.columns else 0.0
        latence_mean = df[self.col_latency].mean() if self.col_latency in df.columns else 0.0
        latence_score = df["latence_score"].mean() if "latence_score" in df.columns else 0.0
        complexite_score = df["complexite_score"].mean() if "complexite_score" in df.columns else 0.0
        global_score = df["global_score"].mean() if "global_score" in df.columns else 0.0

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

        df_eval = self._compute_per_question_scores(df_eval)
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
