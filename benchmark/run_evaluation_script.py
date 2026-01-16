from data_access import extract_questions, load_faq_base
from methods_runners import LLMOnlyRunner, RAGRunner, ExtractiveQARunner
from evaluate_answers import GoldenSetEvaluator


def benchmark_llm_only():
    system_prompt = (
        """
        Tu es un assistant municipal francais expert de la communauté de communes Val de Loire Numérique.
        Ton but est de répondre EXCLUSIVEMENT aux questions sur les sujets : état civil, urbanisme, déchets, transports, petite-enfance, social, vie associative, élections, logement, culture/sport, fiscalité, eau/assainissement.
        Régles OBLIGATOIRES :
        - Si la question est hors sujet, ou si tu n'as pas suffisement d'informations pour répondre, répond UNIQUEMENT cette phrase: "Bonjour, je suis désolé mais je ne suis pas en mesure de répondre à cette question."
        - Sinon, commence toujours par "Bonjour"
        """
    )

    df_questions = extract_questions()

    runner = LLMOnlyRunner(
        question_col="question",
        answer_col="answer_model",
        latency_col="latency_seconds",
    )

    df_with_answers = runner.run_on_dataframe(
        df_questions,
        system_prompt=system_prompt,
        delay_seconds=1,
    )

    df_with_answers.to_csv(
        "data/llm-only-with-answers.csv",
        index=False,
        encoding="utf-8",
    )

    evaluator = GoldenSetEvaluator(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        col_answer_model="answer_model",
        col_expected_keywords="expected_keywords",
        col_expected_summary="expected_answer_summary",
        col_question="question",
        col_expected_answer="expected_answer_summary",
        col_latency="latency_seconds",
        col_difficulty="difficulty",
    )

    df_eval, summary = evaluator.evaluate(df_with_answers)

    df_eval.to_csv(
        "data/llm-only-eval.csv",
        index=False,
        encoding="utf-8",
    )

    GoldenSetEvaluator.append_summary_to_csv("llm_only", summary)

    return summary


def benchmark_rag():
    system_prompt = (
        """
        Ton but est de répondre en francais EXCLUSIVEMENT aux questions sur les sujets : état civil, urbanisme, déchets, transports, petite-enfance, social, vie associative, élections, logement, culture/sport, fiscalité, eau/assainissement.
        Régles OBLIGATOIRES :
        - Si la question est hors sujet, ou si tu n'as pas suffisement d'informations pour répondre, répond UNIQUEMENT cette phrase: "Bonjour, je suis désolé mais je ne suis pas en mesure de répondre à cette question."
        - Sinon, commence toujours par "Bonjour"
        Tu dois t'appuyer STRICTEMENT sur la FAQ fournie en contexte pour répondre. Ne mentionne JAMAIS la FAQ dans ta réponse.
        """
    )

    df_questions = extract_questions()
    faq_df = load_faq_base()

    runner = RAGRunner(
        faq_df=faq_df,
        question_col="question",
        answer_col="answer_model",
        latency_col="latency_seconds",
        top_k=5,
    )

    df_with_answers = runner.run_on_dataframe(
        df_questions,
        system_prompt=system_prompt,
        delay_seconds=1,
    )

    df_with_answers.to_csv(
        "data/rag-with-answers.csv",
        index=False,
        encoding="utf-8",
    )

    evaluator = GoldenSetEvaluator(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        col_answer_model="answer_model",
        col_expected_keywords="expected_keywords",
        col_expected_summary="expected_answer_summary",
        col_question="question",
        col_expected_answer="expected_answer_summary",
        col_latency="latency_seconds",
        col_difficulty="difficulty",
    )

    df_eval, summary = evaluator.evaluate(df_with_answers)

    df_eval.to_csv(
        "data/rag-eval.csv",
        index=False,
        encoding="utf-8",
    )

    GoldenSetEvaluator.append_summary_to_csv("rag", summary)

    return summary


def benchmark_extractive_qa():
    df_questions = extract_questions()
    faq_df = load_faq_base()

    runner = ExtractiveQARunner(
        faq_df=faq_df,
        question_col="question",
        answer_col="answer_model",
        latency_col="latency_seconds",
        top_k=15,
    )

    df_with_answers = runner.run_on_dataframe(
        df_questions,
        delay_seconds=1,
    )

    df_with_answers.to_csv(
        "data/extractive-qa-with-answers.csv",
        index=False,
        encoding="utf-8",
    )

    evaluator = GoldenSetEvaluator(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        col_answer_model="answer_model",
        col_expected_keywords="expected_keywords",
        col_expected_summary="expected_answer_summary",
        col_question="question",
        col_expected_answer="expected_answer_summary",
        col_latency="latency_seconds",
        col_difficulty="difficulty",
    )

    df_eval, summary = evaluator.evaluate(df_with_answers)

    df_eval.to_csv(
        "data/extractive-qa-eval.csv",
        index=False,
        encoding="utf-8",
    )

    GoldenSetEvaluator.append_summary_to_csv("extractive_qa", summary)

    return summary


if __name__ == "__main__":
    try:
        # print("=== Benchmark LLM-only ===")
        # summary_llm = benchmark_llm_only()

        print("\n=== Benchmark RAG ===")
        summary_rag = benchmark_rag()

        print("\n=== Benchmark QA extractif ===")
        summary_qa = benchmark_extractive_qa()

        print("\nÉvaluation terminée.")
    except Exception as e:
        print(f"Erreur pendant l'exécution du script : {e}")
