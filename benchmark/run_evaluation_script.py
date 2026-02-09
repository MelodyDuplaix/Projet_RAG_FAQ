import pandas as pd
from data_access import extract_questions, load_faq_base
from methods_runners import LLMOnlyRunner, RAGRunner, ExtractiveQARunner
import json
from datetime import datetime


def run_all_benchmarks():
    all_results = []
    df_questions = extract_questions()
    faq_df = load_faq_base()

    # --- LLM Only Benchmark ---
    system_prompt_llm = (
        """
        Tu es un assistant municipal francais expert de la communauté de communes Val de Loire Numérique.
        Ton but est de répondre EXCLUSIVEMENT aux questions sur les sujets : état civil, urbanisme, déchets, transports, petite-enfance, social, vie associative, élections, logement, culture/sport, fiscalité, eau/assainissement.
        Régles OBLIGATOIRES :
        - Si la question est hors sujet, ou si tu n'as pas suffisement d'informations pour répondre, répond UNIQUEMENT cette phrase: "Bonjour, je suis désolé mais je ne suis pas en mesure de répondre à cette question."
        - Sinon, commence toujours par "Bonjour"
        """
    )
    llm_runner = LLMOnlyRunner(
        question_col="question",
        answer_col="answer_model",
        latency_col="latency_seconds",
    )
    print("=== Exécution du benchmark LLM-only ===")
    df_llm_results = llm_runner.run_on_dataframe(
        df_questions,
        system_prompt=system_prompt_llm,
        delay_seconds=1,
    )
    all_results.extend(df_llm_results.to_dict(orient="records"))

    # --- RAG Benchmark ---
    system_prompt_rag = (
        """
        Ton but est de répondre en francais EXCLUSIVEMENT aux questions sur les sujets : état civil, urbanisme, déchets, transports, petite-enfance, social, vie associative, élections, logement, culture/sport, fiscalité, eau/assainissement.
        Régles OBLIGATOIRES :
        - Si la question est hors sujet, ou si tu n'as pas suffisement d'informations pour répondre, répond UNIQUEMENT cette phrase: "Bonjour, je suis désolé mais je ne suis pas en mesure de répondre à cette question."
        - Sinon, commence toujours par "Bonjour"
        Tu dois t'appuyer STRICTEMENT sur la FAQ fournie en contexte pour répondre. Ne mentionne JAMAIS la FAQ dans ta réponse.
        """
    )
    rag_runner = RAGRunner(
        faq_df=faq_df,
        question_col="question",
        answer_col="answer_model",
        latency_col="latency_seconds",
        top_k=5,
    )
    print("\n=== Exécution du benchmark RAG ===")
    df_rag_results = rag_runner.run_on_dataframe(
        df_questions,
        system_prompt=system_prompt_rag,
        delay_seconds=1,
    )
    all_results.extend(df_rag_results.to_dict(orient="records"))

    # --- Extractive QA Benchmark ---
    qa_runner = ExtractiveQARunner(
        faq_df=faq_df,
        question_col="question",
        answer_col="answer_model",
        latency_col="latency_seconds",
        top_k=15,
    )
    print("\n=== Exécution du benchmark QA extractif ===")
    df_qa_results = qa_runner.run_on_dataframe(
        df_questions,
        delay_seconds=1,
    )
    all_results.extend(df_qa_results.to_dict(orient="records"))

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"data/benchmark_results_{timestamp}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"\nBenchmark terminé. Résultats sauvegardés dans {output_filename}")


if __name__ == "__main__":
    try:
        run_all_benchmarks()
    except Exception as e:
        print(f"Erreur pendant l'exécution du script : {e}")
