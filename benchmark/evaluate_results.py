import pandas as pd
import json
import argparse
from evaluate_answers import GoldenSetEvaluator
import os # Import os for file path checks

def evaluate_benchmark_results(json_file_path):
    if not os.path.exists(json_file_path):
        print(f"Erreur: Le fichier '{json_file_path}' n'existe pas.")
        return

    with open(json_file_path, "r", encoding="utf-8") as f:
        all_benchmark_results = json.load(f)

    df_all_results = pd.DataFrame(all_benchmark_results)

    unique_strategies = df_all_results["strategy"].unique()

    all_df_evals = []
    all_summaries = {}

    print("=== Lancement de l'évaluation ===")

    for strategy_name in unique_strategies:
        print(f"--- Évaluation de la stratégie : {strategy_name} ---")
        df_strategy = df_all_results[df_all_results["strategy"] == strategy_name].copy()

        evaluator = GoldenSetEvaluator(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            col_answer_model="answer_model",
            col_expected_keywords="expected_keywords",
            col_expected_summary="expected_answer_summary",
            col_question="question",
            col_expected_answer="expected_answer_summary",
            col_latency="latency_seconds",
        )

        df_eval, summary = evaluator.evaluate(df_strategy)
        df_eval["strategy"] = strategy_name # Add strategy column to detailed eval results
        all_df_evals.append(df_eval)
        all_summaries[strategy_name] = summary

    # Consolidate and save detailed evaluation results
    df_final_eval = pd.concat(all_df_evals, ignore_index=True)
    output_eval_csv = "data/evaluation_results.csv"
    df_final_eval.to_csv(output_eval_csv, index=False, encoding="utf-8")
    print(f"Résultats d'évaluation détaillés sauvegardés dans {output_eval_csv}")

    # Save summary report to CSV
    df_summary = pd.DataFrame.from_dict(all_summaries, orient='index')
    df_summary.index.name = 'method' # Add index name as 'method'
    output_report_csv = "data/evaluation_report.csv"
    df_summary.to_csv(output_report_csv, encoding="utf-8")
    print(f"Rapport de synthèse sauvegardé dans {output_report_csv}")

    # Provide recommendation
    print("=== Recommandation de la meilleure stratégie ===")
    best_strategy = None
    max_global_score = -1.0

    for strategy, summary in all_summaries.items():
        print(f"- Stratégie '{strategy}': Score global = {summary.get('global_score', 0.0):.4f}")
        if summary.get("global_score", 0.0) > max_global_score:
            max_global_score = summary["global_score"]
            best_strategy = strategy

    if best_strategy:
        print(f"La stratégie recommandée est : '{best_strategy}' avec un score global de {max_global_score:.4f}")
    else:
        print("Impossible de déterminer la meilleure stratégie.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue les résultats d'un benchmark de stratégies de réponse.")
    parser.add_argument("benchmark_json_file", type=str,
                        help="Chemin vers le fichier JSON contenant les résultats bruts du benchmark.")
    args = parser.parse_args()

    evaluate_benchmark_results(args.benchmark_json_file)