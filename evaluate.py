"""
Run the evaluation suite.

Usage:
    python evaluate.py
    python evaluate.py --output data/eval/my_report.json
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.agents.graph import build_graph
from src.core.chroma_store import ChromaStore
from src.core.config import get_settings
from src.core.embeddings import HuggingFaceEmbeddings
from src.core.llm import get_llm_provider
from src.core.retriever import Retriever
from src.evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/eval/report.json",
        help="Path to save the JSON report",
    )
    parser.add_argument(
        "--golden-set",
        default="data/eval/golden_set.json",
        help="Path to the golden test set",
    )
    args = parser.parse_args()

    settings  = get_settings()
    llm       = get_llm_provider()
    embedder  = HuggingFaceEmbeddings(model_name=settings.vs.embedding_model)
    store     = ChromaStore(
        persist_dir=Path(settings.vs.chroma_persist_dir),
        collection_name=settings.vs.collection_name,
    )
    retriever = Retriever(embedder, store)
    graph     = build_graph(llm, retriever)

    evaluator = Evaluator(
        graph=graph,
        llm=llm,
        golden_set_path=Path(args.golden_set),
    )

    print(f"\n🔬 Running evaluation on {args.golden_set}...")
    report = evaluator.run()

    # Fill in chunk count
    report.config_snapshot["chunks_indexed"] = store.count()

    # Save
    Evaluator.save_report(report, Path(args.output))

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total cases       : {report.total_cases}")
    print(f"Agent accuracy    : {report.agent_accuracy:.1%}")
    print(f"Avg faithfulness  : {report.avg_faithfulness:.3f}")
    print(f"Avg relevancy     : {report.avg_relevancy:.3f}")
    print(f"Avg retrieval prec: {report.avg_retrieval_prec:.3f}")
    print(f"\nLatency (ms):")
    print(f"  p50 : {report.latency['p50']:.1f}")
    print(f"  p95 : {report.latency['p95']:.1f}")
    print(f"  p99 : {report.latency['p99']:.1f}")
    print(f"  mean: {report.latency['mean']:.1f}")
    print(f"\nPer-case routing:")
    for r in report.case_results:
        mark = "✅" if r.agent_correct else "❌"
        print(f"  {mark} [{r.case_id}] expected={r.expected_agent} "
              f"actual={r.actual_agent} prec={r.retrieval_precision:.2f}")
    print(f"\n📄 Report saved to: {args.output}")


if __name__ == "__main__":
    main()