"""
Evaluation runner — runs the golden test set through the agent graph
and scores each case on all four metrics.

Output: JSON report with per-case scores + aggregate statistics.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.core.llm import LLMProvider
from src.evaluation.metrics import (
    compute_answer_relevancy,
    compute_faithfulness,
    compute_latency_percentiles,
    compute_retrieval_precision,
)

logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    """Result for a single golden test case."""
    case_id:             str
    query:               str
    expected_agent:      str
    actual_agent:        str
    agent_correct:       bool
    faithfulness:        float
    answer_relevancy:    float
    retrieval_precision: float
    latency_ms:          float
    answer_preview:      str     # first 200 chars
    retrieved_nct_ids:   list[str] = field(default_factory=list)
    error:               Optional[str] = None


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    total_cases:         int
    agent_accuracy:      float   # fraction routed to correct agent
    avg_faithfulness:    float
    avg_relevancy:       float
    avg_retrieval_prec:  float
    latency:             dict
    case_results:        list[CaseResult] = field(default_factory=list)
    config_snapshot:     dict = field(default_factory=dict)


class Evaluator:
    """
    Runs evaluation suite against the agent graph.

    Usage:
        evaluator = Evaluator(graph, llm, golden_set_path)
        report = evaluator.run()
        evaluator.save_report(report, "data/eval/report.json")
    """

    def __init__(
        self,
        graph,
        llm: LLMProvider,
        golden_set_path: Path = Path("data/eval/golden_set.json"),
    ):
        self.graph       = graph
        self.llm         = llm
        self.golden_cases = json.loads(golden_set_path.read_text())

    def run(self) -> EvalReport:
        """Run all golden cases and return an EvalReport."""
        logger.info(f"Running evaluation on {len(self.golden_cases)} cases...")
        case_results: list[CaseResult] = []

        for case in self.golden_cases:
            result = self._run_case(case)
            case_results.append(result)
            logger.info(
                f"[{case['id']}] agent={result.actual_agent} "
                f"faith={result.faithfulness:.2f} "
                f"rel={result.answer_relevancy:.2f} "
                f"prec={result.retrieval_precision:.2f} "
                f"lat={result.latency_ms:.0f}ms"
            )

        return self._aggregate(case_results)

    def _run_case(self, case: dict) -> CaseResult:
        """Run one golden test case."""
        start = time.monotonic()
        error = None

        try:
            state = self.graph.invoke({"query": case["query"]})
            latency_ms = (time.monotonic() - start) * 1000

            answer   = state.get("final_answer", "")
            route    = state.get("agent_route", "unknown")
            chunks   = state.get("retrieved_chunks", [])

            faithfulness = compute_faithfulness(
                case["query"], answer, chunks, self.llm
            )
            relevancy = compute_answer_relevancy(
                case["query"], answer, self.llm
            )
            precision = compute_retrieval_precision(
                chunks, case.get("expected_nct_ids", [])
            )
            retrieved_ids = list({h.chunk.nct_id for h in chunks})

        except Exception as e:
            latency_ms   = (time.monotonic() - start) * 1000
            error        = str(e)
            answer       = ""
            route        = "error"
            faithfulness = 0.0
            relevancy    = 0.0
            precision    = 0.0
            retrieved_ids = []
            logger.error(f"Case {case['id']} failed: {e}")

        return CaseResult(
            case_id=case["id"],
            query=case["query"],
            expected_agent=case["expected_agent"],
            actual_agent=route,
            agent_correct=(route == case["expected_agent"]),
            faithfulness=round(faithfulness, 3),
            answer_relevancy=round(relevancy, 3),
            retrieval_precision=round(precision, 3),
            latency_ms=round(latency_ms, 1),
            answer_preview=answer[:200],
            retrieved_nct_ids=retrieved_ids,
            error=error,
        )

    def _aggregate(self, results: list[CaseResult]) -> EvalReport:
        """Aggregate per-case results into an EvalReport."""
        n = len(results)
        if n == 0:
            return EvalReport(0, 0.0, 0.0, 0.0, 0.0, {})

        agent_correct   = sum(1 for r in results if r.agent_correct) / n
        avg_faith       = sum(r.faithfulness for r in results) / n
        avg_rel         = sum(r.answer_relevancy for r in results) / n
        avg_prec        = sum(r.retrieval_precision for r in results) / n
        latency_stats   = compute_latency_percentiles(
            [r.latency_ms for r in results]
        )

        from src.core.config import get_settings
        settings = get_settings()
        config_snapshot = {
            "llm_provider":   settings.llm_provider,
            "embedding_model": settings.vs.embedding_model,
            "vector_store":   settings.vector_store_type,
            "chunks_indexed": None,  # filled in by evaluate.py
        }

        return EvalReport(
            total_cases=n,
            agent_accuracy=round(agent_correct, 3),
            avg_faithfulness=round(avg_faith, 3),
            avg_relevancy=round(avg_rel, 3),
            avg_retrieval_prec=round(avg_prec, 3),
            latency=latency_stats,
            case_results=results,
            config_snapshot=config_snapshot,
        )

    @staticmethod
    def save_report(report: EvalReport, path: Path) -> None:
        """Save report to JSON for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert dataclasses to dicts
        data = asdict(report)
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Report saved to {path}")