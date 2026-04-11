"""
Eval router — POST /eval/run

Runs the agentic pipeline against 5 hardcoded ground-truth fixtures
and reports precision/recall/F1 for each extraction category.
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Set

from fastapi import APIRouter
from app import schemas
from app.services import agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eval", tags=["eval"])

FIXTURES_PATH = Path(__file__).parent.parent.parent / "tests" / "eval_fixtures.json"


def _normalize(text: str) -> str:
    """Lowercase + strip for fuzzy matching."""
    return text.lower().strip()


def _token_set(items: List[dict], key: str) -> Set[str]:
    return {_normalize(i.get(key, "")) for i in items if i.get(key)}


def _f1(predicted: Set[str], ground_truth: Set[str]) -> dict:
    if not ground_truth and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def _score_case(predicted: dict, expected: dict) -> dict:
    scores = {}
    for category, key in [
        ("decisions", "text"),
        ("action_items", "text"),
        ("conflicts", "issue"),
        ("unresolved_questions", "question"),
    ]:
        pred_set = _token_set(predicted.get(category, []), key)
        exp_set = _token_set(expected.get(category, []), key)

        # Partial match: predicted item counts as hit if it shares >= 3 words with any ground truth
        matched_pred = set()
        matched_exp = set()
        for p in pred_set:
            p_words = set(p.split())
            for e in exp_set:
                e_words = set(e.split())
                if len(p_words & e_words) >= 3:
                    matched_pred.add(p)
                    matched_exp.add(e)

        scores[category] = _f1(matched_pred, exp_set)
    return scores


@router.post("/run", response_model=schemas.EvalReport)
def run_eval():
    """Run ground-truth evaluation against all fixtures. Returns F1 per category."""
    with open(FIXTURES_PATH) as f:
        fixtures = json.load(f)

    all_scores = {cat: [] for cat in ["decisions", "action_items", "conflicts", "unresolved_questions"]}
    latencies = []
    providers_used = []

    for fixture in fixtures:
        t0 = time.time()
        result = agent.run_pipeline(fixture["transcript"])
        elapsed = (time.time() - t0) * 1000
        latencies.append(round(elapsed, 1))
        providers_used.append(result.get("llm_provider", "unknown"))

        case_scores = _score_case(result, fixture["expected"])
        for cat, score in case_scores.items():
            all_scores[cat].append(score)

    def avg_metric(scores_list: list, metric: str) -> float:
        return round(sum(s[metric] for s in scores_list) / len(scores_list), 3) if scores_list else 0.0

    def make_metrics(cat: str) -> schemas.EvalMetrics:
        sl = all_scores[cat]
        return schemas.EvalMetrics(
            category=cat,
            precision=avg_metric(sl, "precision"),
            recall=avg_metric(sl, "recall"),
            f1=avg_metric(sl, "f1"),
        )

    return schemas.EvalReport(
        num_cases=len(fixtures),
        decisions=make_metrics("decisions"),
        action_items=make_metrics("action_items"),
        conflicts=make_metrics("conflicts"),
        unresolved_questions=make_metrics("unresolved_questions"),
        latency_ms=latencies,
        avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        llm_providers_used=providers_used,
    )
