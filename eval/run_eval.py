"""Score the system against its two golden sets.

1. `data/golden_set.csv` — per-message classification labels (sender_type/intent/topic).
   Reproduces the Phase 2 numbers in RESULTS.md, which had no runnable code behind them.
2. `data/qa_golden.csv` — question -> expected KB entry / decision. This is the
   product-level scoreboard: retrieval hit@1/hit@3, decision accuracy, escalation
   precision/recall (the headline metric for the "confident wrong answer" failure mode:
   a wrong answer shown confidently is an escalation-RECALL miss), and topic accuracy of
   the LLM few-shot classifier (`_llm_few_shot_classify`, the same function
   `server.py` falls back to for spike-monitor topic tagging when nothing was
   retrieved).

Usage:
    LLM_BACKEND=ollama PYTHONPATH=. python eval/run_eval.py

Requires a live Ollama server (LLM access is a hard requirement, no rule-based
fallback path exists anymore).

Decision-accuracy scoring rule: `expected_decision` in qa_golden.csv is a FLOOR, not an
exact target. A curated question is hand-labeled "suggest_to_staff" because that is the
minimum acceptable outcome (grounded in the composite-score calibration in RESULTS.md:
known-correct paraphrased queries measured 0.62-0.76, safely in that band) -- getting
`auto_reply` instead is a better outcome, not a miss. `escalate` rows require an exact
match: nothing beats correctly declining to answer.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.graph import build_graph, run_graph
from src.alerts.console_alerter import ConsoleAlerter
from src.config import load_settings
from src.kb.store import KBStore
from src.llm.ollama_llm import OllamaLLM
from src.pipeline.classify import _llm_few_shot_classify, classify
from src.pipeline.synthesize_kb import load_curated_kb

REPO_ROOT = Path(__file__).resolve().parent.parent
QA_GOLDEN_CSV = REPO_ROOT / "data" / "qa_golden.csv"
CLASSIFY_GOLDEN_CSV = REPO_ROOT / "data" / "golden_set.csv"
CURATED_KB_CSV = REPO_ROOT / "data" / "curated_kb.csv"
CLASSIFY_CACHE = REPO_ROOT / "data" / "processed" / "classify_cache.json"

# decisions ranked worst -> best; used to score a hand-labeled floor as "met or beaten".
# Every qa_golden.csv row is a real technical question, so `direct_reply` (the
# chit_chat/off_topic short-circuit) is always a turn-type misclassification on this
# set -- ranked below escalate, since escalate at least correctly recognizes "this
# needs a human" while direct_reply fails to engage with the question at all.
_DECISION_RANK = {"direct_reply": -1, "escalate": 0, "suggest_to_staff": 1, "auto_reply": 2}

DECISIONS = ("auto_reply", "suggest_to_staff", "escalate", "direct_reply")


# --------------------------------------------------------------------------------------
# Pure scoring functions (no store/LLM/network) -- these are what tests exercise directly.
# --------------------------------------------------------------------------------------


@dataclass
class QARowResult:
    question: str
    expected_topic: str
    expected_source: str
    expected_decision: str
    hit_at_1: bool | None  # None when not applicable (expected_source == NONE)
    hit_at_3: bool | None
    decision_correct: bool
    actual_decision: str
    actual_topic: str
    topic_correct: bool
    should_escalate: bool
    did_escalate: bool


def score_retrieval(expected_snippet: str, retrieved: list, top_n: int) -> bool:
    """True if `expected_snippet` appears (case-insensitive substring) in any of the
    top-`top_n` retrieved entries' problem text."""
    if not expected_snippet:
        return False
    needle = expected_snippet.lower()
    for scored in retrieved[:top_n]:
        if needle in scored.entry.problem.lower():
            return True
    return False


def score_decision(expected_decision: str, actual_decision: str) -> bool:
    """expected_decision is a floor: actual must be >= expected on the
    escalate < suggest_to_staff < auto_reply scale. `escalate` requires an exact match."""
    if expected_decision == "escalate":
        return actual_decision == "escalate"
    return _DECISION_RANK[actual_decision] >= _DECISION_RANK[expected_decision]


def score_qa_row(row: dict, retrieved: list, decision: str, actual_topic: str) -> QARowResult:
    expected_source = row["expected_source"]
    should_escalate = expected_source == "NONE"
    # rows with expected_topic == "none" are genuinely off-topic/chit_chat (weather,
    # food, gold prices) -- for those, direct_reply IS the correct decline, same as
    # escalate. Rows with a real expected_topic are on-domain KB gaps (e.g. password
    # reset, PDF export): a direct_reply there means the turn-classifier wrongly called
    # a real technical question chit_chat/off_topic, which is a miss, not a correct
    # decline.
    is_off_topic_row = row["expected_topic"] == "none"

    if should_escalate:
        hit_1 = hit_3 = None
    else:
        hit_1 = score_retrieval(row["expected_kb_snippet"], retrieved, top_n=1)
        hit_3 = score_retrieval(row["expected_kb_snippet"], retrieved, top_n=3)

    did_decline = decision == "escalate" or (decision == "direct_reply" and is_off_topic_row)

    return QARowResult(
        question=row["question"],
        expected_topic=row["expected_topic"],
        expected_source=expected_source,
        expected_decision=row["expected_decision"],
        hit_at_1=hit_1,
        hit_at_3=hit_3,
        decision_correct=score_decision(row["expected_decision"], decision) if not is_off_topic_row
        else did_decline,
        actual_decision=decision,
        actual_topic=actual_topic,
        topic_correct=actual_topic == row["expected_topic"],
        should_escalate=should_escalate,
        did_escalate=did_decline,
    )


@dataclass
class QASummary:
    n: int = 0
    hit_1_n: int = 0
    hit_1_total: int = 0
    hit_3_n: int = 0
    hit_3_total: int = 0
    decision_correct_n: int = 0
    topic_correct_n: int = 0
    # escalation confusion counts
    true_positive: int = 0  # should escalate, did escalate
    false_negative: int = 0  # should escalate, did NOT (confident-wrong-answer risk)
    false_positive: int = 0  # should NOT escalate, did (over-cautious)
    true_negative: int = 0  # should NOT escalate, did NOT
    rows: list[QARowResult] = field(default_factory=list)

    @property
    def hit_at_1(self) -> float:
        return self.hit_1_n / self.hit_1_total if self.hit_1_total else float("nan")

    @property
    def hit_at_3(self) -> float:
        return self.hit_3_n / self.hit_3_total if self.hit_3_total else float("nan")

    @property
    def decision_accuracy(self) -> float:
        return self.decision_correct_n / self.n if self.n else float("nan")

    @property
    def topic_accuracy(self) -> float:
        return self.topic_correct_n / self.n if self.n else float("nan")

    @property
    def escalation_precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else float("nan")

    @property
    def escalation_recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else float("nan")


def summarize_qa(results: list[QARowResult]) -> QASummary:
    s = QASummary()
    for r in results:
        s.n += 1
        s.rows.append(r)
        if r.hit_at_1 is not None:
            s.hit_1_total += 1
            s.hit_1_n += int(r.hit_at_1)
        if r.hit_at_3 is not None:
            s.hit_3_total += 1
            s.hit_3_n += int(r.hit_at_3)
        s.decision_correct_n += int(r.decision_correct)
        s.topic_correct_n += int(r.topic_correct)

        if r.should_escalate and r.did_escalate:
            s.true_positive += 1
        elif r.should_escalate and not r.did_escalate:
            s.false_negative += 1
        elif not r.should_escalate and r.did_escalate:
            s.false_positive += 1
        else:
            s.true_negative += 1
    return s


# --------------------------------------------------------------------------------------
# Wiring (store / LLM / graph) -- mirrors src/server.py so the eval matches production.
# --------------------------------------------------------------------------------------


def _build_store(settings) -> KBStore:
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
    except Exception:
        client = QdrantClient(":memory:")
    return KBStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=f"{settings.qdrant_collection}_eval",
        embedding_model=settings.embedding_model,
        embedding_model_fallback=settings.embedding_model_fallback,
        client=client,
    )


def _build_llm(settings):
    if settings.llm_backend == "ollama":
        return OllamaLLM(host=settings.ollama_host, model=settings.ollama_model)
    raise RuntimeError(
        f"Unknown or unconfigured LLM_BACKEND={settings.llm_backend!r} -- LLM access is "
        "a hard requirement now. Set LLM_BACKEND=ollama (and have `ollama serve` running)."
    )


def run_qa_eval(settings) -> QASummary:
    store = _build_store(settings)
    load_curated_kb(store, CURATED_KB_CSV)
    llm = _build_llm(settings)
    graph = build_graph(store, llm, ConsoleAlerter())

    df = pd.read_csv(QA_GOLDEN_CSV)
    results = []
    for _, row in df.iterrows():
        state = run_graph(graph, str(row["question"]))
        # mirrors server.py's spike-tagging rule: prefer the matched entry's topic
        # (free), only spend an LLM call classifying the raw question if nothing hit.
        if state.retrieved:
            actual_topic = state.retrieved[0].entry.topic
        else:
            _, _, actual_topic, _ = _llm_few_shot_classify(str(row["question"]), llm)
        results.append(
            score_qa_row(
                row.to_dict(),
                state.retrieved,
                state.decision or "escalate",
                actual_topic,
            )
        )
    return summarize_qa(results)


def run_classify_eval(llm) -> dict:
    df = pd.read_csv(CLASSIFY_GOLDEN_CSV)
    labeled = df.dropna(subset=["sender_type"]).copy()
    predicted = classify(labeled[["content"]].reset_index(drop=True), llm, CLASSIFY_CACHE)

    metrics = {}
    for field_name in ("sender_type", "intent", "topic"):
        correct = (predicted[field_name].values == labeled[field_name].values).sum()
        metrics[field_name] = correct / len(labeled) if len(labeled) else float("nan")
    return metrics


def _print_qa_summary(label: str, s: QASummary) -> None:
    print(f"\n--- QA eval ({label}) ---")
    print(f"rows: {s.n}")
    print(f"retrieval hit@1: {s.hit_at_1:.2f} ({s.hit_1_n}/{s.hit_1_total})")
    print(f"retrieval hit@3: {s.hit_at_3:.2f} ({s.hit_3_n}/{s.hit_3_total})")
    print(f"decision accuracy: {s.decision_accuracy:.2f} ({s.decision_correct_n}/{s.n})")
    print(f"topic accuracy: {s.topic_accuracy:.2f} ({s.topic_correct_n}/{s.n})")
    print(
        f"escalation precision: {s.escalation_precision:.2f}  recall: {s.escalation_recall:.2f}  "
        f"(TP={s.true_positive} FP={s.false_positive} FN={s.false_negative} TN={s.true_negative})"
    )
    if s.false_negative:
        print("  FN rows (should have escalated but got a confident answer -- highest risk):")
        for r in s.rows:
            if r.should_escalate and not r.did_escalate:
                print(f"    - {r.question!r} -> {r.actual_decision}")


def main() -> None:
    settings = load_settings()

    print("=" * 60)
    print(f"LLM_BACKEND={settings.llm_backend}")
    print("=" * 60)

    t0 = time.time()
    qa_summary = run_qa_eval(settings)
    _print_qa_summary(settings.llm_backend, qa_summary)
    print(f"(qa eval took {time.time() - t0:.1f}s)")

    print(f"\n--- Classifier eval ({settings.llm_backend}, vs data/golden_set.csv) ---")
    if CLASSIFY_GOLDEN_CSV.exists():
        llm = _build_llm(settings)
        classify_metrics = run_classify_eval(llm)
        for field_name, acc in classify_metrics.items():
            print(f"{field_name}: accuracy={acc:.3f}")
    else:
        print(f"skipped -- {CLASSIFY_GOLDEN_CSV} not found (private, not in repo)")


if __name__ == "__main__":
    main()
