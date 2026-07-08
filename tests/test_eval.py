"""Unit tests for the eval scoring functions (no embeddings, no store — pure functions
over fixture data, per conftest's fast tier)."""
from __future__ import annotations

import math

from tests.conftest import make_scored

from eval.run_eval import score_decision, score_qa_row, score_retrieval, summarize_qa


def test_score_retrieval_hit_case_insensitive():
    retrieved = [make_scored("Workflow mất Publish do vmail", "s", "workflow_publish", 0.8)]
    assert score_retrieval("MẤT PUBLISH", retrieved, top_n=1) is True


def test_score_retrieval_miss_when_snippet_absent():
    retrieved = [make_scored("credential hết hạn", "s", "credential", 0.8)]
    assert score_retrieval("mất publish", retrieved, top_n=1) is False


def test_score_retrieval_respects_top_n():
    retrieved = [
        make_scored("credential hết hạn", "s", "credential", 0.8),
        make_scored("workflow mất publish", "s", "workflow_publish", 0.5),
    ]
    assert score_retrieval("mất publish", retrieved, top_n=1) is False
    assert score_retrieval("mất publish", retrieved, top_n=3) is True


def test_score_decision_floor_semantics():
    # expected suggest_to_staff: auto_reply (better) and suggest_to_staff (exact) both pass
    assert score_decision("suggest_to_staff", "auto_reply") is True
    assert score_decision("suggest_to_staff", "suggest_to_staff") is True
    # but escalate (worse than the floor) fails
    assert score_decision("suggest_to_staff", "escalate") is False


def test_score_decision_escalate_requires_exact_match():
    # a confident answer when escalation was expected must NOT count as correct, no
    # matter how "good" auto_reply looks on the ranking scale -- this is the headline
    # confident-wrong-answer risk the eval exists to catch.
    assert score_decision("escalate", "suggest_to_staff") is False
    assert score_decision("escalate", "auto_reply") is False
    assert score_decision("escalate", "escalate") is True


def test_score_qa_row_none_expected_skips_retrieval_scoring():
    row = {
        "question": "q",
        "expected_topic": "none",
        "expected_source": "NONE",
        "expected_kb_snippet": "",
        "expected_decision": "escalate",
    }
    retrieved = [make_scored("unrelated", "s", "other", 0.9)]
    result = score_qa_row(row, retrieved, decision="escalate", actual_topic="none")
    assert result.hit_at_1 is None
    assert result.hit_at_3 is None
    assert result.should_escalate is True
    assert result.did_escalate is True


def test_score_qa_row_curated_expected_scores_retrieval():
    row = {
        "question": "q",
        "expected_topic": "workflow_publish",
        "expected_source": "curated",
        "expected_kb_snippet": "mất publish",
        "expected_decision": "suggest_to_staff",
    }
    retrieved = [make_scored("workflow mất publish", "s", "workflow_publish", 0.8)]
    result = score_qa_row(row, retrieved, decision="suggest_to_staff", actual_topic="workflow_publish")
    assert result.hit_at_1 is True
    assert result.should_escalate is False
    assert result.topic_correct is True


def test_summarize_qa_escalation_confusion_counts():
    rows = [
        # should escalate, did escalate -> TP
        score_qa_row(
            {"question": "a", "expected_topic": "none", "expected_source": "NONE",
             "expected_kb_snippet": "", "expected_decision": "escalate"},
            [], "escalate", "none",
        ),
        # should escalate, did NOT -> FN (the confident-wrong-answer risk case)
        score_qa_row(
            {"question": "b", "expected_topic": "credential", "expected_source": "NONE",
             "expected_kb_snippet": "", "expected_decision": "escalate"},
            [make_scored("p", "s", "credential", 0.9)], "suggest_to_staff", "credential",
        ),
        # should NOT escalate, did NOT -> TN
        score_qa_row(
            {"question": "c", "expected_topic": "workflow_publish", "expected_source": "curated",
             "expected_kb_snippet": "mất publish", "expected_decision": "suggest_to_staff"},
            [make_scored("workflow mất publish", "s", "workflow_publish", 0.8)],
            "suggest_to_staff", "workflow_publish",
        ),
    ]
    summary = summarize_qa(rows)
    assert summary.true_positive == 1
    assert summary.false_negative == 1
    assert summary.true_negative == 1
    assert summary.false_positive == 0
    assert summary.escalation_recall == 0.5  # 1 TP / (1 TP + 1 FN)
    assert summary.escalation_precision == 1.0  # 1 TP / (1 TP + 0 FP)


def test_summarize_qa_empty_gives_nan_not_crash():
    summary = summarize_qa([])
    assert math.isnan(summary.hit_at_1)
    assert math.isnan(summary.decision_accuracy)
    assert math.isnan(summary.escalation_recall)


def test_score_qa_row_direct_reply_on_off_topic_row_is_correct_decline():
    # expected_topic == "none" marks a genuinely off-topic row (weather/food/etc) --
    # direct_reply there is a correct decline, same as escalate.
    row = {
        "question": "thời tiết hôm nay thế nào",
        "expected_topic": "none",
        "expected_source": "NONE",
        "expected_kb_snippet": "",
        "expected_decision": "escalate",
    }
    result = score_qa_row(row, [], decision="direct_reply", actual_topic="none")
    assert result.did_escalate is True
    assert result.decision_correct is True


def test_score_qa_row_direct_reply_on_on_domain_gap_row_is_a_miss():
    # expected_topic is a real topic -> this is an on-domain KB gap, not chit_chat.
    # direct_reply here means the turn-classifier wrongly called a real technical
    # question off_topic/chit_chat -- a miss, not a correct decline.
    row = {
        "question": "làm sao đổi mật khẩu đăng nhập",
        "expected_topic": "credential",
        "expected_source": "NONE",
        "expected_kb_snippet": "",
        "expected_decision": "escalate",
    }
    result = score_qa_row(row, [], decision="direct_reply", actual_topic="credential")
    assert result.did_escalate is False
    assert result.decision_correct is False
