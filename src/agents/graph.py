"""Wires orchestrator -> retrieval -> answerer -> critic into a LangGraph StateGraph.

orchestrate classifies turn_type first; chit_chat/off_topic route straight to
direct_reply (no KB search, no escalation). new_problem/follow_up continue through
retrieve -> answer -> critic, with a conditional edge on the gate decision
(auto-reply / suggest-to-staff / escalate). On escalate, the Alerter is fired with a
structured escalation record.
"""
from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from src.agents.answerer import answer, direct_reply
from src.agents.critic import critique
from src.agents.orchestrator import orchestrate
from src.agents.retrieval import retrieve
from src.agents.state import AgentState, Turn
from src.alerts.base import AlertRecord, Alerter
from src.config import GateThresholds
from src.kb.store import KBStore
from src.llm.base import LLMClient

DEFAULT_TOP_K = 3


def build_graph(
    store: KBStore,
    llm: LLMClient,
    alerter: Alerter,
    thresholds: GateThresholds | None = None,
    top_k: int = DEFAULT_TOP_K,
):
    thresholds = thresholds or GateThresholds()

    def orchestrate_node(state: AgentState) -> AgentState:
        return orchestrate(state, llm)

    def retrieve_node(state: AgentState) -> AgentState:
        return retrieve(state, store, llm, top_k)

    def answer_node(state: AgentState) -> AgentState:
        return answer(state, llm)

    def direct_reply_node(state: AgentState) -> AgentState:
        return direct_reply(state, llm)

    def critic_node(state: AgentState) -> AgentState:
        return critique(state, llm, thresholds)

    def escalate_node(state: AgentState) -> AgentState:
        top_topic = state.retrieved[0].entry.topic if state.retrieved else "unknown"
        alerter.send(
            AlertRecord(
                topic=top_topic,
                message=(
                    f"Escalation: could not answer with confidence "
                    f"(conf={state.confidence:.2f}). Question: {state.question}"
                ),
                severity="escalate",
                triggered_at=time.time() * 1000,
            )
        )
        return state

    graph = StateGraph(AgentState)
    graph.add_node("orchestrate", orchestrate_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("direct_reply", direct_reply_node)
    graph.add_node("critic", critic_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("orchestrate")

    # chit_chat/off_topic skip the KB entirely; new_problem/follow_up go through
    # retrieval as before.
    graph.add_conditional_edges(
        "orchestrate",
        lambda state: state.turn_type,
        {
            "new_problem": "retrieve",
            "follow_up": "retrieve",
            "chit_chat": "direct_reply",
            "off_topic": "direct_reply",
        },
    )
    graph.add_edge("direct_reply", END)

    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "critic")

    # conditional routing on the gate decision: escalate fires the alerter then ends;
    # auto_reply and suggest_to_staff both just end (the decision is on the state for
    # the caller to act on).
    graph.add_conditional_edges(
        "critic",
        lambda state: state.decision,
        {
            "auto_reply": END,
            "suggest_to_staff": END,
            "escalate": "escalate",
        },
    )
    graph.add_edge("escalate", END)

    return graph.compile()


def run_graph(graph, question: str, history: list[Turn] | None = None) -> AgentState:
    result = graph.invoke(AgentState(question=question, history=history or []))
    # langgraph returns the state as a dict-like; normalize back to AgentState
    return AgentState.model_validate(result)
