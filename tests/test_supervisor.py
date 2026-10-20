"""Supervisor 路由规则单测（确定性，不调 LLM）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.supervisor import supervisor_node, route_after_guard, _needs_rules


def test_route_guard_refuse_to_synthesis():
    assert route_after_guard({"gate": "REFUSE"}) == "synthesis"


def test_route_guard_escalate_to_ticket():
    assert route_after_guard({"gate": "ESCALATE"}) == "ticket"


def test_route_guard_pass_to_supervisor():
    assert route_after_guard({"gate": "PASS"}) == "supervisor"


def test_supervisor_first_call_goes_knowledge():
    """PASS 且无 rag_evidence → knowledge"""
    r = supervisor_node({"gate": "PASS", "iterations": 0, "rag_evidence": None})
    assert r["next_agent"] == "knowledge"


def test_supervisor_high_conf_no_number_to_synthesis():
    """高置信且无数字信号 → synthesis 直答"""
    r = supervisor_node({
        "gate": "PASS", "iterations": 1,
        "rag_evidence": {"best_sim": 0.8}, "confidence_level": "high",
        "query": "大额转账怎么预约",
    })
    assert r["next_agent"] == "synthesis"


def test_supervisor_high_conf_with_number_to_rules():
    """高置信但含数字 → 必须查表"""
    r = supervisor_node({
        "gate": "PASS", "iterations": 1,
        "rag_evidence": {"best_sim": 0.8}, "confidence_level": "high",
        "query": "信用卡分12期手续费多少",
    })
    assert r["next_agent"] == "rules"


def test_supervisor_mid_conf_to_rules():
    r = supervisor_node({
        "gate": "PASS", "iterations": 1,
        "rag_evidence": {"best_sim": 0.5}, "confidence_level": "mid",
        "query": "转账限额", "rule_evidence": [],
    })
    assert r["next_agent"] == "rules"


def test_supervisor_mid_conf_with_rules_to_synthesis():
    r = supervisor_node({
        "gate": "PASS", "iterations": 2,
        "rag_evidence": {"best_sim": 0.5}, "confidence_level": "mid",
        "query": "转账限额", "rule_evidence": [{"tool": "x"}],
    })
    assert r["next_agent"] == "synthesis"


def test_supervisor_low_conf_no_rules_to_rules():
    r = supervisor_node({
        "gate": "PASS", "iterations": 1,
        "rag_evidence": {"best_sim": 0.2}, "confidence_level": "low",
        "query": "随便问问", "rule_evidence": [],
    })
    assert r["next_agent"] == "rules"


def test_supervisor_low_conf_with_rules_to_ticket():
    """低置信且已查过 rules 仍不够 → ticket 兜底"""
    r = supervisor_node({
        "gate": "PASS", "iterations": 2,
        "rag_evidence": {"best_sim": 0.2}, "confidence_level": "low",
        "query": "随便问问", "rule_evidence": [{"tool": "x"}],
    })
    assert r["next_agent"] == "ticket"


def test_supervisor_max_iterations_force_synthesis():
    r = supervisor_node({"gate": "PASS", "iterations": 10, "rag_evidence": {"best_sim": 0.3}})
    assert r["next_agent"] == "synthesis"


def test_needs_rules_number():
    assert _needs_rules({"query": "利率多少"})
    assert _needs_rules({"query": "分12期"})
    assert _needs_rules({"query": "转60万"})


def test_needs_rules_condition():
    assert _needs_rules({"query": "能不能代办"})
    assert _needs_rules({"query": "要不要预约"})


def test_needs_rules_false():
    assert not _needs_rules({"query": "大额转账怎么预约"})
