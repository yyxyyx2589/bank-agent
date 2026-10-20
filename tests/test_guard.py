"""Guard 守卫节点单测（纯规则，不调 LLM）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.guard import guard_node


def test_refuse_stock_recommendation():
    r = guard_node({"query": "你们能推荐哪只基金吗", "trace": []})
    assert r["gate"] == "REFUSE"
    assert "行外护栏" in r["gate_reason"]


def test_refuse_explicit_saiqu():
    r = guard_node({"query": "帮我荐股", "trace": []})
    assert r["gate"] == "REFUSE"


def test_refuse_guarantee_return():
    r = guard_node({"query": "这个理财保证收益吗", "trace": []})
    assert r["gate"] == "REFUSE"


def test_escalate_fund_safety():
    r = guard_node({"query": "我卡被盗刷了，有几千块没了", "trace": []})
    assert r["gate"] == "ESCALATE"
    assert r["next_agent"] == "ticket"
    assert r["ticket_intent"]["suggested_team"] == "反欺诈组"
    assert r["ticket_intent"]["priority"] == "紧急"


def test_escalate_complaint():
    r = guard_node({"query": "我要投诉你们态度差", "trace": []})
    assert r["gate"] == "ESCALATE"
    assert r["ticket_intent"]["suggested_team"] == "投诉处理组"


def test_escalate_request_human():
    r = guard_node({"query": "给我转人工", "trace": []})
    assert r["gate"] == "ESCALATE"
    assert r["ticket_intent"]["priority"] == "常规"


def test_pass_normal_business():
    r = guard_node({"query": "信用卡分12期手续费多少", "trace": []})
    assert r["gate"] == "PASS"
    assert "ticket_intent" not in r


def test_pass开户材料():
    r = guard_node({"query": "开卡需要带什么材料", "trace": []})
    assert r["gate"] == "PASS"


def test_greeting_short():
    r = guard_node({"query": "你好", "trace": []})
    assert r["gate"] == "PASS"
    assert r.get("confidence_level") == "greeting"
    assert "您好" in r.get("answer", "")


def test_greeting_not_triggered_for_long_query():
    r = guard_node({"query": "你好我想问下信用卡分期手续费", "trace": []})
    assert r["gate"] == "PASS"
    assert r.get("confidence_level") != "greeting"  # 长query含业务，不当问候


def test_complaint_query_context_not_escalated():
    """『客户有没有投诉记录』是查工单不是投诉本身"""
    r = guard_node({"query": "客户 C2026031800237 有没有投诉记录", "trace": []})
    assert r["gate"] == "PASS"
