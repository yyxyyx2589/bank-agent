"""工具层单测（不调 LLM，验证工具直接调用结果）。"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools import TOOL_REGISTRY
from src.tools.rag_search import SearchKnowledgeTool
from src.tools.product_rate import GetProductRateTool
from src.tools.business_rule import CheckBusinessRuleTool
from src.tools.ticket_ops import CreateTicketTool, EscalateToHumanTool


def test_all_tools_registered():
    names = set(TOOL_REGISTRY.keys())
    assert names == {"search_knowledge", "get_product_rate",
                     "check_business_rule", "query_ticket",
                     "create_ticket", "escalate_to_human"}


def test_rag_search_returns_confidence():
    r = json.loads(SearchKnowledgeTool().call({"query": "信用卡分期手续费", "top_k": 1}))
    assert "best_sim" in r and "best_kw_coverage" in r
    assert len(r["results"]) >= 1
    assert r["best_sim"] > 0  # 知识库内有此条


def test_rag_search_empty_query():
    r = json.loads(SearchKnowledgeTool().call({"query": ""}))
    assert "error" in r


def test_product_rate_keyword():
    r = json.loads(GetProductRateTool().call({"keyword": "分期"}))
    assert r["count"] >= 1


def test_product_rate_no_filter_rejected():
    r = json.loads(GetProductRateTool().call({}))
    assert "error" in r


def test_business_rule_with_amount_trigger():
    r = json.loads(CheckBusinessRuleTool().call({"scenario": "大额转账", "amount": 600000}))
    assert r["count"] >= 1
    assert any(rule.get("triggered") for rule in r["rules"])


def test_business_rule_empty_scenario():
    r = json.loads(CheckBusinessRuleTool().call({"scenario": ""}))
    assert "error" in r


def test_escalate_generates_transfer_no():
    r = json.loads(EscalateToHumanTool().call({"reason": "盗刷", "priority": "紧急"}))
    assert r["status"] == "已转人工"
    assert r["transfer_no"].startswith("ZR")
    assert r["priority"] == "紧急"


def test_escalate_empty_reason():
    r = json.loads(EscalateToHumanTool().call({"reason": ""}))
    assert "error" in r


def test_create_ticket():
    r = json.loads(CreateTicketTool().call({
        "customer_id": "C202601010000001", "category": "挂失补卡", "summary": "卡丢了"}))
    assert r["ticket_no"].startswith("T")
    assert r["status"] == "已创建"


def test_query_ticket_by_customer():
    from src.tools.query_ticket import QueryTicketTool
    r = json.loads(QueryTicketTool().call({"customer_id": "C2026031800237"}))
    assert "count" in r or "result" in r


def test_query_ticket_no_filter_rejected():
    from src.tools.query_ticket import QueryTicketTool
    r = json.loads(QueryTicketTool().call({}))
    assert "error" in r
