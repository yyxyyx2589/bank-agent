"""Knowledge 知识检索 Agent 节点。

调用 search_knowledge 工具，拿 RAG 证据包 + 置信度（sim/kw_coverage）。
派生 confidence_level（high/mid/low）供 Supervisor 路由。

工具权限最小化：本节点只能用 search_knowledge。
"""
import json
from src.state import AgentState
from src.tools import get_tool
from src.config import SIM_HIGH, SIM_LOW, KW_COVERAGE_LOW


def knowledge_node(state: AgentState) -> dict:
    query = state.get("query", "")
    trace = state.get("trace", [])
    tool = get_tool("search_knowledge")
    raw = tool.call({"query": query, "top_k": 3})
    evidence = json.loads(raw)

    sim = evidence.get("best_sim", 0.0)
    kwc = evidence.get("best_kw_coverage", 0.0)

    # 置信度分级（v1 阈值沿用）
    if sim >= SIM_HIGH and kwc >= KW_COVERAGE_LOW:
        level = "high"
    elif sim >= SIM_LOW:
        level = "mid"
    else:
        level = "low"

    return {
        "rag_evidence": evidence,
        "confidence_level": level,
        "iterations": state.get("iterations", 0) + 1,
        "trace": trace + [{"agent": "knowledge", "action": "search",
                           "detail": f"sim={sim} kw={kwc} level={level}"}],
    }
