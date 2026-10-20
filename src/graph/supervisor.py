"""Supervisor 编排节点 + 条件路由。

显式规则路由（确定性优先，LLM 不参与路由决策）：
  gate==REFUSE            → synthesis（guard 已写答案）
  gate==ESCALATE          → ticket（guard 已给 intent）
  gate==PASS:
    还没检索过（rag_evidence 为空）         → knowledge（先拿置信度）
    高置信 + 无具体数字需求                  → synthesis（直答）
    中置信 或 query 含数字/条件判定信号      → rules（补结构化查询）
    有 rule_evidence 且足够                  → synthesis
    低置信                                  → ticket（转人工兜底）
    迭代超 MAX_ITERATIONS                   → synthesis（强制收敛）

为什么不用 LLM 路由：开源项目普遍让 supervisor 用 LLM 决定下一个 agent，
实测路由不一致且不可解释。本项目用「置信度 + 关键词信号」做确定性规则，
LLM 只在 synthesis/rules 的工具选择里用——路由层完全可归因。
"""
import re
from src.state import AgentState
from src.config import MAX_ITERATIONS, SIM_LOW

# 触发"必须查结构化表"的信号词：具体数字 / 条件判定 / 工单查询
NUMBER_PATTERN = re.compile(r"\d+[%万千百元]|分\d+期|分\s*\d+\s*期|利率|费率|手续费|起存|限额|上限|能不能|要不要|是否|能不能代办")
CONDITION_SIGNALS = ["能不能", "要不要", "是否", "能不能代办", "可不可以", "需要什么", "要带什么", "提前预约"]
TICKET_QUERY_SIGNALS = ["工单", "记录", "之前", "历史", "处理状态", "来问过", "有没有投诉记录", "诉求"]


def _needs_rules(state: AgentState) -> bool:
    """query 含具体数字、条件判定或工单查询信号 → 必须走 rules 查表。"""
    q = state.get("query", "")
    if NUMBER_PATTERN.search(q):
        return True
    if any(s in q for s in CONDITION_SIGNALS):
        return True
    if any(s in q for s in TICKET_QUERY_SIGNALS):
        return True
    return False


def supervisor_node(state: AgentState) -> dict:
    """返回 next_agent 字段，供条件边消费。不修改业务数据。"""
    gate = state.get("gate", "PASS")

    if gate == "REFUSE":
        return {"next_agent": "synthesis"}
    if gate == "ESCALATE":
        return {"next_agent": "ticket"}

    # PASS 分支
    iters = state.get("iterations", 0)
    if iters >= MAX_ITERATIONS:
        return {"next_agent": "synthesis"}  # 强制收敛

    # 问候类：guard 已给答案，直接合成（保留预设答案）
    if state.get("confidence_level") == "greeting":
        return {"next_agent": "synthesis"}

    rag = state.get("rag_evidence")
    if not rag:
        return {"next_agent": "knowledge"}  # 先检索

    # 已有 RAG 证据，看置信度
    level = state.get("confidence_level", "low")
    has_rules = bool(state.get("rule_evidence"))

    if level == "low":
        # 低置信：rules 跑过仍无有效证据 → ticket 兜底；没跑过 → 先 rules
        rules_run = state.get("iterations", 0) >= 2  # knowledge(1)+rules(2) 后 iterations>=2
        has_rules = bool(state.get("rule_evidence"))
        if rules_run and not has_rules:
            return {"next_agent": "ticket"}  # rules 跑了没查到东西 → 转人工
        return {"next_agent": "rules" if not has_rules else "ticket"}

    if level == "high" and not _needs_rules(state):
        return {"next_agent": "synthesis"}  # 高置信且无数字需求 → 直答

    if level == "mid" or _needs_rules(state):
        # 中置信 或 含数字/条件信号 → 需要 rules 补充；已查过则合成
        return {"next_agent": "synthesis" if has_rules else "rules"}

    return {"next_agent": "synthesis"}


# ============================================================
# 条件边函数：返回下一个节点名
# ============================================================
def route_after_guard(state: AgentState) -> str:
    g = state.get("gate", "PASS")
    if g == "REFUSE":
        return "synthesis"
    if g == "ESCALATE":
        return "ticket"
    return "supervisor"


def route_after_supervisor(state: AgentState) -> str:
    nxt = state.get("next_agent", "synthesis")
    # 终止保护
    if nxt in ("synthesis", "ticket"):
        return nxt
    if nxt == "knowledge":
        return "knowledge"
    if nxt == "rules":
        return "rules"
    return "synthesis"
