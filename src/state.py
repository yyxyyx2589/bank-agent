"""共享 AgentState schema（多 Agent 全局状态契约）。

设计原则：所有 Agent 读写同一个 schema，supervisor 据此做确定性路由。
字段最小化——大块数据放外部存储（chroma/sqlite），state 只放决策信号。
"""
from __future__ import annotations
from typing import TypedDict, Any


class AgentState(TypedDict, total=False):
    # —— 输入 ——
    query: str                       # 坐席/客户原始问题
    messages: list[dict]             # 对话历史（OpenAI 消息格式）

    # —— Guard 输出 ——
    gate: str                        # PASS / REFUSE / ESCALATE
    gate_reason: str                 # 命中的规则说明（可溯源）

    # —— Knowledge Agent 输出 ——
    rag_evidence: dict | None        # {"best_sim", "best_kw_coverage", "results": [...]}
    confidence_level: str            # high / mid / low（由 sim/kw 派生）

    # —— Rules Agent 输出 ——
    rule_evidence: list[dict]        # 费率/规则表查询结果

    # —— Ticket Agent 输出 ——
    ticket: dict | None              # 工单/转接记录

    # —— 编排控制 ——
    next_agent: str                  # supervisor 决策：knowledge/rules/ticket/synthesis/END
    iterations: int                  # 调度轮数（防死循环）
    ticket_intent: dict | None       # Guard 注入给 Ticket 的确定性转接意图

    # —— 输出 ——
    answer: str                      # 最终答案
    citations: list[str]             # 引用列表（kb-xxx / R-xxx / 工单号）
    trace: list[dict]                # 每步记录 {"agent", "action", "detail"}，评测归因用
