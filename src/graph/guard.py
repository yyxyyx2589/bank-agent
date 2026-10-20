"""Guard 合规守卫节点。

必经的第一道闸门（不在 Supervisor 调度环内）。纯规则引擎，不依赖 LLM——
合规判断不能是模型的心情。输出 gate: PASS / REFUSE / ESCALATE。

设计依据：v1 实测资金安全场景提示词服从率仅约 50%，靠路由层强制工具调用
兜底才过关。v2 把"该不该答"前置成独立节点，规则命中即确定。
"""
from src.state import AgentState
from src.config import (GUARD_REFUSE_PATTERNS, GUARD_ESCALATE_FUND_SAFETY,
                        GUARD_ESCALATE_COMPLAINT, GUARD_ESCALATE_REQUEST_HUMAN)


def _hit(query: str, patterns: list) -> str | None:
    q = query.lower()
    for p in patterns:
        if p.lower() in q:
            return p
    return None


def _hit_combination(query: str, anchors: list, targets: list) -> str | None:
    """组合匹配：anchor + target 同时出现才算命中（如『推荐』+『基金/股票』）。
    解决『你们能推荐哪只基金吗』这类中间隔词的变体。"""
    q = query.lower()
    has_anchor = any(a.lower() in q for a in anchors)
    if not has_anchor:
        return None
    for t in targets:
        if t.lower() in q:
            return f"{anchors[0]}+{t}"
    return None


COMPLAINT_QUERY_CONTEXTS = ["投诉记录", "投诉状态", "有没有投诉", "投诉工单", "投诉历史"]


def _is_complaint_query_context(query: str) -> bool:
    """query 是在『查投诉记录』而非『发起投诉』——如『客户有没有投诉记录』。
    命中查询语境词则不当投诉处理，走正常业务流（查工单）。"""
    q = query.lower()
    return any(ctx in q for ctx in COMPLAINT_QUERY_CONTEXTS)


GREETING_PATTERNS = ["你好", "您好", "在吗", "在不在", "谢谢", "辛苦了", "hi", "hello", "嗨"]


def _is_greeting(query: str) -> bool:
    """纯问候（无业务意图）→ 直接回应。query 很短且只含问候词才算。"""
    q = query.strip().lower()
    if len(q) > 8:  # 长query大概率含业务内容
        return False
    return any(q == g or q == g + "吗" or q == g + "啊" for g in GREETING_PATTERNS) or \
           (len(q) <= 4 and any(g in q for g in GREETING_PATTERNS))


def guard_node(state: AgentState) -> dict:
    query = state.get("query", "")
    trace = state.get("trace", [])

    # 1. 行外护栏（最高优先级，合规红线）
    hit = _hit(query, GUARD_REFUSE_PATTERNS)
    if not hit:
        # 组合匹配：『推荐/建议』+『基金/股票/理财』= 荐股行为
        hit = _hit_combination(query, ["推荐", "建议买", "推荐买"], ["基金", "股票", "哪只", "哪个"])
    if hit:
        return {
            "gate": "REFUSE",
            "gate_reason": f"命中行外护栏关键词『{hit}』——银行不得荐股/承诺收益/代客决策",
            "answer": ("抱歉，根据监管规定，银行不得向客户推荐具体股票或基金、"
                       "不得承诺理财收益、不得代客作出投资决策。"
                       "如您需要投资建议，建议咨询持牌的证券或基金投顾机构。"),
            "citations": ["合规红线-行外业务"],
            "trace": trace + [{"agent": "guard", "action": "REFUSE", "detail": hit}],
        }

    # 2. 资金安全（盗刷/非本人交易）→ 紧急转接
    hit = _hit(query, GUARD_ESCALATE_FUND_SAFETY)
    if hit:
        return {
            "gate": "ESCALATE",
            "gate_reason": f"命中资金安全关键词『{hit}』——涉盗刷/诈骗必须转反欺诈组",
            "next_agent": "ticket",
            "ticket_intent": {"reason": f"资金安全：{hit}", "priority": "紧急",
                              "suggested_team": "反欺诈组"},
            "trace": trace + [{"agent": "guard", "action": "ESCALATE_FUND", "detail": hit}],
        }

    # 3. 投诉 → 转投诉专席（排除『投诉记录/投诉状态/有没有投诉』等查询语境）
    hit = _hit(query, GUARD_ESCALATE_COMPLAINT)
    if hit and not _is_complaint_query_context(query):
        return {
            "gate": "ESCALATE",
            "gate_reason": f"命中投诉关键词『{hit}』——投诉需转投诉专席，AI 不辩解",
            "next_agent": "ticket",
            "ticket_intent": {"reason": f"客户投诉：{hit}", "priority": "优先",
                              "suggested_team": "投诉处理组"},
            "trace": trace + [{"agent": "guard", "action": "ESCALATE_COMPLAINT", "detail": hit}],
        }

    # 4. 要求人工 → 转人工
    hit = _hit(query, GUARD_ESCALATE_REQUEST_HUMAN)
    if hit:
        return {
            "gate": "ESCALATE",
            "gate_reason": f"客户要求人工服务『{hit}』——尊重客户选择",
            "next_agent": "ticket",
            "ticket_intent": {"reason": "客户要求人工服务", "priority": "常规",
                              "suggested_team": "综合业务组"},
            "trace": trace + [{"agent": "guard", "action": "ESCALATE_HUMAN", "detail": hit}],
        }

    # 5. 问候类（你好/在吗/谢谢）→ 给问候语，标记 PASS 但带预设答案
    if _is_greeting(query):
        return {
            "gate": "PASS",
            "gate_reason": "问候类，预设答案",
            "answer": "您好，我是银行远程坐席知识助手。请问有什么业务问题需要帮您查询？"
                      "（如开户材料、转账限额、挂失补卡、信用卡分期、贷款利率等）",
            "citations": ["系统问候"],
            "confidence_level": "greeting",
            "trace": trace + [{"agent": "guard", "action": "GREETING", "detail": query[:10]}],
        }

    # 6. 放行
    return {
        "gate": "PASS",
        "gate_reason": "未命中任何护栏关键词，放行进入业务处理",
        "trace": trace + [{"agent": "guard", "action": "PASS", "detail": ""}],
    }
