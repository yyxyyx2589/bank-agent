"""Ticket 工单处置 Agent 节点。

系统唯一的副作用写操作节点。接收 Guard 的 ticket_intent（资金安全/投诉/要求人工）
或 Supervisor 的低置信兜底，调 escalate_to_human 或 create_ticket。

工具权限：只有这两个写工具。Guard 已给出 intent 时直接执行，不调 LLM——
确定性路径不交给模型决定。
"""
import json
from src.state import AgentState
from src.tools import get_tool
from src.llm_client import LLMClient
from src.config import require_api_key, BASE_URL, MODEL

SYSTEM = """你是银行坐席助手的工单处置专家。根据情况调用合适工具：
- escalate_to_human: 转人工（资金安全/投诉/要求人工/超出能力）
- create_ticket: 建业务工单留痕（需要后续跟进的业务诉求）

调用后用一句话说明已做的处置。不要自行回答业务问题——你的职责是转交和留痕。"""


def ticket_node(state: AgentState) -> dict:
    query = state.get("query", "")
    trace = state.get("trace", [])
    intent = state.get("ticket_intent")  # Guard 注入的确定性 intent

    if intent:
        # 确定性路径：Guard 已决定，直接执行，不调 LLM
        t = get_tool("escalate_to_human")
        result = json.loads(t.call(intent))
        return {
            "ticket": result,
            "answer": result.get("message", "已转人工"),
            "citations": [result.get("transfer_no", "转接单")],
            "trace": trace + [{"agent": "ticket", "action": "escalate(deterministic)",
                               "detail": intent.get("reason", "")}],
        }

    # 低置信兜底：调 LLM 判断走 escalate 还是 create
    llm = LLMClient(require_api_key(), BASE_URL, MODEL)
    from src.tools.base import to_openai_schema
    tools_schema = [to_openai_schema(get_tool(n)) for n in ["escalate_to_human", "create_ticket"]]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"坐席问题（知识库无法可靠回答）：{query}\n请转人工或建工单。"},
    ]
    resp = llm.chat(messages, tools=tools_schema)
    msg = resp.choices[0].message
    ticket = None
    answer = "已转人工处理，请保持通话。"
    citations = []

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        tname = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except Exception:
            args = {}
        t = get_tool(tname)
        if t:
            result = json.loads(t.call(args))
            ticket = result
            answer = result.get("message", answer)
            no = result.get("transfer_no") or result.get("ticket_no")
            if no:
                citations.append(no)
        detail = f"{tname}({args})"
    else:
        answer = msg.content or answer
        detail = "未调工具"

    return {
        "gate": "ESCALATE",  # 兜底转人工也标记 ESCALATE，便于评测归因
        "ticket": ticket,
        "answer": answer,
        "citations": citations,
        "trace": trace + [{"agent": "ticket", "action": "llm_dispatch", "detail": detail}],
    }
