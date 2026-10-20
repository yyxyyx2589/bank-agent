"""Rules 业务规则 Agent 节点。

用 LLM 决定调哪个结构化查询工具（get_product_rate / check_business_rule），
执行后把结果存进 rule_evidence。处理「制度文档只给区间、精确值在表里」的硬场景。

工具权限：只给这两个判定型工具，不给 rag_search、不给 ticket。
"""
import json
from src.state import AgentState
from src.tools.base import to_openai_schema
from src.tools import get_tool
from src.llm_client import LLMClient
from src.config import require_api_key, BASE_URL, MODEL

RULES_TOOLS = ["get_product_rate", "check_business_rule", "query_ticket"]
SYSTEM = """你是银行远程坐席助手的业务规则专家。你的职责是查结构化数据表，给出精确的费率/规则判定。

你可以调用以下工具：
- get_product_rate: 查产品利率/费率/期限（存款/贷款/信用卡/理财）
- check_business_rule: 校验业务办理规则（能不能办、要带什么、能不能代办、金额上限）

判定原则：
1. 客户问具体数字（利率、费率、手续费、起存金额）→ 必须调 get_product_rate，不能用制度文档的区间值回答
2. 客户问条件判定（能不能办、要不要预约、能不能代办、要不要先测评）→ 调 check_business_rule，必要时带 amount
3. 调用工具后，把工具返回的关键事实用一句话总结，不要照抄全部字段
4. 只查你需要的，不要无脑调两个工具

记住：你的输出会被下游 Synthesis 节点用来组织给坐席的答案，所以要给精确事实，不要给模糊区间。"""


def rules_node(state: AgentState) -> dict:
    query = state.get("query", "")
    trace = state.get("trace", [])
    iterations = state.get("iterations", 0) + 1

    tools_schema = [to_openai_schema(get_tool(n)) for n in RULES_TOOLS]
    llm = LLMClient(require_api_key(), BASE_URL, MODEL)

    # 给 LLM 看 RAG 证据作为背景（避免重复查已知的）
    rag = state.get("rag_evidence") or {}
    rag_summary = ""
    if rag.get("results"):
        rag_summary = "\n\n知识库已检索到的背景（供参考，不要照抄）：\n" + \
            "\n".join(f"- {r['id']}: {r['text'][:60]}" for r in rag["results"][:2])

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"坐席问题：{query}{rag_summary}\n\n请调用合适的工具查精确数据，然后用一句话给出关键事实。"},
    ]

    resp = llm.chat(messages, tools=tools_schema)
    msg = resp.choices[0].message
    rule_evidence = list(state.get("rule_evidence", []))
    tool_calls_made = []

    if msg.tool_calls:
        for tc in msg.tool_calls:
            tname = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            t = get_tool(tname)
            if t and tname in RULES_TOOLS:
                result = t.call(args)
                rule_evidence.append({"tool": tname, "args": args, "result": json.loads(result)})
                tool_calls_made.append(f"{tname}({args})")
            else:
                tool_calls_made.append(f"{tname}[拒绝：非授权工具]")

    return {
        "rule_evidence": rule_evidence,
        "iterations": iterations,
        "trace": trace + [{"agent": "rules", "action": "tool_calls",
                           "detail": "; ".join(tool_calls_made) or "未调工具"}],
    }
