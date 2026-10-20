"""Synthesis 答案合成节点。

把 RAG 证据 + 规则证据组装成给坐席的可溯源答案。引用 kb-xxx / R-xxx / 工单号。
拒答场景（Guard REFUSE）的答案已在 guard_node 给出，本节点不动。
"""
import json
from src.state import AgentState
from src.llm_client import LLMClient
from src.config import require_api_key, BASE_URL, MODEL

SYSTEM = """你是银行远程坐席助手的答案合成器。把检索到的证据组织成给坐席看的答案。

硬要求：
1. 答案必须基于给定的证据，不得编造任何不在证据里的数字、规则或流程
2. 涉及具体数字（利率/费率/限额）必须用规则表查到的精确值，不能用知识库的区间值
3. 答案末尾必须列出引用来源（kb-xxx 知识片段 / R-xxx 规则 / 工单号）
4. 如果证据不足以回答，明确说"知识库暂无可靠依据，建议转人工"，不要硬答
5. 口吻是给坐席看的工作提示，不是直接对客户说的话

输出格式：
[答案正文]
---
引用：kb-xxx, R-xxx"""


def synthesis_node(state: AgentState) -> dict:
    query = state.get("query", "")
    trace = state.get("trace", [])

    # Guard 拒答或问候：直接返回已写答案
    if state.get("gate") == "REFUSE" or state.get("confidence_level") == "greeting":
        return {"trace": trace + [{"agent": "synthesis", "action": "skip(preset)", "detail": ""}]}

    rag = state.get("rag_evidence") or {}
    rules_ev = state.get("rule_evidence") or []

    # 组装证据上下文
    parts = []
    citations = []
    if rag.get("results"):
        parts.append("【知识库证据】")
        for r in rag["results"][:3]:
            parts.append(f"- {r['id']} (sim={r['sim']}): {r['text']}")
            citations.append(r["id"])
    if rules_ev:
        parts.append("\n【规则/费率证据】")
        for ev in rules_ev:
            res = ev.get("result", {})
            if "products" in res:
                for p in res["products"][:3]:
                    parts.append(f"- {p.get('product_code','?')} {p.get('name','')}: "
                                 f"{p.get('rate_desc', p.get('rate',''))} (起存 {p.get('min_amount','')})")
                    citations.append(p.get("product_code", "产品"))
            elif "rules" in res:
                for r in res["rules"][:3]:
                    parts.append(f"- {r['rule_code']} {r['scenario']}: {r['requirement']} "
                                 f"(代办:{r['allow_agent_desc']}; 触发:{r.get('triggered')})")
                    citations.append(r["rule_code"])

    evidence_ctx = "\n".join(parts) if parts else "（无可用证据）"
    confidence = state.get("confidence_level", "unknown")

    llm = LLMClient(require_api_key(), BASE_URL, MODEL)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"坐席问题：{query}\n置信度：{confidence}\n\n证据：\n{evidence_ctx}\n\n请基于证据组织答案。"},
    ]
    resp = llm.chat(messages)
    answer = resp.choices[0].message.content.strip()

    # 若 LLM 没带引用行，补一个
    if "引用" not in answer and citations:
        answer += "\n---\n引用：" + ", ".join(dict.fromkeys(citations))

    return {
        "answer": answer,
        "citations": list(dict.fromkeys(citations)),
        "trace": trace + [{"agent": "synthesis", "action": "compose",
                           "detail": f"conf={confidence} cites={len(citations)}"}],
    }
