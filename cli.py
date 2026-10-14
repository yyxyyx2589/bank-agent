import os

from llm_client import LLMClient
from config import BASE_URL, MODEL, require_api_key
from router import Router
from tools import TOOL_REGISTRY

# 编排引擎开关：react（默认，手写主循环）/ langgraph（图编排）
AGENT_ENGINE = os.environ.get("AGENT_ENGINE", "react").strip().lower()


def _agent_class():
    """按需导入编排引擎，避免没装 langgraph 时连手写版都跑不起来。"""
    if AGENT_ENGINE == "langgraph":
        from agent_langgraph import LangGraphAgent
        return LangGraphAgent
    from agent import Agent
    return Agent

SYSTEM_PROMPT = """你是银行远程银行中心的坐席智能助手，服务对象是客服坐席与网点柜员。

工具使用规范：
1. 涉及利率、费率、限额、办理材料、业务条件的问题，必须先检索知识库（search_knowledge）
   或查实际数据（get_product_rate / check_business_rule），禁止凭记忆作答。
2. 涉及"这位客户之前来问过什么""账户之前是否被处理过"的，先用 query_ticket 查工单记录，
   不要凭猜测描述客户情况。
3. 一次工具结果不够就换关键词多轮查询（同义词、业务别名都试），直到信息足够。
4. 工具返回错误或未找到时如实告知，可调整关键词重试，但最多重试 2 次。
5. 遇到以下任一情形，必须调用 escalate_to_human 转人工，不得自行作答：
   涉及客户资金安全或赔付责任；需要突破坐席权限或双人复核；
   客户明确要求人工服务或已表达强烈不满；知识库与工具均无法给出可靠依据。
6. 回答要给出依据（引用了哪条知识片段、哪条工单或哪条规则），便于坐席复核。
7. 面向客户的答复要口语化，说清下一步该做什么；不要把内部编号堆给客户。
8. 问候、闲聊直接回应，不要调用工具。
"""

ROUTE_BADGE = {
    "rag_direct":  "[知识库直答]",
    "agent_loop":  "[Agent 循环]",
    "refuse_human": "[转人工]",
}


def build_router() -> Router:
    llm = LLMClient(api_key=require_api_key(), base_url=BASE_URL, model=MODEL)
    tools = list(TOOL_REGISTRY.values())
    agent = _agent_class()(llm=llm, tools=tools,
                           system_prompt=SYSTEM_PROMPT, max_turns=8)
    rag_tool = TOOL_REGISTRY["search_knowledge"]
    return Router(rag_tool=rag_tool, agent=agent)



def print_route(result):
    """打印路由结果：徽章 + 置信度 + 工具轨迹 + 答案 + 溯源。"""
    badge = ROUTE_BADGE.get(result.route, f"[{result.route}]")
    print(f"\n  {badge}  sim={result.sim:.2f}  kw_coverage={result.kw_coverage:.2f}")

    if result.tool_trace:
        for i, step in enumerate(result.tool_trace, 1):
            print(f"  [trace {i}] {step['name']}({step['params']})")
            print(f"           -> {step['result'][:120]}")
    else:
        print("  [trace] 本次未调用工具")

    if result.kb_hits:
        print("  [溯源] 命中片段：")
        for kb in result.kb_hits:
            print(f"    {kb['id']} (sim={kb['sim']}) {kb['text'][:60]}...")

    print(f"\n助手: {result.answer}")


def main():
    router = build_router()
    print("=" * 50)
    print("Bank Agent · 远程银行坐席知识助手（输入 exit 退出）")
    print(f"编排引擎：{AGENT_ENGINE}（AGENT_ENGINE=langgraph 可切换）")
    print("路由模式：知识库直答 / Agent循环 / 转人工")
    print("=" * 50)
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            result = router.route(user_input)
        except Exception as e:
            print(f"[错误] {e}")
            continue
        print_route(result)


if __name__ == "__main__":
    main()
