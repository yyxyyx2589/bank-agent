"""决策路由测试：用假 LLM 验证三条路由分支（零 API 消耗）。

sim 数值分布（本知识库 + 本地 embedding 实测）：
  银行卡丢了怎么补办            -> sim≈0.62 -> rag_direct
  企业开基本户要什么材料        -> sim≈0.44 -> agent_loop（中区间）
  你们网点附近哪里有好吃的火锅  -> sim≈0.11 -> refuse_human
  帮我推荐一只股票              -> 显式行外护栏 -> refuse_human（合规红线）

运行：python tests/test_router_mock.py
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent  # noqa: E402
from router import Router  # noqa: E402
from tools import TOOL_REGISTRY  # noqa: E402
from config import SIM_HIGH, SIM_LOW, KW_COVERAGE_LOW  # noqa: E402

SYSTEM_PROMPT = "你是银行远程银行中心的坐席智能助手。"


def make_response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class StubLLM:
    """记录调用次数，返回固定文本（Agent 循环里只走'不调工具'分支）。"""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, stream=False):
        self.calls += 1
        return make_response(content="[stub answer]")


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    return cond


def build_router(llm, sim_high=SIM_HIGH, sim_low=SIM_LOW, kw_low=KW_COVERAGE_LOW):
    tools = list(TOOL_REGISTRY.values())
    agent = Agent(llm=llm, tools=tools,
                 system_prompt=SYSTEM_PROMPT, max_turns=8)
    rag_tool = TOOL_REGISTRY["search_knowledge"]
    return Router(rag_tool=rag_tool, agent=agent,
                 sim_high=sim_high, sim_low=sim_low,
                 kw_coverage_low=kw_low)


def test_rag_direct():
    """高置信：标准业务问题命中知识库，走 RAG 直答。"""
    llm = StubLLM()
    router = build_router(llm)
    result = router.route("银行卡丢了怎么补办")

    assert check("rag_direct 路由命中", result.route == "rag_direct")
    assert check(f"rag_direct sim 超过 {SIM_HIGH}", result.sim >= SIM_HIGH)
    assert check("rag_direct 未进 Agent 循环", result.tool_trace == [])
    assert check("rag_direct 有溯源片段", len(result.kb_hits) >= 1)
    assert check("rag_direct LLM 只调一次（组织语言）", llm.calls == 1)


def test_refuse_human():
    """超纲问题：低置信，拒答转人工，不调 LLM。"""
    llm = StubLLM()
    router = build_router(llm)
    result = router.route("你们网点附近哪里有好吃的火锅")

    assert check("refuse_human 路由命中", result.route == "refuse_human")
    assert check(f"refuse_human sim 低于 {SIM_LOW}", result.sim < SIM_LOW)
    assert check("refuse_human 不调 LLM", llm.calls == 0)
    assert check("refuse_human 答案含转人工提示", "转接人工" in result.answer)


def test_out_of_scope_guard():
    """行外护栏：荐股属合规红线，即使相似度不低也必须转人工，不能进 Agent 回答。"""
    llm = StubLLM()
    router = build_router(llm)
    result = router.route("帮我推荐一只股票")

    assert check("行外问题被拦为 refuse_human", result.route == "refuse_human")
    assert check("行外问题不调 LLM", llm.calls == 0)


def test_agent_loop():
    """中置信：sim 在 [SIM_LOW, SIM_HIGH) 之间，走 Agent 循环查工具。"""
    llm = StubLLM()
    router = build_router(llm)
    result = router.route("企业开基本户要什么材料")

    assert check("agent_loop 路由命中", result.route == "agent_loop")
    assert check(f"agent_loop sim 在中区间 [{SIM_LOW}, {SIM_HIGH})",
                 SIM_LOW <= result.sim < SIM_HIGH)
    assert check("agent_loop 调用了 LLM（进 Agent 循环）", llm.calls >= 1)
    print(f"    (sim={result.sim:.3f}, kw={result.kw_coverage:.3f})")


if __name__ == "__main__":
    for fn in [test_rag_direct, test_refuse_human, test_out_of_scope_guard,
               test_agent_loop]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n全部通过：决策路由三条分支 + 行外护栏逻辑正确。")
