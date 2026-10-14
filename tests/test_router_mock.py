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
    """记录调用次数、每次请求的 tool_choice 与 messages，返回固定文本（走'不调工具'分支）。"""

    def __init__(self):
        self.calls = 0
        self.tool_choices = []
        self.requests = []

    def chat(self, messages, tools=None, stream=False, tool_choice=None):
        self.calls += 1
        self.tool_choices.append(tool_choice)
        self.requests.append(messages)
        return make_response(content="[stub answer]")

    @property
    def last_user_message(self) -> str:
        """最后一次请求里 Agent 实际看到的 user 内容（含路由注入的先验与指令）。"""
        return self.requests[-1][-1]["content"]


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


def test_override_predicates():
    """两个「高置信但不该直答」的判定函数 + 行外护栏。

    直接测判定函数本身而不依赖检索分数：检索分数会随知识库演进而漂移，
    判定规则才是设计意图所在，必须锁死。
    """
    router = build_router(StubLLM())

    # 数字类：知识库只给区间，精确值在产品参数表
    assert check("数字类：问具体费率 -> 需查参数表",
                 router._needs_live_data("信用卡分 12 期的手续费率是多少"))
    assert check("数字类：问办理流程 -> 不应触发（知识库答得了）",
                 not router._needs_live_data("银行卡丢了怎么补办"))

    # 判定类：制度口径模糊，逐条判定在业务规则表
    assert check("判定类：带具体金额 -> 需查规则表",
                 router._needs_rule_check("客户要转 60 万，需要提前预约吗"))
    assert check("判定类：问能否代办 -> 需查规则表",
                 router._needs_rule_check("银行卡挂失可以让人代办吗"))
    assert check("判定类：泛问规定（无金额无条件词）-> 不应触发",
                 not router._needs_rule_check("大额转账要预约吗"))

    # 行外护栏：合规红线
    assert check("行外护栏：荐股 -> 命中", router._is_out_of_scope("帮我推荐一只股票"))
    assert check("行外护栏：正常业务 -> 不命中",
                 not router._is_out_of_scope("基金定投怎么赎回"))


def test_escalation_two_tiers():
    """升级人工分两档，但两档都在路由层强制首轮转接。

    为什么必须分档（收口审查发现的缺陷 + 修复）：
      原实现把升级类问题留在「低置信兜底」里，且只给一句提示词。
      问题一：投诉的处置方式不该随相似度变化——sim 一旦不低就整条绕过；
      问题二：_ESCALATE_NOTE 只是提示词，实测资金安全场景约 50% 服从率
             （同一句话两次运行，一次转接、一次先追问卡号）。
      修复：升级判定上提到阈值决策**之前**（处置与 sim 解耦），
           两档都强制首轮调用 escalate_to_human，差别收进指令内容
           （资金安全：紧急/反欺诈组/禁止先追问身份）。
    注：这里直接测判定函数 + 路由结果 + tool_choice 契约，不依赖检索分数漂移。
    """
    router = build_router(StubLLM())

    # 资金安全判定：只认"资金出事的表达"，不认泛泛的合规问询
    assert check("资金安全：盗刷 -> 命中", router._is_fund_safety("我的卡被盗刷了"))
    assert check("资金安全：非本人交易 -> 命中",
                 router._is_fund_safety("这笔钱不是我取的"))
    assert check("资金安全：普通余额查询 -> 不命中",
                 not router._is_fund_safety("帮我查一下卡里的余额"))

    assert check("升级兜底：资金安全也算升级类",
                 router._needs_escalation("我的卡被盗刷了"))
    assert check("升级兜底：投诉算升级类", router._needs_escalation("我要投诉你们"))
    assert check("升级兜底：正常业务咨询不算",
                 not router._needs_escalation("银行卡丢了怎么补办"))

    # 语境排除：查档案 ≠ 客户投诉（否则会给合规客户凭空开转接单）
    assert check("语境排除：查某客户的投诉记录 -> 不算升级",
                 not router._needs_escalation("客户 C2026031800237 有没有投诉记录"))
    assert check("语境排除：客户之前投诉过什么 -> 不算升级",
                 not router._needs_escalation("这位客户之前投诉过什么"))
    assert check("语境例外：带第一人称的查记录仍算升级",
                 router._needs_escalation("我要查一下我的投诉记录"))

    forced = {"type": "function", "function": {"name": "escalate_to_human"}}

    # ① 投诉类：路由到 agent_loop + 首轮强制转接
    llm1 = StubLLM()
    r1 = build_router(llm1).route("我要投诉你们的服务态度")
    assert check("投诉 -> agent_loop（不再被拒答打发）", r1.route == "agent_loop")
    assert check("投诉 -> 首轮强制 escalate_to_human", llm1.tool_choices[0] == forced)

    # ② 资金安全类：同样强制，但注入的是"紧急/禁止先追问身份"那一档指令
    llm2 = StubLLM()
    r2 = build_router(llm2).route("我的卡被盗刷了，钱被转走了")
    assert check("资金安全 -> agent_loop", r2.route == "agent_loop")
    assert check("资金安全 -> 首轮强制 escalate_to_human",
                 llm2.tool_choices[0] == forced)
    assert check("资金安全 -> 处置与相似度解耦（不再落到 refuse_human）",
                 r2.route != "refuse_human")

    # ③ 两档注入给 Agent 的指令确实不同（时效性差别落在提示词里，不只是换个变量名）
    fund_llm, comp_llm = StubLLM(), StubLLM()
    build_router(fund_llm).route("我的卡被盗刷了，钱被转走了")
    build_router(comp_llm).route("我要投诉你们的服务态度")
    fund_note, comp_note = fund_llm.last_user_message, comp_llm.last_user_message

    assert check("资金安全档：要求转反欺诈组 + 紧急优先级",
                 "反欺诈组" in fund_note and "紧急" in fund_note)
    assert check("资金安全档：明确禁止先追问账户/身份",
                 "不要先向客户追问" in fund_note)
    assert check("投诉档：要求转客户投诉处理组 + 先安抚",
                 "客户投诉处理组" in comp_note and "安抚" in comp_note)
    assert check("投诉档不含资金安全的时效指令（两档确实不同）",
                 "反欺诈组" not in comp_note and "不要先向客户追问" not in comp_note)
    assert check("两档都提示不得用'超出知识库范围'打发客户",
                 "超出知识库范围" in fund_note and "超出知识库范围" in comp_note)


def test_escalation_not_hijacked_by_threshold():
    """回归：升级判定必须在阈值决策之前，否则高置信投诉会被 rag_direct 直接答掉。"""
    llm = StubLLM()
    # 把阈值压到极低，让"投诉"这种低 sim 的问题也能进高置信区间
    router = build_router(llm, sim_high=0.01, sim_low=0.005, kw_low=0.0)
    r = router.route("我要投诉你们乱收费")
    assert check("高置信区间内的投诉仍走 agent_loop（未被 rag_direct 抢走）",
                 r.route == "agent_loop")


def test_high_conf_but_should_not_answer_directly():
    """端到端验证：高置信也照样改走 agent_loop。

    这是本项目与「普通 RAG」最本质的区别——置信度高不等于该由知识库直接答。
    """
    # ① 问具体数字：sim 高（命中 kb-008），但精确值在产品表
    llm = StubLLM()
    router = build_router(llm)
    r1 = router.route("信用卡分 12 期的手续费率是多少")
    assert check("高置信 + 问具体数字 -> agent_loop", r1.route == "agent_loop")
    assert check("  且确实是高置信（否则测不到该分支）", r1.sim >= SIM_HIGH)

    # ② 带具体条件：sim 高（命中 kb-006），但逐条判定在规则表
    llm2 = StubLLM()
    r2 = build_router(llm2).route("客户要转 60 万，需要提前预约吗")
    assert check("高置信 + 带具体金额 -> agent_loop", r2.route == "agent_loop")


if __name__ == "__main__":
    for fn in [test_rag_direct, test_refuse_human, test_out_of_scope_guard,
               test_agent_loop, test_override_predicates,
               test_high_conf_but_should_not_answer_directly,
               test_escalation_two_tiers,
               test_escalation_not_hijacked_by_threshold]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n全部通过：决策路由三条分支 + 行外护栏 + 两类直答改写 + 升级人工两档处置正确。")
