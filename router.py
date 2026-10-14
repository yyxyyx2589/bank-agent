"""决策路由模块：基于知识库检索置信度，把问题分流到三条路径。

银行坐席场景下，三层路由不只是性能优化，更是**合规风控机制**：
银行 AI 的硬约束是"不能幻觉、不能越权、必须可问责"，而三级路由天然
对应了"风险分级处置"。

三层路由：
  rag_direct  : sim >= SIM_HIGH 且 kw_coverage >= KW_COVERAGE_LOW
                命中知识库高置信片段（标准业务问题），模型只需组织语言，
                不进 Agent 循环 —— 快且省 token，且答案有原文可溯源
  agent_loop  : SIM_LOW <= sim < SIM_HIGH
                中置信区间，信息不足，启动 Agent 循环调工具补充
                （查工单 / 查费率 / 校验业务规则）
  refuse_human: sim < SIM_LOW
                超出知识库范围，**必须转人工**，禁止模型自由发挥

设计要点：
  - Router 作为编排层，Agent 作为执行层之一（依赖反转）
  - Router 直接调 RAG 工具拿 sim 分数（不经 LLM），决策后把 RAG 结果
    作为 observation 塞进 Agent 的初始上下文，避免重复检索
  - event_hook 透传给 Agent，SSE 流式时实时推送 route/tool_call/tool_result
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config import SIM_HIGH, SIM_LOW, KW_COVERAGE_LOW

if TYPE_CHECKING:
    from agent import Agent
    from tools.base import BaseTool

# 意图检测关键词（设计动机：向量相似度对"口语化动作型提问"打分偏低。
# 知识库是陈述性的制度条款，没有"我卡丢了""能不能代办"这类口语动作描述，
# 于是低置信的口语提问会被误判到 refuse_human。用「动作词 + 业务主题词」
# 双命中兜底，把本要拒答的业务咨询救回 agent_loop。）
_ACTION_VERBS = {"查", "查询", "查一下", "查找", "看看", "看一下", "调出", "拉取",
                 "能不能", "可不可以", "可以吗", "行不行", "能否", "要不要", "有没有",
                 "怎么办", "怎么处理", "怎么算", "如何办理", "如何解决", "需要什么",
                 "要什么", "要带什么", "是什么", "为什么", "多少", "哪些", "几天",
                 "多久", "多少时间", "问过", "来过", "之前", "帮我查"}
_TOOL_THEMES = {"开户", "开卡", "办卡", "销户", "挂失", "补卡", "补办", "密码",
                "转账", "限额", "额度", "手续费", "利率", "费率", "利息",
                "存款", "定期", "活期", "存单", "大额", "贷款", "房贷", "消费贷",
                "还款", "信用卡", "分期", "逾期", "征信", "理财", "基金",
                "风险", "测评", "保险", "对公", "企业", "基本户", "网银",
                "代发工资", "工资", "账户", "冻结", "解冻", "风控", "诈骗", "反诈",
                "材料", "身份证", "代办", "未成年人", "征信报告",
                # 工单/客户维度：这类问题的答案不在知识库里，只能查工单记录，
                # 必须能命中主题词才能被 _detect_intent 救回 agent_loop。
                # （实测发现：漏了这几个词会导致"客户 C2026xxx 之前来问过什么"
                # 这类纯工单查询被判为转人工。）
                "客户", "工单", "来电", "投诉", "状态", "记录"}

# 寒暄/闲聊词（见 _is_smalltalk）
_GREETING_WORDS = {"你好", "您好", "在吗", "在么", "hello", "hi",
                   "谢谢", "感谢", "再见", "拜拜", "早上好", "下午好",
                   "晚上好", "辛苦了"}

# 数据型问题的信号词（见 _needs_live_data）
_DATA_HINT_WORDS = {"多少", "是多少", "具体", "最新", "现在多少", "目前多少"}
_DATA_THEMES = {"利率", "费率", "手续费", "限额", "额度", "起存", "年化", "汇率"}

# 条件判定型问题的信号词（见 _needs_rule_check）
# 与 _needs_live_data 的分工：
#   数字类问「值是多少」→ 查产品参数表（_DATA_*）
#   判定类问「能不能办 / 要带什么」→ 查业务规则表（_RULE_*），
#   且答案会随客户的具体条件（金额、是否本人）变化，不能只靠制度原文。
_RULE_AMOUNT_RE = re.compile(r"\d+\s*(万|元|块|美元|美金)")
_RULE_COND_WORDS = {"代办", "代他人", "他人代办", "别人帮", "委托他人",
                    "能否", "能不能", "可不可以", "可以吗", "行不行",
                    "需不需要", "要不要", "是否需要", "是否必须", "必须本人"}
_RULE_SCENES = {"转账", "汇款", "挂失", "补卡", "补办", "开户", "开卡", "销户",
                "大额", "预约", "代办", "提前支取", "提前取", "限额", "解控",
                "大额转账", "对公开户", "基本户"}

# 行外/禁答话题（见 _is_out_of_scope）
# 合规红线：银行不得向客户提供证券投资建议（需投顾牌照）、不得涉及博彩。
# 这类问题向量相似度可能不低（"股票"与知识库里的"基金/理财"语义接近），
# 仅靠阈值拦不住，必须用显式护栏。
_OUT_OF_SCOPE_RE = re.compile(
    r"(推荐|荐股|选股|买|买入|炒股|预测|看好|能涨|会涨|值得买).{0,5}(股票|个股)"
    r"|荐股|牛股|黑马股|股票推荐|彩票|赌博|赌球|六合彩|足彩|买彩票")


@dataclass
class RouteResult:
    route: str            # "rag_direct" / "agent_loop" / "refuse_human"
    answer: str           # 最终答案
    sim: float            # RAG top1 相似度（路由依据，前端徽章展示）
    kw_coverage: float    # 关键词覆盖率（路由依据）
    tool_trace: list = field(default_factory=list)  # Agent 路径的工具调用轨迹
    kb_hits: list = field(default_factory=list)      # 命中的知识片段（溯源面板）


class Router:
    def __init__(self, rag_tool: BaseTool, agent: Agent,
                 sim_high: float = SIM_HIGH,
                 sim_low: float = SIM_LOW,
                 kw_coverage_low: float = KW_COVERAGE_LOW,
                 event_hook=None):
        self.rag_tool = rag_tool
        self.agent = agent
        self.sim_high = sim_high
        self.sim_low = sim_low
        self.kw_coverage_low = kw_coverage_low
        self.event_hook = event_hook

    def _emit(self, etype: str, data: dict):
        if self.event_hook:
            self.event_hook(etype, data)

    def _detect_intent(self, query: str) -> bool:
        """意图检测：query 是否同时包含动作词 + 工具主题词。

        设计动机：向量相似度对"我卡丢了怎么办"这类口语化提问打分偏低
        （知识库是陈述性条款，没有口语动作描述），低置信会被误判到 refuse_human。
        用"动作词 + 业务主题词"双命中兜底，把本要拒答的业务咨询救回 agent_loop。

        反例防护：
          - "银行卡挂失流程"：只有主题词无动作词 → 不算意图，走 sim 决策
          - "帮我查一下天气"：只有动作词无主题词 → 不算意图，走 sim 决策
          - "信用卡分期手续费怎么算"：sim 高走 rag_direct，
            根本到不了意图检测层（意图只在 sim<sim_low 时才触发）
        """
        q = query.lower()
        has_action = any(v in q for v in _ACTION_VERBS)
        has_theme = any(t in q for t in _TOOL_THEMES)
        return has_action and has_theme

    def _is_smalltalk(self, query: str) -> bool:
        """寒暄短路：识别问候/致谢类短句，交给 LLM 自然回应。

        为什么需要它（换皮到银行场景后发现的一个真实边界缺陷）：
        知识库是业务条款，对"你好"当然检索不到任何片段，sim 会低到 0.06，
        于是客户刚接通开口说一句"你好"就被判为 refuse_human、直接转人工 ——
        这在坐席场景属于体验事故。

        原文的意图检测（动作词+主题词）也救不了："你好"两个词都不含。
        因此单独加一层寒暄识别，命中就走 agent_loop（system prompt 已规定
        问候闲聊直接回应、不调工具），由 LLM 正常打招呼。

        限制长度到 12 字符以内，避免把"你好，我想问一下我的卡为什么被冻结了"
        这类真正带业务诉求的长句误判为闲聊。
        """
        q = query.strip().lower()
        if len(q) > 12:
            return False
        return any(g in q for g in _GREETING_WORDS)

    def _needs_live_data(self, query: str) -> bool:
        """数据型问题识别：问的是"具体数字"，必须查产品参数表而非知识库。

        为什么高置信也要拦下来（这是换皮到银行场景后发现的最隐蔽一个坑）：
        知识库讲**规则**，产品参数表讲**数字**。客户问"信用卡分 12 期手续费率是多少"，
        知识库 kb-008 命中率极高（sim=0.81）会被直接 rag_direct 拿去作答，
        但它只能给出区间"一般在 7%~9%" —— 而产品表里躺着精确值 7.20%。
        客户拿着一个区间去网点，等于没答。

        更关键的是：挂牌利率、费率是会调整的。制度文档更新慢，参数表更新快，
        凭知识库回答数字类问题，等于在给客户过期信息。银行场景下这是事故。

        判定方式：同时出现「数字疑问词」+「数字主题词」才算。
        这样"企业网银转账要几个人审批"（讲流程，知识库答得了）不会被误伤，
        而"跨行转账手续费多少"会被送去查表。

        这条规则也让 Agent 的角色更完整：它不只在中低置信时兜底，
        还在"知识库答得了但不该由它答"的时候主动接管。
        """
        q = query.lower()
        has_hint = any(w in q for w in _DATA_HINT_WORDS)
        has_theme = any(t in q for t in _DATA_THEMES)
        return has_hint and has_theme

    def _needs_rule_check(self, query: str) -> bool:
        """条件判定型识别：问的是「这笔业务能不能办」，必须查结构化业务规则表。

        为什么和 _needs_live_data 分开（这是换皮时第二个隐蔽的坑）：
        制度文档（知识库）讲的是**一般规定**——"部分银行要求单笔 50 万以上预约"，
        口径模糊、还有"部分/一般/视情况"这种免责措辞。
        而业务规则表里躺着的是**逐条判定**："单笔 > 50 万 → 需提前 1 个工作日预约"。

        坐席在电话里被客户追问「我这 60 万到底要不要预约」，拿一句"部分银行要求"
        是答不了业务的；必须落到具体条件。同理「挂失能不能让人代办」涉及合规
        （allow_agent 字段），绝不能凭常识推断。

        判定方式：出现业务场景词，且（带具体金额 或 带判定词）才算。
        这样"大额转账要预约吗"（泛问规定，知识库答得了）不被误伤，
        而"客户要转 60 万，需要提前预约吗"（带具体金额）会被送去查规则表。
        """
        q = query.lower()
        has_scene = any(s in q for s in _RULE_SCENES)
        has_amount = bool(_RULE_AMOUNT_RE.search(query))
        has_cond = any(w in q for w in _RULE_COND_WORDS)
        return has_scene and (has_amount or has_cond)

    def _is_out_of_scope(self, query: str) -> bool:
        """行外/禁答话题护栏：证券投资建议、博彩等。

        为什么单列一条护栏而不是靠阈值兜底：
        "帮我推荐一只股票"这类问题与知识库里的"基金定投/理财风险等级"语义接近，
        RAG 相似度会落在 [SIM_LOW, SIM_HIGH) 区间被判进 agent_loop ——
        于是 Agent 会真的去"回答"它。但银行无投顾牌照，**不得提供证券投资建议**，
        这不是答得好不好的问题，是能不能答的问题。属于硬合规红线，必须显式拦截。
        """
        return bool(_OUT_OF_SCOPE_RE.search(query))

    def route(self, user_message: str, history: list = None) -> RouteResult:
        # ① 先做一次 RAG 检索，拿置信度信号
        rag_raw = self.rag_tool.call({"query": user_message, "top_k": 3})
        rag = json.loads(rag_raw)
        sim = rag.get("best_sim", 0.0)
        kw_cov = rag.get("best_kw_coverage", 0.0)
        kb_hits = rag.get("results", [])

        # 透传 event_hook 给 Agent（SSE 流式用）
        self.agent.event_hook = self.event_hook

        # ①.5 行外护栏最先判：合规红线优先于任何置信度信号（见 _is_out_of_scope）
        if self._is_out_of_scope(user_message):
            self._emit("route", {"route": "refuse_human",
                                 "sim": sim, "kw_coverage": kw_cov,
                                 "reason": "out_of_scope"})
            return self._route_refuse_human(user_message, sim, kw_cov, kb_hits)

        # ② 阈值决策
        high_conf = sim >= self.sim_high and kw_cov >= self.kw_coverage_low
        if high_conf:
            # 高置信也可能「知识库答得了但不该由它答」：
            #   数字类 → 知识库只给区间，精确值在产品参数表（_needs_live_data）
            #   判定类 → 制度原文口径模糊，逐条判定在业务规则表（_needs_rule_check）
            # 这两种情况强制走 Agent 查表，让 Agent 不只在中低置信时兜底，
            # 还在「高置信但不该直答」时主动接管。
            if self._needs_live_data(user_message):
                reason = "data_query"
            elif self._needs_rule_check(user_message):
                reason = "rule_check"
            else:
                self._emit("route", {"route": "rag_direct",
                                     "sim": sim, "kw_coverage": kw_cov})
                return self._route_rag_direct(user_message, sim, kw_cov, kb_hits, history)
            self._emit("route", {"route": "agent_loop",
                                 "sim": sim, "kw_coverage": kw_cov,
                                 "reason": reason})
            return self._route_agent_loop(user_message, sim, kw_cov, kb_hits, history)
        if sim >= self.sim_low:
            self._emit("route", {"route": "agent_loop",
                                 "sim": sim, "kw_coverage": kw_cov})
            return self._route_agent_loop(user_message, sim, kw_cov, kb_hits, history)
        # 低置信：先过意图检测，动作型查询救回 agent_loop（避免误拒答）
        if self._detect_intent(user_message):
            self._emit("route", {"route": "agent_loop",
                                 "sim": sim, "kw_coverage": kw_cov,
                                 "reason": "intent_override"})
            return self._route_agent_loop(user_message, sim, kw_cov, kb_hits, history)
        # 寒暄短路：客户开口说"你好"不能被判为转人工（见 _is_smalltalk）
        if self._is_smalltalk(user_message):
            self._emit("route", {"route": "agent_loop",
                                 "sim": sim, "kw_coverage": kw_cov,
                                 "reason": "smalltalk"})
            return self._route_agent_loop(user_message, sim, kw_cov, kb_hits, history)
        self._emit("route", {"route": "refuse_human",
                             "sim": sim, "kw_coverage": kw_cov})
        return self._route_refuse_human(user_message, sim, kw_cov, kb_hits)

    # ---------- 三条路径 ----------

    def _route_rag_direct(self, query, sim, kw_cov, kb_hits, history=None) -> RouteResult:
        """高置信：把命中片段交给 LLM 组织成自然语言答案，不进 Agent 循环。

        银行场景下这条路径的价值不只是"快"：答案完全基于命中的制度条款生成，
        并强制引用片段编号，坐席可以一键复核原文 —— 这就是"可溯源"。
        """
        context = "\n".join(f"[{kb['id']}] {kb['text']}" for kb in kb_hits[:1])
        prompt = (f"基于以下银行业务知识库片段，回答坐席提出的问题。"
                  f"答案里要引用片段编号，并给出客户下一步该做什么。\n\n"
                  f"知识片段：\n{context}\n\n问题：{query}")

        messages = [
            {"role": "system",
             "content": "你是银行远程银行中心的坐席知识助手，严格基于知识库片段作答，"
                        "不得补充片段中没有的规定。"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        resp = self.agent.llm.chat(messages)
        answer = resp.choices[0].message.content or ""
        self._emit("answer_chunk", {"content": answer})
        self._emit("answer_done", {})

        return RouteResult(route="rag_direct", answer=answer,
                          sim=sim, kw_coverage=kw_cov, kb_hits=kb_hits[:1])

    def _route_agent_loop(self, query, sim, kw_cov, kb_hits, history=None) -> RouteResult:
        """中置信：把 RAG 命中片段作为先验 observation 注入 Agent 初始上下文。

        门控：只有 sim >= sim_low 才注入先验。sim 低于阈值的命中实际不相关，
        注入只会误导 LLM（如"我卡丢了怎么办"sim=0.0，却注入了信用卡分期的 KB，
        导致 LLM 开头复述分期规则，而不是直接调工具查工单）。
        sim < sim_low 时（意图检测救回的查询）纯走 Agent 调工具，不注入先验。
        """
        prior = ""
        if kb_hits and sim >= self.sim_low:
            prior = f"\n\n[知识库先验，相似度 {sim:.2f}] {kb_hits[0]['text']}"
        enriched_query = query + prior

        result = self.agent.run(enriched_query, history=history)
        return RouteResult(route="agent_loop", answer=result.answer,
                          sim=sim, kw_coverage=kw_cov,
                          tool_trace=result.tool_trace,
                          kb_hits=kb_hits[:1] if sim >= self.sim_low else [])

    def _route_refuse_human(self, query, sim, kw_cov, kb_hits) -> RouteResult:
        """低置信：超出知识库范围，转人工并说明原因，避免模型幻觉。

        这条路径在银行场景里是**合规要求**而非妥协：宁可转人工，
        也不能让模型对着没有依据的问题编一个听起来合理的答案。
        """
        answer = (f"这个问题超出了当前知识库的覆盖范围（最高相似度 {sim:.2f}，"
                  f"低于阈值 {self.sim_low}）。为避免给出不准确的信息，"
                  f"已为您转接人工坐席处理。请保持通话，坐席会核验您的身份后为您办理。")
        self._emit("answer_chunk", {"content": answer})
        self._emit("answer_done", {})
        return RouteResult(route="refuse_human", answer=answer,
                          sim=sim, kw_coverage=kw_cov, kb_hits=[])
