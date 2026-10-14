"""转人工工具（银行版新增 —— 原 ops-agent 没有这个工具）。

为什么必须新增它：

在 ops-agent 里，「转人工」是 Router 在低置信分支输出的一句**兜底文案**——
是架构层被动触发的，Agent 自己并没有「请求转人工」的能力。

银行场景下，这个能力必须显式化。原因有三：
  1. **合规**：坐席助手不能替客户作出超出权限的承诺，遇到涉及资金安全、
     需双人复核、突破权限的场景，转人工是唯一正确动作，且必须留痕可追溯。
  2. **风险分层**：把「该不该自己处理」的判断交给 Agent 主动做，而不是
     等相似度掉到阈值以下才被动兜底——这样即便检索置信度很高，
     Agent 也能识别出「这个问题我不该答」。
  3. **架构叙事**：它是「从 RAG 演进到 Agent」最直接的证据——
     工具池里多了一个「自我限制」的工具，说明 Agent 具备了对自身边界的判断力。

调用后返回可追溯的转接工单号，前端时间线会展示这一步。
"""
import json
import random
from datetime import datetime

from tools import register_tool
from tools.base import BaseTool


@register_tool
class EscalateToHumanTool(BaseTool):
    name = "escalate_to_human"
    description = (
        "将当前问题转接人工坐席，并生成可追溯的转接记录。"
        "遇到以下任一情形必须调用本工具，不得自行作答："
        "① 涉及具体客户的资金安全、赔付责任或法律责任；"
        "② 需要突破当前坐席权限或需双人复核（如大额资金划转、账户解控审批）；"
        "③ 客户明确要求人工服务，或已表达强烈不满；"
        "④ 知识库检索与其余工具都无法给出可靠依据。"
        "调用后应告知客户已转人工的原因、优先级与后续流程，不要只报工单号。")
    parameters = [
        {"name": "reason", "type": "string",
         "description": "转人工的具体原因，一句话说明为什么不能由智能助手处理",
         "required": True},
        {"name": "priority", "type": "string",
         "description": "优先级：常规 / 优先 / 紧急", "required": False},
        {"name": "suggested_team", "type": "string",
         "description": "建议转接团队，如 银行卡业务组 / 信贷审批组 / 反欺诈组 / 对公业务组",
         "required": False},
    ]

    def call(self, params: dict) -> str:
        reason = params.get("reason", "")
        priority = params.get("priority", "常规")
        team = params.get("suggested_team", "综合业务组")

        if not reason:
            return json.dumps({"error": "reason 不能为空"}, ensure_ascii=False)

        now = datetime.now()
        transfer_no = f"ZR{now.strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"

        return json.dumps(
            {"status": "已转人工",
             "transfer_no": transfer_no,
             "priority": priority,
             "suggested_team": team,
             "reason": reason,
             "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
             "message": f"已为您转接{team}人工坐席（工单号 {transfer_no}），"
                        f"优先级：{priority}。请保持通话，坐席将核验您的身份后为您处理。"},
            ensure_ascii=False)
