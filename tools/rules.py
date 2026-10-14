"""业务规则校验工具。

替代 ops-agent 的 run_diagnostic_command（白名单命令沙箱）：从「这条诊断命令
能不能执行」换成「这笔业务能不能办、需要什么」。两者在架构里承担同一个角色
——**判定型工具**：都在回答「条件是否满足」，所以替换后工具在编排里的位置不变。
"""
import json
import sys
from pathlib import Path

from tools import register_tool
from tools.base import BaseTool

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.db import check_rule


@register_tool
class CheckBusinessRuleTool(BaseTool):
    name = "check_business_rule"
    description = (
        "校验业务办理规则：能不能办、需要带什么材料、能不能他人代办、金额有没有上限。"
        "当客户或坐席询问『这张卡能不能代办挂失』『开卡需要什么材料』"
        "『超过 50 万转账要不要预约』『定期能不能提前取』『买理财要不要先测评』时调用。"
        "传入业务场景关键词，可附上涉及金额，返回适用规则条款。"
        "返回内容包括：规则编号、场景、触发条件、所需材料或动作、金额上限、是否允许代办。"
        "重要：涉及『是否允许他人代办』的判断必须以本工具返回的 allow_agent 字段为准，"
        "不得凭常识推断——这属于合规要求，答错可能造成客户白跑一趟或违规受理。")
    parameters = [
        {"name": "scenario", "type": "string",
         "description": "业务场景关键词，如 挂失、开户、大额转账、提前支取、理财首次购买",
         "required": True},
        {"name": "amount", "type": "number",
         "description": "涉及金额（元）。传入后会优先返回本次业务已触发的大额条款",
         "required": False},
    ]

    def call(self, params: dict) -> str:
        scenario = params.get("scenario", "")
        amount = params.get("amount")

        if not scenario:
            return json.dumps({"error": "scenario 不能为空"}, ensure_ascii=False)

        rules = check_rule(scenario=scenario, amount=amount, limit=8)
        if not rules:
            return json.dumps(
                {"result": "未找到对应业务规则", "scenario": scenario},
                ensure_ascii=False)

        formatted = []
        for r in rules:
            item = {
                "rule_code": r["rule_code"],
                "scenario": r["scenario"],
                "condition": r["condition"],
                "requirement": r["requirement"],
                "allow_agent": bool(r["allow_agent"]),
                "allow_agent_desc": "可由他人代办" if r["allow_agent"] else "必须本人办理",
            }
            if r.get("limit_amount"):
                item["limit_amount"] = r["limit_amount"]
                if amount is not None:
                    item["triggered"] = bool(amount >= r["limit_amount"])
            if r.get("note"):
                item["note"] = r["note"]
            formatted.append(item)

        return json.dumps(
            {"scenario": scenario, "amount": amount,
             "count": len(formatted), "rules": formatted},
            ensure_ascii=False)
