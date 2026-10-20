"""业务规则校验工具（Rules Agent 调用）。查 business_rules 表。

判定型工具——回答「能不能办、要带什么、能不能代办」。涉及是否允许
他人代办必须以 allow_agent 字段为准，不得凭常识推断（合规要求）。
"""
import json
from src.tools.base import BaseTool, register_tool
from src.db import check_rule


@register_tool
class CheckBusinessRuleTool(BaseTool):
    name = "check_business_rule"
    description = (
        "校验业务办理规则：能不能办、需要什么材料、能不能他人代办、金额有没有上限。"
        "用于『卡能不能代办挂失』『开卡要什么材料』『超 50 万转账要预约吗』"
        "『定期能不能提前取』『买理财要不要先测评』等。"
        "返回规则编号、场景、触发条件、所需材料、金额上限、是否允许代办。"
        "涉及是否允许他人代办必须以 allow_agent 字段为准。")
    parameters = [
        {"name": "scenario", "type": "string", "description": "业务场景关键词，如 挂失、开户、大额转账、提前支取、理财首次购买", "required": True},
        {"name": "amount", "type": "number", "description": "涉及金额（元），传入会标记触发的大额条款", "required": False},
    ]

    def call(self, params: dict) -> str:
        scenario = params.get("scenario", "")
        amount = params.get("amount")
        if not scenario:
            return json.dumps({"error": "scenario 不能为空"}, ensure_ascii=False)
        rules = check_rule(scenario=scenario, amount=amount, limit=8)
        if not rules:
            return json.dumps({"result": "未找到对应业务规则", "scenario": scenario}, ensure_ascii=False)
        formatted = [{
            "rule_code": r["rule_code"], "scenario": r["scenario"],
            "condition": r["condition"], "requirement": r["requirement"],
            "allow_agent": bool(r["allow_agent"]),
            "allow_agent_desc": "可由他人代办" if r["allow_agent"] else "必须本人办理",
            "triggered": r.get("triggered", False),
        } for r in rules]
        return json.dumps({"count": len(formatted), "rules": formatted}, ensure_ascii=False)
