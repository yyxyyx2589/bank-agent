"""工单处置工具（Ticket Agent 调用）——系统唯一的副作用写操作。

create_ticket: 建业务工单（落 SQLite）
escalate_to_human: 转人工（生成可追溯转接单号，不落库——转接是即时动作）

权限隔离设计：这两个工具只在 Ticket Agent 节点出现，其余 Agent 拿不到。
爆炸半径最小化：出事时只需审查 Ticket Agent 调用日志。
"""
import json
import random
from datetime import datetime

from src.tools.base import BaseTool, register_tool
from src.db import create_ticket as _create_ticket


@register_tool
class CreateTicketTool(BaseTool):
    name = "create_ticket"
    description = (
        "为客户创建业务工单，落系统留痕。用于坐席确认需要后续跟进的业务诉求。"
        "返回工单号、类别、状态。")
    parameters = [
        {"name": "customer_id", "type": "string", "description": "客户号（C 开头 15 位），未知可传 空字符串", "required": True},
        {"name": "category", "type": "string", "description": "工单类别：挂失补卡/转账限额/信用卡/理财咨询/账户冻结/对公业务/投诉建议", "required": True},
        {"name": "summary", "type": "string", "description": "工单摘要，一句话说明客户诉求", "required": True},
    ]

    def call(self, params: dict) -> str:
        customer_id = params.get("customer_id", "")
        category = params.get("category", "综合业务")
        summary = params.get("summary", "")
        if not category or not summary:
            return json.dumps({"error": "category 和 summary 不能为空"}, ensure_ascii=False)
        t = _create_ticket(customer_id=customer_id, category=category, summary=summary)
        return json.dumps(t, ensure_ascii=False)


@register_tool
class EscalateToHumanTool(BaseTool):
    name = "escalate_to_human"
    description = (
        "将当前问题转接人工坐席，生成可追溯转接记录。遇到以下情形必须调用，不得自行作答："
        "① 涉及具体客户资金安全、赔付或法律责任；"
        "② 需突破坐席权限或双人复核（大额划转、账户解控审批）；"
        "③ 客户明确要求人工服务或表达强烈不满；"
        "④ 知识库与其余工具都无法给出可靠依据。"
        "调用后应告知客户转接原因、优先级与后续流程。")
    parameters = [
        {"name": "reason", "type": "string", "description": "转人工原因，一句话说明为什么不能由智能助手处理", "required": True},
        {"name": "priority", "type": "string", "description": "优先级：常规/优先/紧急", "required": False},
        {"name": "suggested_team", "type": "string", "description": "建议转接团队：银行卡业务组/信贷审批组/反欺诈组/对公业务组", "required": False},
    ]

    def call(self, params: dict) -> str:
        reason = params.get("reason", "")
        priority = params.get("priority", "常规")
        team = params.get("suggested_team", "综合业务组")
        if not reason:
            return json.dumps({"error": "reason 不能为空"}, ensure_ascii=False)
        now = datetime.now()
        transfer_no = f"ZR{now.strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"
        return json.dumps({
            "status": "已转人工",
            "transfer_no": transfer_no,
            "priority": priority,
            "suggested_team": team,
            "reason": reason,
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"已为您转接{team}人工坐席（工单号 {transfer_no}），"
                       f"优先级：{priority}。请保持通话，坐席将核验身份后处理。",
        }, ensure_ascii=False)
