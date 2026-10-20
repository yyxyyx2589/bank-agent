"""历史工单查询工具（Rules Agent 调用）。查 tickets 表。

v1 的 query_ticket 工具迁移。处理『客户之前来问过什么』『工单处理状态』类诉求。
"""
import json
from src.tools.base import BaseTool, register_tool
from src.db import query_tickets


@register_tool
class QueryTicketTool(BaseTool):
    name = "query_ticket"
    description = (
        "查询客户的历史业务工单和来电记录。"
        "用于『这位客户之前来问过什么』『工单处理状态』『有没有相关诉求记录』。"
        "支持按客户号精确定位，也可按关键词和类别过滤。"
        "客户号格式 C 开头 15 位，如 C2026031800237。")
    parameters = [
        {"name": "customer_id", "type": "string", "description": "客户号，如 C2026031800237", "required": False},
        {"name": "keyword", "type": "string", "description": "工单关键词，如 挂失、转账、投诉", "required": False},
        {"name": "category", "type": "string", "description": "工单类别：挂失补卡/转账限额/信用卡/理财咨询/账户冻结/对公业务/投诉建议", "required": False},
        {"name": "hours", "type": "integer", "description": "回溯小时数，默认 720（30天）", "required": False},
    ]

    def call(self, params: dict) -> str:
        customer_id = params.get("customer_id")
        keyword = params.get("keyword")
        category = params.get("category")
        hours = params.get("hours", 720)
        if not any([customer_id, keyword, category]):
            return json.dumps({"error": "至少提供 customer_id/keyword/category 之一"}, ensure_ascii=False)
        tickets = query_tickets(customer_id=customer_id, category=category,
                                keyword=keyword, hours=hours, limit=20)
        if not tickets:
            return json.dumps({"result": "未找到匹配工单", "customer_id": customer_id,
                               "keyword": keyword, "category": category}, ensure_ascii=False)
        formatted = [f"{t['timestamp']} [{t['category']}/{t['status']}] "
                     f"{t.get('customer_name','')}({t['customer_id']}) {t['channel']}：{t['summary']}"
                     for t in tickets]
        return json.dumps({"count": len(tickets), "tickets": formatted}, ensure_ascii=False)
