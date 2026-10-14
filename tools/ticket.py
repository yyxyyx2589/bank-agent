"""业务工单查询工具。

替代 ops-agent 的 QueryLogTool（查运维日志）：场景从「排查服务故障」换成
「了解客户历史诉求」。接口形态保持一致（关键词 + 时间回溯 + 格式化输出），
底层从 logs 表换成 tickets 表。
"""
import json
import sys
from pathlib import Path

from tools import register_tool
from tools.base import BaseTool

# 让 tools 能找到 db 模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from db.db import query_tickets


@register_tool
class QueryTicketTool(BaseTool):
    name = "query_ticket"
    description = (
        "查询客户的历史业务工单和来电记录。"
        "当需要了解『这位客户之前来问过什么』『他有没有相关诉求记录』"
        "『这个账户之前是否被处理过』时调用。"
        "支持按客户号精确定位，也可按关键词（如 挂失/转账/冻结/理财）和类别过滤。"
        "返回工单号、时间、渠道、类别、状态、摘要与受理人。"
        "客户号格式为 C 开头的 15 位编号，如 C2026031800237；"
        "若用户未提供客户号，可只用关键词或类别查询。")
    parameters = [
        {"name": "customer_id", "type": "string",
         "description": "客户号，如 C2026031800237。已知客户身份时优先使用",
         "required": False},
        {"name": "keyword", "type": "string",
         "description": "工单关键词，如 挂失、转账、冻结、理财、信用卡",
         "required": False},
        {"name": "category", "type": "string",
         "description": "工单类别：挂失补卡 / 转账限额 / 信用卡 / 理财咨询 / "
                        "账户冻结 / 对公业务 / 投诉建议",
         "required": False},
        {"name": "hours", "type": "integer",
         "description": "回溯时间范围（小时），默认 720（30 天）", "required": False},
    ]

    def call(self, params: dict) -> str:
        customer_id = params.get("customer_id")
        keyword = params.get("keyword")
        category = params.get("category")
        hours = params.get("hours", 720)

        # 三个过滤条件全空会拉全量工单，既贵又没意义，直接拒绝并给出提示
        if not any([customer_id, keyword, category]):
            return json.dumps(
                {"error": "至少需要提供 customer_id / keyword / category 之一，"
                          "否则将返回全量工单而无法定位"},
                ensure_ascii=False)

        tickets = query_tickets(customer_id=customer_id, category=category,
                                keyword=keyword, hours=hours, limit=20)

        if not tickets:
            return json.dumps(
                {"result": "未找到匹配的工单记录",
                 "customer_id": customer_id, "keyword": keyword,
                 "category": category, "hours": hours},
                ensure_ascii=False)

        # 格式化成易读文本（LLM 看 observation 更容易理解）
        formatted = []
        for t in tickets:
            formatted.append(
                f"{t['timestamp']} [{t['category']}/{t['status']}] "
                f"{t['customer_name']}({t['customer_id']}) {t['channel']}：{t['summary']}")

        return json.dumps(
            {"customer_id": customer_id, "keyword": keyword,
             "category": category, "count": len(tickets), "tickets": formatted},
            ensure_ascii=False)
