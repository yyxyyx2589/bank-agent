"""产品费率查询工具。

替代 ops-agent 的 get_monitor_metrics（查监控指标）：从「看主机资源」
换成「查产品利率费率」，是坐席最高频的一类查询。
"""
import json
import sys
from pathlib import Path

from tools import register_tool
from tools.base import BaseTool

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.db import get_product


@register_tool
class GetProductRateTool(BaseTool):
    name = "get_product_rate"
    description = (
        "查询银行产品的利率、费率、期限和起存金额。"
        "当客户或坐席询问『XX 利率是多少』『信用卡分期手续费多少』"
        "『大额存单起存多少钱』『理财业绩比较基准是多少』时调用。"
        "支持按产品类别（存款/贷款/信用卡/理财）和名称关键词过滤。"
        "返回产品名、期限、年化利率或费率、人话描述、起购金额与风险等级。"
        "无法判断属于哪一类时可不传 category，只传 keyword。"
        "注意：本工具返回的是挂牌利率，具体执行利率以网点或线上渠道实际报价为准。")
    parameters = [
        {"name": "category", "type": "string",
         "description": "产品类别：存款 / 贷款 / 信用卡 / 理财",
         "required": False},
        {"name": "keyword", "type": "string",
         "description": "产品名或描述关键词，如 大额存单、三年、分期、房贷、LPR",
         "required": False},
        {"name": "term", "type": "string",
         "description": "期限过滤，如 一年、三年、90天", "required": False},
    ]

    def call(self, params: dict) -> str:
        category = params.get("category")
        keyword = params.get("keyword")
        term = params.get("term")

        if not any([category, keyword, term]):
            return json.dumps(
                {"error": "至少需要提供 category / keyword / term 之一，"
                          "否则会返回全部产品"},
                ensure_ascii=False)

        products = get_product(category=category, keyword=keyword, term=term, limit=20)
        if not products:
            return json.dumps(
                {"result": "未找到匹配的产品",
                 "category": category, "keyword": keyword, "term": term},
                ensure_ascii=False)

        formatted = []
        for p in products:
            line = f"{p['name']}（{p['term']}）：{p['rate_desc']}"
            if p.get("min_amount"):
                line += f"，起购 {int(p['min_amount'])} 元"
            if p.get("risk_level"):
                line += f"，风险等级 {p['risk_level']}"
            if p.get("note"):
                line += f"（{p['note']}）"
            formatted.append(line)

        return json.dumps(
            {"category": category, "keyword": keyword, "term": term,
             "count": len(products), "products": formatted},
            ensure_ascii=False)
