"""产品费率查询工具（Rules Agent 调用）。查 products 表。"""
import json
from src.tools.base import BaseTool, register_tool
from src.db import get_product


@register_tool
class GetProductRateTool(BaseTool):
    name = "get_product_rate"
    description = (
        "查询银行产品的利率、费率、期限和起存金额。"
        "用于『XX 利率多少』『信用卡分期手续费』『大额存单起存』『理财业绩比较基准』等。"
        "支持按类别（存款/贷款/信用卡/理财）和名称关键词过滤。"
        "返回挂牌利率，具体执行利率以网点实际报价为准。")
    parameters = [
        {"name": "category", "type": "string", "description": "产品类别：存款/贷款/信用卡/理财", "required": False},
        {"name": "keyword", "type": "string", "description": "产品名或关键词，如 大额存单、三年、分期、房贷", "required": False},
        {"name": "term", "type": "string", "description": "期限过滤，如 一年、三年、90天", "required": False},
    ]

    def call(self, params: dict) -> str:
        category = params.get("category")
        keyword = params.get("keyword")
        term = params.get("term")
        if not any([category, keyword, term]):
            return json.dumps({"error": "至少提供 category/keyword/term 之一"}, ensure_ascii=False)
        products = get_product(category=category, keyword=keyword, term=term, limit=20)
        if not products:
            return json.dumps({"result": "未找到匹配产品",
                               "category": category, "keyword": keyword}, ensure_ascii=False)
        return json.dumps({"count": len(products), "products": products}, ensure_ascii=False)
