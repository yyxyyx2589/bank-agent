"""工具注册中心（多 Agent v2）。"""
from src.tools.base import TOOL_REGISTRY, register_tool, get_tool  # noqa: F401

# 导入触发注册
from src.tools.rag_search import SearchKnowledgeTool  # noqa: E402,F401
from src.tools.product_rate import GetProductRateTool  # noqa: E402,F401
from src.tools.business_rule import CheckBusinessRuleTool  # noqa: E402,F401
from src.tools.query_ticket import QueryTicketTool  # noqa: E402,F401
from src.tools.ticket_ops import CreateTicketTool, EscalateToHumanTool  # noqa: E402,F401
