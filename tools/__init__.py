


TOOL_REGISTRY = {}
def register_tool(cls):
    tool = cls()
    TOOL_REGISTRY[tool.name] = tool
    return cls

from tools import ticket     # 业务工单查询（替代 query_log）
from tools import product    # 产品费率查询（替代 get_monitor_metrics）
from tools import rules      # 业务规则校验（替代 run_diagnostic_command）
from tools import escalate   # 转人工（银行版新增：把合规兜底从路由文案升级为工具）
from tools import rag_search # 知识库检索（路由信号源 + Agent 工具池）