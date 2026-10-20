from abc import ABC, abstractmethod

TOOL_REGISTRY: dict = {}


def register_tool(cls):
    """装饰器：实例化工具并注册到全局表。"""
    tool = cls()
    TOOL_REGISTRY[tool.name] = tool
    return cls


def get_tool(name: str):
    return TOOL_REGISTRY.get(name)


class BaseTool(ABC):
    """所有工具的基类。description 和 parameters 会被注入给 LLM。"""

    name: str            # 工具名，如 "search_knowledge"
    description: str     # 给 LLM 看的功能描述——写得好坏直接决定调用准确率
    parameters: list     # JSON Schema 参数声明

    @abstractmethod
    def call(self, params: dict) -> str:
        """执行工具。入参是 LLM 生成的参数 dict，返回值必须是 str。"""

def to_openai_schema(tool: BaseTool) -> dict:
    """把 BaseTool 转成 OpenAI tools 参数格式：
    {"type": "function", "function": {"name", "description", "parameters"}}
    注意 OpenAI 的 parameters 是 JSON Schema 对象（properties/required），要做一次格式转换。"""
    properties={}
    for p in tool.parameters:
        properties[p["name"]]={"type": p["type"], "description": p["description"]}
    required = [p["name"] for p in tool.parameters if p.get("required", False)]

    return {"type": "function", "function": {"name": tool.name,
                                             "description": tool.description,
                                             "parameters": {"type": "object",
                                                            "properties": properties,
                                                            "required": required}}}