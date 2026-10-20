from openai import OpenAI


class LLMClient:
    """OpenAI 兼容 LLM 客户端。DeepSeek/Qwen 本地服务都是同一套接口。"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client=OpenAI(base_url=base_url, api_key=api_key)
        self.model=model

    def chat(self, messages: list, tools: list | None = None,
             stream: bool = False, tool_choice: object = None) -> object:
        """调用 chat.completions.create。
        - messages:    OpenAI 消息格式
        - tools:       OpenAI 工具 schema 列表
        - stream:      流式开关（第 3 周 SSE 用）
        - tool_choice: 约束本轮必须调用哪个工具。None = 模型自由决定；
                       传 {"type":"function","function":{"name":...}} = 强制调用。
                       用于「不允许模型犹豫」的确定性路径（资金安全 → 立即转人工）。
        返回原始 response 对象，解析交给 Agent 层。
        """
        kwargs = {"model": self.model, "messages": messages,
                  "tools": tools, "stream": stream}
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kwargs)