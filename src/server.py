"""FastAPI 服务入口。

POST /chat    多 Agent 对话
GET  /health  健康检查
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import get_app

app = FastAPI(title="银行远程坐席知识助手 v2", version="2.0")


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    gate: str
    confidence_level: str | None
    citations: list[str]
    trace: list[dict]
    iterations: int


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0", "engine": "langgraph-supervisor"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = get_app().invoke({"query": req.query, "messages": [], "trace": []})
    return ChatResponse(
        answer=result.get("answer", ""),
        gate=result.get("gate", "PASS"),
        confidence_level=result.get("confidence_level"),
        citations=result.get("citations", []),
        trace=result.get("trace", []),
        iterations=result.get("iterations", 0),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
