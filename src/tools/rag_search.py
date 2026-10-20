"""知识库检索工具（RAG）。

双重身份（面试可讲）：
  1. 被 Knowledge Agent 调用 —— 拿 sim/kw_coverage 派生置信度，喂给 Supervisor 路由
  2. 提供证据包给 Synthesis 合成可溯源答案

embedding: paraphrase-multilingual-MiniLM-L12-v2（384 维，含中文）
向量库: ChromaDB（cosine，data/chroma/）
相似度: 手动算余弦（ChromaDB distance 是 squared L2，不直接用）
"""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.tools.base import BaseTool, register_tool

_embed_model = None
_chroma_collection = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embed_model


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        from src.config import CHROMA_PATH, COLLECTION_NAME
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0


def _kw_coverage(query: str, keywords_str: str) -> float:
    q = query.lower()
    kws = [k.strip() for k in keywords_str.split(",") if k.strip()]
    if not kws:
        return 0.0
    return sum(1 for k in kws if k.lower() in q) / len(kws)


@register_tool
class SearchKnowledgeTool(BaseTool):
    name = "search_knowledge"
    description = (
        "检索银行业务知识库，返回与提问最相关的制度条款和业务规则片段。"
        "用于开户材料、转账限额、挂失补卡、信用卡分期、贷款利率、存款利率、"
        "理财产品、对公账户、账户冻结等业务规则查询。"
        "返回结果含相似度 sim、关键词覆盖率 kw_coverage，供置信度路由。")
    parameters = [
        {"name": "query", "type": "string", "description": "要检索的问题或关键词", "required": True},
        {"name": "top_k", "type": "integer", "description": "返回前 K 条，默认 3", "required": False},
    ]

    def call(self, params: dict) -> str:
        query = params.get("query", "")
        top_k = params.get("top_k", 3)
        if not query:
            return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)
        try:
            model = _get_embed_model()
            collection = _get_collection()
        except Exception as e:
            return json.dumps({"error": f"向量库未初始化: {e}",
                               "hint": "运行 python scripts/build_kb.py"}, ensure_ascii=False)

        import numpy as np
        q_emb = model.encode([query]).tolist()
        candidates = collection.query(
            query_embeddings=q_emb,
            n_results=min(max(top_k * 3, 5), 15),
            include=["documents", "metadatas"])

        q_vec = np.array(q_emb[0])
        scored = []
        for i, doc_id in enumerate(candidates["ids"][0]):
            doc_text = candidates["documents"][0][i]
            kws = candidates["metadatas"][0][i].get("keywords", "")
            kb_data = collection.get(ids=[doc_id], include=["embeddings"])
            kb_vec = np.array(kb_data["embeddings"][0])
            sim = _cosine(q_vec, kb_vec)
            kwc = _kw_coverage(query, kws)
            scored.append({
                "id": doc_id, "text": doc_text,
                "sim": round(sim, 3), "kw_coverage": round(kwc, 3),
                "score": round(0.7 * sim + 0.3 * kwc, 3),
                "keywords": kws,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:max(top_k, 1)]
        if not top:
            return json.dumps({"query": query, "best_sim": 0.0,
                               "best_kw_coverage": 0.0, "results": []}, ensure_ascii=False)
        best = top[0]
        return json.dumps({
            "query": query,
            "best_sim": best["sim"],
            "best_kw_coverage": best["kw_coverage"],
            "results": top,
        }, ensure_ascii=False)
