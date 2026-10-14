import os
from pathlib import Path


def _load_dotenv():
    """从 .env 文件加载环境变量（不覆盖已有值）。零依赖手动实现。

    用法：把 DEEPSEEK_API_KEY=sk-xxx 写进项目根目录的 .env 文件
    （已被 .gitignore 排除，不会误提交）。双击运行/换终端都不用重新 export。
    """
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"  # 若报 model not found，以 DeepSeek 文档页列出的模型名为准
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")


def require_api_key() -> str:
    """运行时校验 API key（不在 import 时 raise——import 副作用会破坏可测试性，
    mock 测试和路由测试不应被真实 key 绑架；只有真正要调 LLM 时才校验）。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "请先设置环境变量 DEEPSEEK_API_KEY，或在项目根目录 .env 文件中写入"
            " DEEPSEEK_API_KEY=sk-xxx（参考 .env.example）")
    return DEEPSEEK_API_KEY

# ============================================================
# 决策路由阈值
# ============================================================
# 【换知识库后必须重新标定 —— 这是本作品从 ops-agent 换皮时做得最实的一件事】
#
# 原运维项目的 0.70 / 0.45 / 0.30 是针对 15 条运维 KB 调出来的。换成 24 条
# 银行业务 KB 后，相似度分布整体下移：embedding 模型没变，但中文业务长文本
# 之间的语义区分度低于英文技术术语（"开卡/补卡/挂失"彼此高度接近）。
# 直接沿用旧阈值会导致「标准业务问题被判为低置信、走拒答转人工」——线上就是事故。
#
# 实测重标（tools/rag_search.py 综合评分 = 0.7*sim + 0.3*kw_coverage）：
#   正样本 24 条银行问题（top1 命中正确 KB）：sim ∈ [0.445, 0.880]，均值 0.699
#                                               kw_coverage ∈ [0.167, 0.600]
#   负样本 7 条知识库外问题：                  sim ∈ [0.066, 0.434]，均值 0.281
#                                               kw_coverage ≤ 0.143
#
# 标定依据：
#   SIM_HIGH 取 0.55 —— 位于正样本均值与负样本上限之间，且负样本最大值 0.434
#                       距此线仍有 0.116 余量，保证知识库外问题不会被直答。
#   SIM_LOW  取 0.40 —— 正样本最低 0.445 距此线留 0.045 余量，避免标准业务问题
#                       被误判进拒答；同时能拒掉 sim<0.40 的明显无关提问。
#   KW_COVERAGE_LOW 取 0.15 —— 卡在正样本最低 0.167 与负样本最高 0.143 之间，
#                       作为 sim 之外的第二个门槛，拦截「语义像但关键词完全不搭」。
#
# 重标后的路由分布：24 条正样本中 22 条走 rag_direct，2 条走 agent_loop，
#                   0 条被拒答；7 条负样本 0 条被误判为直答。
SIM_HIGH = 0.55        # >= 高置信：直接走 RAG 直答（模型只需组织语言）
SIM_LOW = 0.40         # < 低置信：拒答转人工；[SIM_LOW, SIM_HIGH) 走 Agent 循环
KW_COVERAGE_LOW = 0.15  # 关键词覆盖率下限，与 sim 联合判定高置信
