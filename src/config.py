"""配置与决策阈值（多 Agent v2）。

阈值沿用 v1 标定结果，技术设计文档有完整标定依据。
"""
import os
from pathlib import Path


def _load_dotenv():
    """从 .env 加载环境变量（不覆盖已有值）。零依赖手动实现。"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "src" / "db" / "bank.db"
CHROMA_PATH = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "bank_kb"

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("LLM_API_KEY"))


def require_api_key() -> str:
    """运行时校验 API key（import 时不 raise，保证 mock 测试可跑）。"""
    if not API_KEY:
        raise RuntimeError(
            "请在 .env 中设置 DEEPSEEK_API_KEY（参考 .env.example）")
    return API_KEY


# ============================================================
# 决策路由阈值（v1 标定沿用）
#   综合评分 = 0.7*sim + 0.3*kw_coverage
#   正样本 sim∈[0.445,0.880] 均值 0.699；负样本 sim∈[0.066,0.434] 均值 0.281
# ============================================================
SIM_HIGH = 0.55        # >= 高置信：RAG 直答
SIM_LOW = 0.40         # < 低置信：拒答转人工；[SIM_LOW, SIM_HIGH) 走 Agent 补检索
KW_COVERAGE_LOW = 0.15  # 关键词覆盖率下限，与 sim 联合判定高置信

# Supervisor 循环硬上限（防死循环；超限强制收敛到 synthesis）
MAX_ITERATIONS = 3

# Guard 规则引擎关键词（纯规则，不依赖 LLM；命中即确定，合规第一）
GUARD_REFUSE_PATTERNS = [
    "荐股", "推荐股票", "推荐基金", "保证收益", "稳赚", "代客决策",
    "代客理财", "预测涨跌", "内部消息", "内幕消息",
]
GUARD_ESCALATE_FUND_SAFETY = [
    "盗刷", "被盗", "非本人交易", "诈骗", "被骗", "资金被盗",
    "卡被盗", "伪卡", "疑似盗刷",
]
GUARD_ESCALATE_COMPLAINT = [
    "投诉", "态度差", "态度不好", "不满意", "我要投诉", "差评", "投诉你们",
]
GUARD_ESCALATE_REQUEST_HUMAN = [
    "转人工", "找人工", "要人工", "人工服务", "真人客服", "接人工",
]
