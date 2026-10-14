"""SQLite 数据库连接模块（银行坐席场景）。

四张表：
  tickets        业务工单（ticket_no, timestamp, customer_id, customer_name,
                channel, category, status, summary, handler）
  products       产品费率（product_code, name, category, term, rate, rate_desc,
                min_amount, risk_level, effective_date, note）
  business_rules 业务规则（rule_code, scenario, condition, requirement,
                limit_amount, allow_agent, note）
  sessions       会话记忆（session_id, role, content, created_at）

初始化：python db/init_db.py（建表 + 生成数据）
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "bank.db"


def get_conn() -> sqlite3.Connection:
    """获取数据库连接。Row 工厂让结果可按列名访问。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_cursor():
    """上下文管理器：自动提交 + 关闭。工具层用这个。"""
    conn = get_conn()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def query_tickets(customer_id: str = None, category: str = None,
                  keyword: str = None, hours: int = 720,
                  limit: int = 20) -> list[dict]:
    """查业务工单，支持按客户号 / 类别 / 关键词过滤 + 时间回溯。

    设计说明（沿用 ops-agent 的 query_logs 思路）：
    1. keyword 同时匹配 category 和 summary —— 客户说"我卡丢了"时，
       类别是"挂失补卡"而在摘要里，只搜一个字段会漏。
    2. hours 基于 DB 最新工单时间回溯，而不是真实当前时间。预置数据的时间
       窗口与运行时刻可能错位，用 MAX(timestamp) 做基准可避免把历史工单
       全部过滤掉（默认 720 小时 = 30 天）。
    3. 默认按时间倒序，最新工单优先 —— 坐席最关心客户"最近一次来电说了什么"。

    生产化换工单系统（如工单中台 / CRM）时，本函数接口不变，内部改调对方 API。
    """
    sql = ("SELECT ticket_no, timestamp, customer_id, customer_name, channel, "
           "category, status, summary, handler FROM tickets "
           "WHERE timestamp >= "
           "  (SELECT datetime(MAX(timestamp), '-' || ? || ' hours') FROM tickets)")
    params = [hours]
    if customer_id:
        sql += " AND customer_id = ?"
        params.append(customer_id)
    if category:
        sql += " AND category = ?"
        params.append(category)
    if keyword:
        kw = f"%{keyword.lower()}%"
        sql += " AND (LOWER(category) LIKE ? OR LOWER(summary) LIKE ?)"
        params.extend([kw, kw])
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with db_cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def get_product(category: str = None, keyword: str = None,
                term: str = None, limit: int = 20) -> list[dict]:
    """查产品费率，支持按类别 / 名称关键词 / 期限过滤。

    设计说明：坐席查利率时表述很随意（"三年期定存多少""大额存单利率"），
    因此 keyword 同时匹配 name 和 rate_desc 两个字段。
    """
    sql = ("SELECT product_code, name, category, term, rate, rate_desc, "
           "min_amount, risk_level, effective_date, note FROM products WHERE 1=1")
    params = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    if keyword:
        kw = f"%{keyword.lower()}%"
        sql += " AND (LOWER(name) LIKE ? OR LOWER(rate_desc) LIKE ?)"
        params.extend([kw, kw])
    if term:
        sql += " AND term LIKE ?"
        params.append(f"%{term}%")
    sql += " ORDER BY category, rate DESC LIMIT ?"
    params.append(limit)

    with db_cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def check_rule(scenario: str = None, amount: float = None,
               limit: int = 10) -> list[dict]:
    """按业务场景查规则，可选按金额筛选适用条款。

    设计说明：scenario 模糊匹配 scenario / condition / requirement 三个字段，
    因为坐席提问的口径五花八门（"能不能代办""要带什么""超过50万怎么办"）。
    amount 传入时，额外把「有金额门槛且门槛 <= amount」的规则排在前面，
    便于 Agent 直接判断这笔业务是否触发大额条款。
    """
    sql = ("SELECT rule_code, scenario, condition, requirement, limit_amount, "
           "allow_agent, note FROM business_rules WHERE 1=1")
    params = []
    if scenario:
        kw = f"%{scenario.lower()}%"
        sql += (" AND (LOWER(scenario) LIKE ? OR LOWER(condition) LIKE ? "
                "OR LOWER(requirement) LIKE ?)")
        params.extend([kw, kw, kw])
    sql += " ORDER BY (limit_amount IS NULL), limit_amount ASC LIMIT ?"
    params.append(limit)

    with db_cursor() as cur:
        cur.execute(sql, params)
        rows = [dict(row) for row in cur.fetchall()]

    # 金额门槛优先：把本次业务金额已触发的限额规则提到最前
    if amount is not None:
        def rank(r):
            lim = r.get("limit_amount")
            return 0 if (lim is not None and amount >= lim) else 1
        rows.sort(key=rank)
    return rows


def save_message(session_id: str, role: str, content: str):
    """保存一条会话消息。"""
    import datetime as dt
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, role, content, dt.datetime.now().isoformat()))


def load_history(session_id: str) -> list[dict]:
    """加载会话历史（按时间正序）。"""
    with db_cursor() as cur:
        cur.execute(
            "SELECT role, content FROM sessions WHERE session_id = ? ORDER BY id ASC",
            (session_id,))
        return [dict(row) for row in cur.fetchall()]


if __name__ == "__main__":
    # 快速自测
    print("=== query_tickets 测试 ===")
    t = query_tickets(customer_id="C2026031800237")
    print(f"客户 C2026031800237 工单: {len(t)} 条")
    for r in t:
        print(f"  {r['timestamp']} [{r['category']}/{r['status']}] {r['summary'][:44]}")

    print("\n=== get_product 测试 ===")
    p = get_product(category="存款", term="三年")
    for r in p:
        print(f"  {r['name']:22} {r['rate']}%  {r['rate_desc']}")

    print("\n=== check_rule 测试（大额转账 60 万）===")
    rs = check_rule(scenario="转账", amount=600000)
    for r in rs[:3]:
        tag = "【已触发限额】" if r["limit_amount"] and 600000 >= r["limit_amount"] else ""
        print(f"  {r['rule_code']} {r['scenario']} {tag}")
        print(f"      {r['requirement'][:60]}")

    print("\n=== sessions 测试 ===")
    save_message("test-sid", "user", "你好")
    save_message("test-sid", "assistant", "您好，请问有什么可以帮您")
    print(f"会话历史: {load_history('test-sid')}")
    with db_cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE session_id='test-sid'")
    print("测试数据已清理")
