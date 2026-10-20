"""SQLite 数据访问层（银行坐席场景）。

四张表（bank.db，已迁移自 v1）：
  tickets         业务工单
  products        产品费率
  business_rules  业务规则
  sessions        会话记忆
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager

from src.config import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_cursor():
    conn = get_conn()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


# ============================================================
# 查询函数（供工具调用）
# ============================================================
def get_product(category=None, keyword=None, term=None, limit=20) -> list[dict]:
    """查产品费率。category: 存款/贷款/信用卡/理财。"""
    sql = "SELECT * FROM products WHERE 1=1"
    args: list = []
    if category:
        sql += " AND category = ?"
        args.append(category)
    if keyword:
        sql += " AND (name LIKE ? OR note LIKE ?)"
        args += [f"%{keyword}%", f"%{keyword}%"]
    if term:
        sql += " AND term LIKE ?"
        args.append(f"%{term}%")
    sql += " LIMIT ?"
    args.append(limit)
    with db_cursor() as cur:
        cur.execute(sql, args)
        return [dict(r) for r in cur.fetchall()]


def check_rule(scenario=None, amount=None, limit=8) -> list[dict]:
    """查业务规则。amount 传入会优先返回已触发的大额条款。"""
    sql = "SELECT * FROM business_rules WHERE 1=1"
    args: list = []
    if scenario:
        sql += " AND (scenario LIKE ? OR condition LIKE ? OR note LIKE ?)"
        args += [f"%{scenario}%"] * 3
    sql += " LIMIT ?"
    args.append(limit)
    with db_cursor() as cur:
        cur.execute(sql, args)
        rules = [dict(r) for r in cur.fetchall()]
    # 金额触发判定：limit_amount 非 0 且 amount 超过则标记 triggered
    if amount is not None:
        for r in rules:
            lim = r.get("limit_amount") or 0
            r["triggered"] = bool(lim and amount >= lim)
    return rules


def query_tickets(customer_id=None, category=None, keyword=None, hours=720, limit=20) -> list[dict]:
    """查历史工单。"""
    sql = ("SELECT * FROM tickets WHERE "
           "timestamp >= datetime('now', ?) ")
    args: list = [f"-{int(hours)} hours"]
    if customer_id:
        sql += " AND customer_id = ?"
        args.append(customer_id)
    if category:
        sql += " AND category = ?"
        args.append(category)
    if keyword:
        sql += " AND summary LIKE ?"
        args.append(f"%{keyword}%")
    sql += " ORDER BY timestamp DESC LIMIT ?"
    args.append(limit)
    with db_cursor() as cur:
        cur.execute(sql, args)
        return [dict(r) for r in cur.fetchall()]


def create_ticket(customer_id: str, category: str, summary: str,
                  channel: str = "remote", handler: str = "AI助手") -> dict:
    """新建工单（Ticket Agent 的副作用写操作）。"""
    import random
    from datetime import datetime
    now = datetime.now()
    ticket_no = f"T{now.strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO tickets (ticket_no, timestamp, customer_id, customer_name, channel, "
            "category, status, summary, handler) VALUES (?,?,?,?,?,?,?,?,?)",
            (ticket_no, now.strftime("%Y-%m-%d %H:%M:%S"),
             customer_id, "", channel, category, "已创建", summary, handler))
    return {"ticket_no": ticket_no, "category": category, "summary": summary, "status": "已创建"}
