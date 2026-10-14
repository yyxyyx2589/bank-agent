"""数据库初始化 + 数据生成脚本（银行远程银行坐席知识助手版）。

建 SQLite schema 并生成银行坐席场景的业务数据，替代 ops-agent 的
运维日志 / 监控指标。

表结构：
  tickets        业务工单（客户历史来电与处理记录）
  products       产品费率（存款/贷款/信用卡/理财/基金）
  business_rules 业务规则（可办理条件、限额、所需材料）
  sessions       会话记录（结构沿用）

数据说明：全部为脚本生成的模拟数据，不涉及任何真实机构与客户信息；
         产品费率参照公开信息设定，仅用于技术演示。

运行：python db/init_db.py
"""
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "bank.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_no TEXT NOT NULL,          -- 工单号，如 GD2026100700031
    timestamp TEXT NOT NULL,          -- ISO8601: 2026-10-07 09:12:01
    customer_id TEXT NOT NULL,        -- 客户号 C2026031800237
    customer_name TEXT NOT NULL,
    channel TEXT NOT NULL,            -- 电话 / 网点 / 手机银行 / 微信银行
    category TEXT NOT NULL,           -- 挂失补卡 / 转账限额 / 信用卡 / 理财咨询 / 账户冻结 / 对公业务 / 投诉建议
    status TEXT NOT NULL,             -- 已解决 / 处理中 / 转人工 / 已转网点
    summary TEXT NOT NULL,            -- 工单摘要
    handler TEXT NOT NULL             -- 受理人
);
CREATE INDEX IF NOT EXISTS idx_tickets_ts ON tickets(timestamp);
CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_code TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL,           -- 存款 / 贷款 / 信用卡 / 理财 / 基金
    term TEXT,                        -- 期限描述
    rate REAL,                        -- 年化利率或费率(%)
    rate_desc TEXT NOT NULL,          -- 人话描述，直接可读给客户
    min_amount REAL,                  -- 起存/起购金额(元)
    risk_level TEXT,                  -- R1-R5，仅理财有
    effective_date TEXT NOT NULL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);

CREATE TABLE IF NOT EXISTS business_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_code TEXT NOT NULL,
    scenario TEXT NOT NULL,           -- 场景，如"银行卡挂失""大额转账"
    condition TEXT NOT NULL,          -- 触发条件
    requirement TEXT NOT NULL,        -- 需要什么材料/动作
    limit_amount REAL,                -- 涉及金额上限(元)，无则 NULL
    allow_agent INTEGER NOT NULL,     -- 是否允许他人代办 1可 / 0不可
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_rules_scenario ON business_rules(scenario);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,               -- user / assistant
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_sid ON sessions(session_id);
"""


def init_schema(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def gen_tickets():
    """生成业务工单。

    围绕一个可复现的工单链故事（客户 C2026031800237 张伟：
    9月28日先咨询转账限额 → 10月3日卡片丢失挂失 → 10月6日账户被风控拦截
    → 10月8日到网点解控），保证 demo 时"查客户历史工单"有因果可讲。
    """
    random.seed(42)
    tickets = []
    base = datetime(2026, 9, 25, 9, 0, 0)

    # 核心工单链（固定，保证 demo 效果）
    story = [
        ("2026-09-28 10:12:33", "C2026031800237", "张伟", "电话", "转账限额",
         "已解决", "客户咨询手机银行单笔转账限额，告知短信验证方式单笔上限5万元，如需提高可到网点办理U盾。", "坐席A03"),
        ("2026-10-03 15:47:02", "C2026031800237", "张伟", "电话", "挂失补卡",
         "已解决", "客户银行卡丢失，来电办理口头挂失，有效期7天。告知需本人持身份证到网点办理正式挂失与补卡。", "坐席A07"),
        ("2026-10-06 08:31:19", "C2026031800237", "张伟", "手机银行", "账户冻结",
         "转人工", "客户反馈手机银行无法转账，提示交易受限。系统显示账户命中反欺诈风控规则，已引导客户到就近网点核实身份。", "坐席B02"),
        ("2026-10-08 11:05:44", "C2026031800237", "张伟", "网点", "账户冻结",
         "已解决", "客户到网点核实身份及资金用途，确认为误判（客户为新购房首付款转账），已解除非柜面交易限制。", "柜员C11"),
    ]

    # 注意：故事客户 C2026031800237 不进随机池，避免随机工单污染工单链叙事
    # （否则 demo 时讲"张伟：咨询限额→挂失→被冻结→解控"会被无关工单打断）
    customers = [
        ("C2026012600912", "李静"),
        ("C2026041100558", "王强"), ("C2026020300147", "陈慧敏"),
        ("C2026052000881", "刘洋"), ("C2026071500306", "赵鹏"),
        ("C2026080900724", "孙丽娜"), ("C2026090100415", "周建国"),
    ]

    categories = {
        "挂失补卡": ["客户银行卡丢失，来电咨询挂失流程。", "客户咨询同号换卡所需时间。",
                     "客户卡片消磁，咨询补办所需材料。"],
        "转账限额": ["客户咨询手机银行转账限额如何提高。", "客户反馈转账超限被拒，咨询解决方案。",
                     "客户咨询大额转账是否需要预约。"],
        "信用卡": ["客户咨询信用卡分期手续费如何计算。", "客户咨询账单分期期数与费率。",
                   "客户咨询信用卡逾期对征信的影响。", "客户咨询信用卡挂失补卡时效。"],
        "理财咨询": ["客户咨询理财产品风险等级划分。", "客户咨询首次购买理财是否需要风险测评。",
                     "客户咨询 R2 与 R3 理财产品的区别。"],
        "账户冻结": ["客户反馈账户无法转账，咨询原因。", "客户咨询账户被风控拦截如何处理。",
                     "客户咨询涉案冻结如何申请解冻。"],
        "对公业务": ["企业客户咨询开立基本户所需材料。", "企业客户咨询网银转账审批流程设置。",
                     "企业客户咨询代发工资到账时间。"],
        "投诉建议": ["客户投诉网点排队时间过长。", "客户建议手机银行增加人脸登录功能。"],
    }

    handlers = ["坐席A03", "坐席A07", "坐席B02", "坐席B05", "坐席C01", "柜员C11"]
    channels = ["电话", "手机银行", "网点", "微信银行"]

    cat_list = list(categories.keys())
    for i in range(56):
        # 时间落在工作时段（8:00-19:59），日期在 20 天窗口内随机
        # —— 早期版本用 base + 随机整点小时，生成的时间戳全是 02:00:00 / 19:00:00
        # 这种整点，一眼假；加上随机分秒后才像真实工单记录。
        ts = (base + timedelta(days=random.randint(0, 20))
              + timedelta(hours=random.randint(8, 19),
                          minutes=random.randint(0, 59),
                          seconds=random.randint(0, 59)))
        cust_id, cust_name = random.choice(customers)
        cat = cat_list[i % len(cat_list)]   # 轮询类别，保证 7 类分布均衡
        summary = random.choice(categories[cat])
        status = random.choices(
            ["已解决", "处理中", "转人工", "已转网点"], weights=[6, 2, 1, 1])[0]
        if cat in ("转账限额", "账户冻结") and random.random() < 0.5:
            summary += f" 涉及金额约 {random.choice([3, 5, 18, 30, 50, 80])} 万元。"
        tickets.append((
            ts.strftime("%Y-%m-%d %H:%M:%S"), cust_id, cust_name,
            random.choice(channels), cat, status, summary, random.choice(handlers)))

    # 合并并按时间排序，统一生成工单号
    all_rows = [(datetime.strptime(t[0], "%Y-%m-%d %H:%M:%S"),) + t[1:]
                for t in story + tickets]
    all_rows.sort(key=lambda r: r[0])

    out = []
    for i, r in enumerate(all_rows, start=1):
        ts = r[0].strftime("%Y-%m-%d %H:%M:%S")
        ticket_no = f"GD{ts[0:4]}{ts[5:7]}{ts[8:10]}{i:05d}"
        out.append((ticket_no, ts, r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
    return out


def gen_products():
    """产品费率表。费率参照公开信息设定，仅用于演示。"""
    eff = "2026-09-01"
    return [
        # 存款
        ("CK001", "活期存款", "存款", "活期", 0.20, "年利率 0.20%，随存随取", 1, None, eff, "按季结息"),
        ("CK003", "整存整取三个月", "存款", "三个月", 1.10, "年利率 1.10%，起存50元", 50, None, eff, "到期自动转存"),
        ("CK006", "整存整取半年", "存款", "半年", 1.30, "年利率 1.30%，起存50元", 50, None, eff, "到期自动转存"),
        ("CK012", "整存整取一年", "存款", "一年", 1.45, "年利率 1.45%，起存50元", 50, None, eff, "到期自动转存"),
        ("CK024", "整存整取二年", "存款", "二年", 1.65, "年利率 1.65%，起存50元", 50, None, eff, ""),
        ("CK036", "整存整取三年", "存款", "三年", 1.95, "年利率 1.95%，起存50元", 50, None, eff, ""),
        ("CK060", "整存整取五年", "存款", "五年", 2.00, "年利率 2.00%，起存50元", 50, None, eff, ""),
        ("CK100", "个人大额存单三年期", "存款", "三年", 2.35, "年利率 2.35%，起存20万元，可转让", 200000, None, eff, "属存款保险保障范围"),
        ("CK101", "个人大额存单一年期", "存款", "一年", 1.75, "年利率 1.75%，起存20万元，可转让", 200000, None, eff, "属存款保险保障范围"),
        # 贷款
        ("DK001", "LPR一年期", "贷款", "一年期报价利率", 3.00, "1年期 LPR 为 3.00%", None, None, "2026-09-22", "每月20日公布"),
        ("DK002", "LPR五年期以上", "贷款", "五年期以上报价利率", 3.50, "5年期以上 LPR 为 3.50%", None, None, "2026-09-22", "每月20日公布"),
        ("DK010", "个人住房贷款（首套）", "贷款", "最长30年", 3.30, "首套不低于 LPR-20BP，即 3.30%", None, None, eff, "以LPR为定价基准浮动"),
        ("DK011", "个人住房贷款（二套）", "贷款", "最长30年", 3.70, "二套不低于 LPR+20BP，即 3.70%", None, None, eff, "以LPR为定价基准浮动"),
        ("DK020", "住房公积金贷款（5年以上）", "贷款", "最长30年", 2.85, "5年以上公积金贷款利率 2.85%", None, None, eff, "人民银行统一规定"),
        ("DK030", "个人消费贷款（信用方式）", "贷款", "1-5年", 4.20, "年化利率 4.20% 起，按征信差异定价", 10000, None, eff, "不得用于购房与投资"),
        # 信用卡
        ("XY001", "信用卡账单分期 3 期", "信用卡", "3期", 1.80, "3期总手续费率 1.80%", None, None, eff, "折算年化约 10.8%"),
        ("XY002", "信用卡账单分期 6 期", "信用卡", "6期", 3.60, "6期总手续费率 3.60%", None, None, eff, "折算年化约 12.2%"),
        ("XY003", "信用卡账单分期 12 期", "信用卡", "12期", 7.20, "12期总手续费率 7.20%", None, None, eff, "折算年化约 13.3%"),
        ("XY004", "信用卡账单分期 24 期", "信用卡", "24期", 14.40, "24期总手续费率 14.40%", None, None, eff, "折算年化约 13.6%"),
        ("XY010", "信用卡取现", "信用卡", "按日计息", 18.25, "日息万分之五，折算年化 18.25%", None, None, eff, "另收取现手续费"),
        ("XY020", "信用卡挂失费", "信用卡", "一次性", 50.00, "挂失手续费 50 元", None, None, eff, "部分高端卡减免"),
        # 理财
        ("LC001", "稳健添利 90 天", "理财", "90天", 2.60, "业绩比较基准 2.60%", 10000, "R2", eff, "非保本浮动收益"),
        ("LC002", "稳健添利 180 天", "理财", "180天", 2.95, "业绩比较基准 2.95%", 10000, "R2", eff, "非保本浮动收益"),
        ("LC003", "固收增强一年期", "理财", "一年", 3.40, "业绩比较基准 3.40%", 10000, "R2", eff, "非保本浮动收益"),
        ("LC010", "均衡配置混合类", "理财", "一年", 4.20, "业绩比较基准 4.20%", 10000, "R3", eff, "可能发生本金损失"),
        ("LC020", "权益优选进取型", "理财", "两年", 6.50, "业绩比较基准 6.50%", 50000, "R4", eff, "本金损失风险较高"),
    ]


def gen_rules():
    """业务规则表。allow_agent: 1 可代办 / 0 必须本人。"""
    return [
        ("R001", "银行卡挂失", "持卡人银行卡丢失或被盗",
         "临时挂失可通过客服热线或手机银行办理；正式挂失必须本人持有效身份证件到营业网点办理",
         None, 0, "挂失后原卡立即失效"),
        ("R002", "银行卡补卡", "挂失后需要补办新卡",
         "本人持有效身份证件到营业网点办理，同号换卡约需7-15个工作日，异号卡可当场领取",
         None, 0, ""),
        ("R003", "个人开户", "申请开立Ⅰ类银行账户",
         "必须本人持有效身份证件原件到营业网点办理，需提供本人实名手机号",
         None, 0, "不得由他人代办"),
        ("R004", "未成年人开户", "不满16周岁客户申请开户",
         "需监护人陪同并提供监护关系证明（户口簿或出生医学证明）",
         None, 1, "16周岁以上以自己劳动收入为主要生活来源者可独立办理"),
        ("R005", "大额转账预约", "单笔转账金额超过50万元",
         "需提前一个工作日预约，并说明资金用途",
         500000, 0, "银行有权核实交易背景"),
        ("R006", "大额现金存取登记", "单日累计现金存取超过5万元",
         "需出示本人有效身份证件，银行按规定上报大额交易报告",
         50000, 0, ""),
        ("R007", "提高转账限额", "客户申请提升手机银行转账限额",
         "需本人持身份证和银行卡到网点办理U盾或数字证书，可提升至单笔100万元",
         None, 0, ""),
        ("R008", "账户解控", "账户被反欺诈风控拦截",
         "本人持有效身份证件和银行卡到营业网点核实身份及资金用途",
         None, 0, "确认为误判的当场解控"),
        ("R009", "对公开户", "企业申请开立基本存款账户",
         "需提供营业执照正副本、法定代表人身份证件、公章与预留印鉴、公司章程，法定代表人需到场面签",
         None, 0, "基本户一个企业只能开一个"),
        ("R010", "企业信息变更", "企业法定代表人、注册地址等发生变更",
         "应在变更后30日内到开户行办理账户信息变更",
         None, 0, "逾期未变更银行可中止账户非柜面业务"),
        ("R011", "理财首次购买", "客户首次购买理财产品",
         "必须先完成风险承受能力评估（风险测评），测评有效期1年",
         None, 0, "风险承受等级不得低于产品风险等级"),
        ("R012", "信用卡分期申请", "持卡人申请账单分期",
         "可通过手机银行、客服热线或网银自助申请，分期一经申请多数不支持撤销",
         None, 1, "提前还款仍需支付剩余期数手续费"),
        ("R013", "定期存款提前支取", "定期存款未到期需取出",
         "本人持身份证和存款凭证到网点办理，支持一次部分提前支取",
         None, 0, "提前支取部分按活期利率计息"),
        ("R014", "征信异议申请", "客户认为信用报告信息有误",
         "提交本人身份证件复印件、征信异议申请表及证明材料",
         None, 0, "受理后20日内核查并书面答复"),
        ("R015", "继承查询与支取", "存款人去世后家属办理账户查询或支取",
         "需提供死亡证明、继承权证明或公证书、继承人身份证件",
         None, 1, "涉及金额较大时需办理公证"),
    ]


def main():
    conn = sqlite3.connect(str(DB_PATH))
    # 用 DROP TABLE 重建，而不是删除 db 文件后重建：
    # 本机环境对 os.unlink / unlink() 有 safe-delete 拦截（尝试送回收站），
    # 在脚本里会抛 OSError；DROP + CREATE 语义等价且不依赖文件系统删除权限。
    for t in ("tickets", "products", "business_rules", "sessions"):
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.commit()
    init_schema(conn)

    tickets = gen_tickets()
    conn.executemany(
        "INSERT INTO tickets (ticket_no, timestamp, customer_id, customer_name, "
        "channel, category, status, summary, handler) VALUES (?,?,?,?,?,?,?,?,?)",
        tickets)
    print(f"业务工单导入: {len(tickets)} 条")

    products = gen_products()
    conn.executemany(
        "INSERT INTO products (product_code, name, category, term, rate, rate_desc, "
        "min_amount, risk_level, effective_date, note) VALUES (?,?,?,?,?,?,?,?,?,?)",
        products)
    print(f"产品费率导入: {len(products)} 条")

    rules = gen_rules()
    conn.executemany(
        "INSERT INTO business_rules (rule_code, scenario, condition, requirement, "
        "limit_amount, allow_agent, note) VALUES (?,?,?,?,?,?,?)",
        rules)
    print(f"业务规则导入: {len(rules)} 条")

    conn.commit()

    print("\n=== 数据校验 ===")
    for t in ("tickets", "products", "business_rules"):
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t:16} 总数: {n}")

    print("\n工单类别分布:")
    for row in conn.execute(
            "SELECT category, COUNT(*) c FROM tickets GROUP BY category ORDER BY c DESC"):
        print(f"  {row[0]:10} {row[1]}")

    print("\n客户 C2026031800237 的工单链（demo 用）:")
    for row in conn.execute(
            "SELECT timestamp, category, status, summary FROM tickets "
            "WHERE customer_id='C2026031800237' ORDER BY timestamp"):
        print(f"  {row[0]}  [{row[1]}/{row[2]}] {row[3][:52]}")

    print("\n存款利率表（前5条）:")
    for row in conn.execute(
            "SELECT name, rate, rate_desc FROM products WHERE category='存款' LIMIT 5"):
        print(f"  {row[0]:22} {row[1]:6}% {row[2]}")

    conn.close()
    print(f"\n数据库初始化完成: {DB_PATH}")


if __name__ == "__main__":
    main()
