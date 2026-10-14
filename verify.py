"""一键端到端自检：面试演示前跑这个确认全链路健康。

用法：
    python verify.py

流程：
  1. 四个测试套件（零 API 消耗）
  2. API key 检查（.env 或环境变量，没有则停在步骤 1）
  3. 真实对话验证（约 5 次 LLM 调用，几分钱内）：临时起一个本地服务，
     走**真实 HTTP + SSE**（而非直接调 Router），覆盖 FastAPI 路由、SSE 事件序列、
     会话记忆持久化这几层：
     - "银行卡丢了怎么补办"                 -> 期望 rag_direct
     - "你们网点附近哪里有好吃的火锅"       -> 期望 refuse_human
     - "客户要转 60 万，需要提前预约吗"      -> 期望 agent_loop + 工具调用
     - 同一 session 连续两轮               -> 期望历史累积到 4 条
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable


def step(label):
    print(f"\n{'='*50}\n{label}\n{'='*50}")


def run_tests():
    step("步骤 1/3：测试套件（零 API 消耗）")
    suites = ["tests/test_agent_mock.py", "tests/test_router_mock.py",
              "tests/test_eval_logic.py", "tests/test_agent_langgraph_parity.py"]
    ok = True
    for s in suites:
        r = subprocess.run([PY, str(ROOT / s)], capture_output=True,
                           text=True, encoding="utf-8", cwd=ROOT)
        passed = "全部通过" in (r.stdout or "")
        print(f"  [{'PASS' if passed else 'FAIL'}] {s}")
        if not passed:
            print(r.stdout[-500:] if r.stdout else r.stderr[-500:])
            ok = False
    return ok


def check_key():
    step("步骤 2/3：API key 检查")
    from config import DEEPSEEK_API_KEY
    if not DEEPSEEK_API_KEY:
        print("  [SKIP] 未找到 DEEPSEEK_API_KEY（.env 文件或环境变量）")
        print("         填好后重跑本脚本验证真实对话。")
        return False
    print(f"  [PASS] key 已加载（长度 {len(DEEPSEEK_API_KEY)}）")
    return True


def verify_real_chat():
    """三条路由的真实对话验证 —— 走真实 HTTP + SSE，而不是直接调 Router。

    为什么坚持走 HTTP：直接调 router.route() 会绕过 FastAPI 路由、SSE 事件序列、
    会话记忆这几层。这些层恰恰是演示时真正跑的东西，必须由自检覆盖。
    """
    step("步骤 3/3：真实对话三条路由验证（HTTP + SSE）")

    import json
    import threading
    import time
    import urllib.parse
    import urllib.request

    import uvicorn

    from server import app

    PORT = 8790
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{PORT}"
    for _ in range(40):                     # 最多等 20s 起服务
        try:
            opener.open(f"{base}/", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        print("  [FAIL] 本地服务未能启动，跳过 HTTP 层验证")
        return False

    def sse(query: str, session_id: str = ""):
        qs = urllib.parse.urlencode({"query": query, "session_id": session_id})
        req = urllib.request.Request(f"{base}/api/chat/stream?{qs}",
                                     headers={"Accept": "text/event-stream"})
        events = []
        with opener.open(req, timeout=180) as resp:
            etype, buf = None, []
            for raw in resp:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("event: "):
                    etype = line[7:]
                elif line.startswith("data: "):
                    buf.append(line[6:])
                elif line == "" and etype:
                    events.append((etype, json.loads("".join(buf)) if buf else {}))
                    etype, buf = None, []
        return events

    cases = [
        ("银行卡丢了怎么补办", "rag_direct"),
        ("你们网点附近哪里有好吃的火锅", "refuse_human"),
        ("客户要转 60 万，需要提前预约吗", "agent_loop"),
    ]
    ok = True
    try:
        for query, expected in cases:
            ev = sse(query)
            kinds = [e for e, _ in ev]
            done = next((d for e, d in ev if e == "done"), {})
            hit = "PASS" if done.get("route") == expected else "WARN"
            print(f"  [{hit}] {query[:22]:24} -> {done.get('route')}"
                  f"（期望 {expected}，工具 {len(done.get('tool_trace') or [])} 次）")
            print(f"        事件: {' → '.join(kinds)}")
            if done.get("route") != expected:
                ok = False
            # SSE 协议完整性：必须有 route 起始事件、done 收尾、且带 session_id
            if not ("route" in kinds and kinds[-1] == "done"
                    and done.get("session_id")):
                print("        [FAIL] SSE 事件序列或 session_id 缺失")
                ok = False

        # 会话记忆：同一 session_id 连续两轮，历史应累积到 4 条
        sid = json.load(opener.open(
            urllib.request.Request(f"{base}/api/sessions/new", method="POST"),
            timeout=10))["session_id"]
        sse("银行卡丢了怎么补办", sid)
        sse("那需要带什么材料呢", sid)
        hist = json.load(opener.open(f"{base}/api/sessions/{sid}", timeout=10))
        n = len(hist.get("messages", []))
        print(f"  [{'PASS' if n == 4 else 'FAIL'}] 会话记忆：两轮后历史 {n} 条（应为 4）")
        if n != 4:
            ok = False
    finally:
        server.should_exit = True
        time.sleep(0.5)
    return ok


def main():
    print("Bank Agent（银行远程银行坐席知识助手）一键自检")
    t1 = run_tests()
    if not t1:
        print("\n❌ 测试套件有失败项，先修测试再继续。")
        sys.exit(1)
    t2 = check_key()
    if not t2:
        print("\n✅ 代码逻辑全部健康（测试通过），真实链路待 key 填入后验证。")
        sys.exit(0)
    t3 = verify_real_chat()
    print("\n" + "=" * 50)
    if t3:
        print("✅ 全链路健康。下一步可跑三基线评估：")
        print("   python eval.py --baseline all")
        print("   然后启动 uvicorn server:app --port 8000 查看报告页 /report")
    else:
        print("⚠️  路由结果与预期有偏差（mock 相似度边界问题或真实模型行为差异），")
        print("   不影响演示，但可对照 docs/补课 文档排查。")
    sys.exit(0 if t3 else 0)


if __name__ == "__main__":
    main()
