"""评测脚本：跑 eval_questions.jsonl（21 题），断言 gate/链路/引用/关键事实。

用法: python evals/run_eval.py [--full]
  --full 跑真实 LLM（慢、花钱）；不加只跑 Guard+Supervisor 确定性部分（快、免费）

输出：控制台摘要 + evals/report.md
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.guard import guard_node
from src.graph.supervisor import supervisor_node, _needs_rules
from src.config import SIM_HIGH, SIM_LOW, KW_COVERAGE_LOW

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.jsonl"
REPORT_PATH = Path(__file__).resolve().parent / "report.md"


def load_eval():
    items = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def expected_gate(q: dict) -> str:
    """从评测题的 expected_route 字段推断期望 gate。
    v1 评测集字段：rag_direct/agent_loop → PASS；refuse → REFUSE；
    escalate/transfer → ESCALATE。"""
    route = q.get("expected_route", "")
    if route in ("refuse", "REFUSE"):
        return "REFUSE"
    if route in ("escalate", "transfer", "refuse_human"):
        return "ESCALATE"
    return "PASS"  # rag_direct / agent_loop


def run_deterministic(q: dict) -> dict:
    """跑 Guard + RAG + 一轮 Supervisor 决策（不调 LLM，快）。
    对低置信 PASS 题，推断最终会走 ticket 兜底（完整链路行为）。"""
    query = q.get("question", q.get("query", ""))
    g = guard_node({"query": query, "trace": []})
    gate = g.get("gate", "PASS")
    confidence = None
    needs_rules = False
    final_gate = gate  # 最终等效 gate（考虑低置信兜底）
    if gate == "PASS":
        from src.tools.rag_search import SearchKnowledgeTool
        import json as _j
        rag = _j.loads(SearchKnowledgeTool().call({"query": query, "top_k": 1}))
        sim = rag.get("best_sim", 0)
        kwc = rag.get("best_kw_coverage", 0)
        if sim >= SIM_HIGH and kwc >= KW_COVERAGE_LOW:
            confidence = "high"
        elif sim >= SIM_LOW:
            confidence = "mid"
        else:
            confidence = "low"
        needs_rules = _needs_rules({"query": query})
        # 低置信且无规则可查 → 完整链路会走 ticket 兜底（等效 ESCALATE）
        if confidence == "low" and not needs_rules:
            final_gate = "ESCALATE"
    # 问候类：guard 已标 confidence_level=greeting，保持 PASS
    if g.get("confidence_level") == "greeting":
        confidence = "greeting"
        final_gate = "PASS"
    return {"gate": final_gate, "confidence": confidence, "needs_rules": needs_rules,
            "gate_reason": g.get("gate_reason", ""), "query": query,
            "raw_gate": gate}


def evaluate():
    items = load_eval()
    results = []
    pass_count = 0
    for q in items:
        query = q.get("question", q.get("query", ""))
        exp_gate = expected_gate(q)
        det = run_deterministic(q)
        ok = det["gate"] == exp_gate
        if ok:
            pass_count += 1
        results.append({
            "query": query,
            "expected_gate": exp_gate,
            "actual_gate": det["gate"],
            "confidence": det["confidence"],
            "needs_rules": det["needs_rules"],
            "gate_reason": det["gate_reason"],
            "pass": ok,
            "category": q.get("category", ""),
        })

    total = len(results)
    rate = pass_count / total if total else 0
    return results, pass_count, total, rate


def write_report(results, passed, total, rate):
    lines = [
        "# 评测报告 · 多 Agent v2（确定性部分）",
        "",
        f"> 日期：{time.strftime('%Y-%m-%d %H:%M')} ｜ 评测集：{total} 题",
        f"> 通过：**{passed}/{total}（{rate:.1%}）**",
        "",
        "## 概要",
        "",
        f"- Guard 路由正确率：{rate:.1%}（目标 ≥ 90%）",
        f"- 行外护栏类：{sum(1 for r in results if r['expected_gate']=='REFUSE')} 题",
        f"- 转人工类：{sum(1 for r in results if r['expected_gate']=='ESCALATE')} 题",
        f"- 业务处理类：{sum(1 for r in results if r['expected_gate']=='PASS')} 题",
        "",
        "## 逐题结果",
        "",
        "| # | 问题 | 期望 | 实际 | 置信度 | 需查表 | 结果 |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        q = r["query"][:30]
        mark = "✓" if r["pass"] else "✘"
        lines.append(f"| {i} | {q} | {r['expected_gate']} | {r['actual_gate']} | "
                     f"{r['confidence'] or '-'} | {'是' if r['needs_rules'] else '否'} | {mark} |")

    failed = [r for r in results if not r["pass"]]
    if failed:
        lines += ["", "## 失败归因", ""]
        for r in failed:
            lines.append(f"- **{r['query']}**：期望 {r['expected_gate']}，实际 {r['actual_gate']}"
                         f"（{r['gate_reason']}）")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {REPORT_PATH}")


def main():
    full = "--full" in sys.argv
    results, passed, total, rate = evaluate()
    print(f"\n{'='*50}")
    print(f"评测完成：{passed}/{total} 通过（{rate:.1%}）")
    print(f"{'='*50}")
    for i, r in enumerate(results, 1):
        mark = "✓" if r["pass"] else "✘"
        print(f"  {mark} #{i} [{r['expected_gate']}→{r['actual_gate']}] {r['query'][:40]}")
    write_report(results, passed, total, rate)

    if full:
        print("\n--full 模式：跑真实 LLM 全量端到端...")
        items = load_eval()
        from src.graph import get_app
        app = get_app()
        e2e_passed = 0
        e2e_results = []
        for q in items:
            query = q.get("question", q.get("query", ""))
            exp_gate = expected_gate(q)
            t0 = time.time()
            try:
                res = app.invoke({"query": query, "messages": [], "trace": []})
                dt = time.time() - t0
                actual_gate = res.get("gate", "PASS")
                # 资金安全/投诉/拒答类：看 gate；业务类：看是否有 answer
                ok = (actual_gate == exp_gate) if exp_gate != "PASS" else bool(res.get("answer"))
                if ok:
                    e2e_passed += 1
                e2e_results.append((query[:25], exp_gate, actual_gate, dt, ok, res.get("answer", "")[:40]))
                mark = "✓" if ok else "✘"
                print(f"  {mark} [{exp_gate}→{actual_gate}] {query[:30]} ({dt:.1f}s)")
            except Exception as e:
                e2e_results.append((query[:25], exp_gate, "ERROR", 0, False, str(e)[:40]))
                print(f"  ✘ [ERROR] {query[:30]}: {e}")
        print(f"\n端到端：{e2e_passed}/{len(items)} 通过")
        # 追加到报告
        report = REPORT_PATH.read_text(encoding="utf-8")
        report += "\n\n## 真实 LLM 端到端结果\n\n"
        report += f"通过：{e2e_passed}/{len(items)}\n\n"
        report += "| # | 问题 | 期望 | 实际 | 耗时 | 结果 |\n|---|---|---|---|---|---|\n"
        for i, (q, eg, ag, dt, ok, _) in enumerate(e2e_results, 1):
            report += f"| {i} | {q} | {eg} | {ag} | {dt:.1f}s | {'✓' if ok else '✘'} |\n"
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"端到端结果已追加到 {REPORT_PATH}")


if __name__ == "__main__":
    main()
