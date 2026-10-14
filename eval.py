"""第 4 周评估脚本：三基线对比 + 六项指标。

三基线：
  baseline_rag        纯 RAG（只用 search_knowledge 直答，不进 Agent 循环）
  baseline_rag_lora   RAG + LoRA（本地 Qwen2.5-1.5B + 银行域 LoRA，私有化小模型方案）
  baseline_agent      Agent（本项目：Router + Agent 循环 + 工具）
  另有 baseline_rag_base（消融档：纯基座不挂 adapter，见文末）

六项指标：
  1. 端到端解决率     三基线：答案是否命中标注的期望要点
  2. 路由准确率       三基线：实际走的分支是否与标注期望一致（纯 RAG 无路由，一律直答）
  3. 工具选择准确率   Agent 路径：模型选的工具和标注的期望工具是否一致
  4. 参数生成准确率   Agent 路径：工具参数 JSON 和标注参数是否匹配
  5. 平均轮次         Agent 路径：达到答案用了几轮工具调用
  6. 平均延迟         三基线：单次处理耗时（ms）

运行（需设 DEEPSEEK_API_KEY）：
  python eval.py --baseline agent --data data/eval_questions.jsonl
  python eval.py --baseline all   # 跑全部三基线，生成 reports/eval_report.json
  python eval.py --baseline all --repeat 3   # 3 次重复，报告稳定性；默认预热排除冷启动

消融实验（验证 LoRA 到底带来了什么）：
  python eval.py --baseline rag_base --out reports/ablation_base.json
  # 同链路、同提示，唯一差别是不挂 adapter。「我训了 LoRA」与
  # 「LoRA 带来了 X 个点」是两件事，后者才经得起追问。
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# 让 data/ 和 reports/ 存在
DATA_FILE = Path(__file__).parent / "data" / "eval_questions.jsonl"
REPORTS_DIR = Path(__file__).parent / "reports"


def load_eval_data(path: Path) -> list:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_baseline_agent(question: str, router) -> dict:
    """Agent 基线：走完整 Router + Agent 循环。"""
    t0 = time.time()
    result = router.route(question)
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "answer": result.answer,
        "route": result.route,
        "tool_trace": result.tool_trace,
        "sim": result.sim,
        "kw_coverage": result.kw_coverage,
        "latency_ms": latency_ms,
        "turns": len(result.tool_trace),
    }


def run_baseline_rag(question: str, rag_tool, llm) -> dict:
    """纯 RAG 基线：只检索 + 直答，不进 Agent 循环（模拟原 RAG 项目）。"""
    t0 = time.time()
    raw = rag_tool.call({"query": question, "top_k": 1})
    rag = json.loads(raw)
    kb = rag["results"][0] if rag["results"] else None

    if kb:
        prompt = f"基于知识片段回答：\n{kb['text']}\n\n问题：{question}"
        resp = llm.chat([
            {"role": "system", "content": "你是银行坐席知识助手。"},
            {"role": "user", "content": prompt},
        ])
        answer = resp.choices[0].message.content or ""
    else:
        answer = "未检索到相关知识。"

    return {
        "answer": answer,
        "sim": rag.get("best_sim", 0),
        "latency_ms": int((time.time() - t0) * 1000),
        "turns": 0,
        "kb_id": kb["id"] if kb else None,
    }


def run_baseline_rag_lora(question: str, rag_tool, lora_llm) -> dict:
    """RAG + LoRA 基线：检索链路与纯 RAG 完全一致，只把作答模型换成
    本地 Qwen2.5-1.5B + 银行域 LoRA（training/out/bank_lora）。

    为什么它必须和纯 RAG 用同一条链路：这样两条基线之间**唯一的变量**
    就是「谁在作答」——云端大模型 vs 私有化小模型。指标差异才能归因到模型，
    而不是被链路上的其他改动混淆。

    为什么这个基线值得存在：银行普遍要求私有化部署（客户数据不出内网），
    「小模型 + 域微调」是真实存在的成本方案。它和 Agent 基线的差距，
    恰好能量化「编排层到底值多少」——这一段不是凑数，是成本-效果曲线的中间点。
    """
    return run_baseline_rag(question, rag_tool, lora_llm)


def _norm(s: str) -> str:
    """归一化：去掉所有空白（含全角空格）。

    为什么必须做：LLM 输出的数字/量词常带空格——期望点写「20万」，
    答案里是「20 万元」；期望「16周岁」，答案是「16 周岁」。
    子串匹配会被一个空格击穿，导致明明答对却判 0 分。
    """
    return re.sub(r"\s+", "", s or "")


def score_answer(answer: str, expected_points: list) -> float:
    """答案命中期望要点的比例（0~1）。空白归一化后做子串匹配。"""
    if not expected_points:
        return 1.0
    a = _norm(answer)
    hit = sum(1 for p in expected_points if _norm(p) in a)
    return round(hit / len(expected_points), 3)


def _param_match(expected, actual) -> bool:
    """参数是否匹配。字符串走双向包含，数字走等值。

    为什么不做严格等值：工具参数（场景、关键词）由 LLM 从自然语言里抽取，
    口径天然不固定——标注写「挂失」，模型可能传「银行卡挂失」；
    标注写「开户」，模型可能传「对公开户」。二者语义一致，应判为命中。
    这与工具内部 scenario 的 LIKE 模糊匹配保持一致（见 db/db.py check_rule）。
    """
    if expected == actual:
        return True
    if isinstance(expected, str) and isinstance(actual, str):
        e, a = expected.strip(), actual.strip()
        return bool(e) and bool(a) and (e in a or a in e)
    return False


def score_tool_accuracy(trace: list, expected_tools: list) -> dict:
    """工具选择准确率 + 参数准确率。"""
    if not expected_tools:
        return {"tool_accuracy": 1.0, "param_accuracy": 1.0}
    actual_names = [s["name"] for s in trace]
    # 工具选择准确率：期望工具集有多少被实际调用
    expected_set = set(t["name"] for t in expected_tools)
    actual_set = set(actual_names)
    tool_acc = len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0

    # 参数准确率：期望工具的参数是否匹配（场景/关键词走包含匹配，见 _param_match）
    param_hits = 0
    param_total = 0
    for exp in expected_tools:
        for act in trace:
            if act["name"] == exp["name"]:
                param_total += 1
                exp_p = exp.get("params", {})
                act_p = act.get("params", {})
                if all(_param_match(v, act_p.get(k)) for k, v in exp_p.items()):
                    param_hits += 1
                break
        else:
            param_total += 1  # 期望调用但没调
    param_acc = param_hits / param_total if param_total else 1.0

    return {"tool_accuracy": round(tool_acc, 3),
            "param_accuracy": round(param_acc, 3)}


def evaluate(baseline: str, data: list, components: dict, verbose: bool = True) -> list:
    results = []
    for item in data:
        q = item["question"]
        expected_points = item.get("expected_points", [])
        expected_tools = item.get("expected_tools", [])
        expected_route = item.get("expected_route")

        if baseline == "agent":
            out = run_baseline_agent(q, components["router"])
            tool_scores = score_tool_accuracy(out["tool_trace"], expected_tools)
            actual_route = out["route"]
        elif baseline == "rag":
            out = run_baseline_rag(q, components["rag_tool"], components["llm"])
            tool_scores = {"tool_accuracy": 0, "param_accuracy": 0}
            # 纯 RAG 没有路由概念：它一律"直答"，不存在查工具/转人工的路径
            actual_route = "rag_direct"
        elif baseline == "rag_lora":
            out = run_baseline_rag_lora(q, components["rag_tool"], components["lora_llm"])
            tool_scores = {"tool_accuracy": 0, "param_accuracy": 0}
            # 纯 RAG 没有路由概念：它一律"直答"，不存在查工具/转人工的路径
            actual_route = "rag_direct"
        elif baseline == "rag_base":
            # 消融档：同链路、同提示，只是不挂 adapter（纯基座）
            out = run_baseline_rag(q, components["rag_tool"], components["base_llm"])
            tool_scores = {"tool_accuracy": 0, "param_accuracy": 0}
            actual_route = "rag_direct"
        else:
            raise ValueError(f"未知 baseline: {baseline}")

        answer_score = score_answer(out["answer"], expected_points)
        route_score = 1.0 if (expected_route is None or actual_route == expected_route) else 0.0
        results.append({
            "qid": item["qid"],
            "question": q,
            "baseline": baseline,
            "answer": out["answer"],
            "answer_score": answer_score,
            "tool_accuracy": tool_scores["tool_accuracy"],
            "param_accuracy": tool_scores["param_accuracy"],
            "route_accuracy": route_score,
            "route": actual_route,
            "expected_route": expected_route,
            "turns": out.get("turns", 0),
            "latency_ms": out["latency_ms"],
            "sim": out.get("sim", 0),
            "tools_called": [s["name"] for s in (out.get("tool_trace") or [])],
        })
        if verbose:
            print(f"  [{item['qid']}] {baseline:8} score={answer_score} "
                  f"route={actual_route}({expected_route}) "
                  f"turns={out.get('turns', 0)} {out['latency_ms']}ms")
    return results


def aggregate(results: list) -> dict:
    """汇总指标。"""
    n = len(results) or 1
    # 按路由分组的延迟：整体均值会被"大多数题走直答"掩盖，看不出代价花在哪。
    by_route = {}
    for r in results:
        by_route.setdefault(r.get("route", "unknown"), []).append(r["latency_ms"])
    return {
        "count": len(results),
        "answer_score_avg": round(sum(r["answer_score"] for r in results) / n, 3),
        "tool_accuracy_avg": round(sum(r["tool_accuracy"] for r in results) / n, 3),
        "param_accuracy_avg": round(sum(r["param_accuracy"] for r in results) / n, 3),
        "route_accuracy_avg": round(sum(r["route_accuracy"] for r in results) / n, 3),
        "turns_avg": round(sum(r["turns"] for r in results) / n, 2),
        "latency_ms_avg": round(sum(r["latency_ms"] for r in results) / n),
        "latency_by_route": {k: round(sum(v) / len(v))
                             for k, v in sorted(by_route.items())},
    }


def mean_aggregate(runs: list) -> dict:
    """多次运行的指标均值（count 取第一次，因为各次题数相同）。

    按指标量纲分别取整：延迟是毫秒（整数）、轮次保留两位、比率保留三位。
    统一 round(...,3) 会得到「平均延迟 2747.667 ms」这种没意义的精度。
    """
    out = {}
    for k in runs[0]:
        vals = [r[k] for r in runs]
        if k == "count":
            out[k] = vals[0]
        elif isinstance(vals[0], dict):
            # 非标量指标（如按路由分组的延迟）：逐键取均值
            out[k] = {kk: round(sum(v[kk] for v in vals if kk in v)
                                / len([v for v in vals if kk in v]))
                      for kk in vals[0]}
        elif k == "latency_ms_avg":
            out[k] = round(sum(vals) / len(vals))
        elif k == "turns_avg":
            out[k] = round(sum(vals) / len(vals), 2)
        else:
            out[k] = round(sum(vals) / len(vals), 3)
    return out


def stability(runs: list) -> dict:
    """稳定性：每个指标给出 均值 / 最小值 / 最大值。

    为什么需要这个：LLM 有随机性，单跑一次的「100%」可能只是运气好。
    面试官问「这个数字稳不稳」时，能答出极差比只会重复一个数字强得多。
    """
    out = {"runs": len(runs)}
    if len(runs) < 2:
        # 单次运行没有"波动"可言，不编造极差
        return out
    metrics = ["answer_score_avg", "route_accuracy_avg", "tool_accuracy_avg",
               "param_accuracy_avg"]
    for m in metrics:
        vals = [r[m] for r in runs if m in r]
        if vals:
            out[m] = {"mean": round(sum(vals) / len(vals), 3),
                      "min": min(vals), "max": max(vals)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="agent",
                        choices=["agent", "rag", "rag_lora", "rag_base", "all"],
                        help="rag_base = 消融档：同链路同提示，但不挂 adapter（纯基座）")
    parser.add_argument("--data", default=str(DATA_FILE))
    parser.add_argument("--repeat", type=int, default=1,
                        help="重复 N 次并报告稳定性（均值/极差）；N>1 时明细只打第一次")
    parser.add_argument("--no-warmup", action="store_true",
                        help="关闭预热（默认会先跑一次不计入统计的空调用）")
    parser.add_argument("--out", default=None,
                        help="报告输出路径（默认 reports/eval_report.json）。"
                             "消融实验请另指定文件，避免覆盖主报告")
    args = parser.parse_args()

    data = load_eval_data(Path(args.data))
    print(f"加载 {len(data)} 条测试问题"
          + (f"，重复 {args.repeat} 次" if args.repeat > 1 else ""))

    # 组装组件（需要 DEEPSEEK_API_KEY）
    from config import BASE_URL, MODEL, require_api_key
    from llm_client import LLMClient
    from agent import Agent
    from router import Router
    from tools import TOOL_REGISTRY
    from cli import SYSTEM_PROMPT  # cli.py 顶层无副作用，直接 import（比正则抽源码可靠）

    llm = LLMClient(api_key=require_api_key(), base_url=BASE_URL, model=MODEL)
    agent = Agent(llm=llm, tools=list(TOOL_REGISTRY.values()),
                 system_prompt=SYSTEM_PROMPT, max_turns=8)
    router = Router(rag_tool=TOOL_REGISTRY["search_knowledge"], agent=agent)
    components = {
        "llm": llm, "rag_tool": TOOL_REGISTRY["search_knowledge"],
        "router": router, "lora_llm": None, "base_llm": None,
        # 两个本地模型都按需加载，不吃不必要的加载时间
    }

    baselines = (["agent", "rag", "rag_lora"] if args.baseline == "all"
                 else [args.baseline])

    # 中间基线要的是**真的本地模型**：
    #   rag_lora = Qwen2.5-1.5B + 银行域 LoRA（training/out/bank_lora）
    #   rag_base = 同一基座但不挂 adapter —— 消融实验，用来回答
    #              「这个分数是 LoRA 带来的，还是基座本来就会？」
    need_local = [b for b in ("rag_lora", "rag_base") if b in baselines]
    if need_local:
        from training.lora_llm import (LocalLoRAClient, adapter_available,
                                       DEFAULT_ADAPTER)
        if "rag_lora" in need_local and not adapter_available(Path(DEFAULT_ADAPTER)):
            print(f"[x] 缺少银行域 adapter: {DEFAULT_ADAPTER}")
            print("    先跑：python training/build_sft.py && python training/train_lora.py")
            return
        print("[*] 加载本地模型（Qwen2.5-1.5B）…")
        if "rag_lora" in need_local:
            components["lora_llm"] = LocalLoRAClient(DEFAULT_ADAPTER)
        if "rag_base" in need_local:
            components["base_llm"] = LocalLoRAClient(None)

    # 预热：两样东西都要热，否则它们的开销会全部记到第 1 题头上（实测第 1 题 36 秒）——
    #   ① LLM 连接（TLS 握手 / 服务端冷启动）
    #   ② RAG 的 embedding 模型懒加载（实测首次 30114ms，第二次 36ms，差 800 倍）
    # 这不是"美化数字"：把一次性初始化成本算进单题处理耗时，本身就是测量口径错误。
    if not args.no_warmup:
        try:
            llm.chat([{"role": "user", "content": "ping"}])
            components["rag_tool"].call({"query": "银行卡挂失流程", "top_k": 1})
            if components.get("lora_llm"):
                # 本地模型第一次 generate 还要付 CUDA kernel 预热，一并热掉
                components["lora_llm"].chat([{"role": "user", "content": "你好"}])
            if components.get("base_llm"):
                components["base_llm"].chat([{"role": "user", "content": "你好"}])
            print("已预热（LLM 连接 + embedding 模型 + 本地模型），"
                  "首次冷启动开销不计入延迟统计\n")
        except Exception as e:  # 预热失败不该让整个评估挂掉
            print(f"预热失败（忽略，继续评估）: {e}\n")

    if "rag_lora" in baselines:
        print("中间基线 = 本地 Qwen2.5-1.5B + 银行域 LoRA，检索链路与纯 RAG 完全相同，"
              "只把作答模型换成私有化小模型（贪心解码，结果可复现）。")
    if "rag_base" in baselines:
        print("消融档 = 同一基座但不挂 adapter（纯 Qwen2.5-1.5B），用来回答"
              "「这个分数是 LoRA 带来的，还是基座本来就会」。")
    REPORTS_DIR.mkdir(exist_ok=True)
    all_results = {}

    for bl in baselines:
        print(f"\n=== 基线: {bl} ===")
        run_aggs, last_results = [], None
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"  ---- 第 {i + 1}/{args.repeat} 次 ----")
            results = evaluate(bl, data, components,
                               verbose=(args.repeat == 1 or i == 0))
            last_results = results
            run_aggs.append(aggregate(results))
            if args.repeat > 1:
                print(f"       本次汇总: {run_aggs[-1]}")
        entry = {"aggregate": mean_aggregate(run_aggs), "details": last_results}
        if args.repeat > 1:
            entry["runs"] = run_aggs
            entry["stability"] = stability(run_aggs)
            print(f"  稳定性: {json.dumps(entry['stability'], ensure_ascii=False)}")
        all_results[bl] = entry
        print(f"  汇总（{args.repeat} 次均值）: {entry['aggregate']}")

    out_file = Path(args.out) if args.out else (REPORTS_DIR / "eval_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_file}")
    if args.out:
        print("（消融档单独出报告，未覆盖主报告 reports/eval_report.json）")


if __name__ == "__main__":
    main()
