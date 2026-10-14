"""评估逻辑测试：验证评分函数（不调 LLM，零 API 消耗）。

覆盖：
  1. score_answer        —— 答案命中期望要点比例（含空白归一化）
  2. score_tool_accuracy —— 工具选择准确率 + 参数准确率（字符串双向包含）
  3. aggregate           —— 汇总指标计算（含路由准确率）

运行：python tests/test_eval_logic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import (score_answer, score_tool_accuracy, aggregate,
                  mean_aggregate, stability)


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    return cond


def test_score_answer():
    points = ["身份证", "网点"]
    assert check("全命中得 1.0",
                 score_answer("需本人持身份证到营业网点办理", points) == 1.0)
    assert check("命中一个得 0.5",
                 score_answer("需本人持身份证办理", points) == 0.5)
    assert check("全没命中得 0.0",
                 score_answer("请致电客服", points) == 0.0)
    assert check("无期望要点默认 1.0",
                 score_answer("任意答案", []) == 1.0)
    # 空白归一化：期望「20万」，答案是「20 万元」也要判命中
    assert check("空白归一化后命中",
                 score_answer("起存金额为 20 万元", ["20万"]) == 1.0)
    assert check("全角空格也归一化",
                 score_answer("起存金额为 20\u3000万元", ["20万"]) == 1.0)


def test_score_tool_accuracy():
    expected = [
        {"name": "query_ticket", "params": {"customer_id": "C2026031800237"}},
        {"name": "get_product_rate", "params": {"keyword": "大额存单"}},
    ]
    # 完全匹配
    trace_full = [
        {"name": "query_ticket", "params": {"customer_id": "C2026031800237"}, "result": "..."},
        {"name": "get_product_rate", "params": {"keyword": "大额存单"}, "result": "..."},
    ]
    s = score_tool_accuracy(trace_full, expected)
    assert check("全匹配：工具准确率 1.0", s["tool_accuracy"] == 1.0)
    assert check("全匹配：参数准确率 1.0", s["param_accuracy"] == 1.0)

    # 只调了一个工具
    trace_half = [{"name": "query_ticket", "params": {"customer_id": "C2026031800237"}, "result": "..."}]
    s = score_tool_accuracy(trace_half, expected)
    assert check("调一半：工具准确率 0.5", s["tool_accuracy"] == 0.5)

    # 参数错（客户号不同）
    trace_bad_param = [
        {"name": "query_ticket", "params": {"customer_id": "C9999999999999"}, "result": "..."},
    ]
    s = score_tool_accuracy(trace_bad_param, expected)
    assert check("参数错：参数准确率 0.0", s["param_accuracy"] == 0.0)

    # 参数为自然语言近似（标注「挂失」，模型传「银行卡挂失」）应判命中
    s = score_tool_accuracy(
        [{"name": "check_business_rule", "params": {"scenario": "银行卡挂失"}, "result": "..."}],
        [{"name": "check_business_rule", "params": {"scenario": "挂失"}}])
    assert check("参数双向包含：判命中", s["param_accuracy"] == 1.0)

    # 无期望工具
    s = score_tool_accuracy([], [])
    assert check("无期望工具默认全对", s["tool_accuracy"] == 1.0)


def test_aggregate():
    results = [
        {"answer_score": 1.0, "tool_accuracy": 1.0, "param_accuracy": 1.0,
         "route_accuracy": 1.0, "turns": 2, "latency_ms": 100,
         "route": "rag_direct"},
        {"answer_score": 0.5, "tool_accuracy": 0.5, "param_accuracy": 0.0,
         "route_accuracy": 0.0, "turns": 4, "latency_ms": 300,
         "route": "agent_loop"},
        {"answer_score": 1.0, "tool_accuracy": 1.0, "param_accuracy": 1.0,
         "route_accuracy": 1.0, "turns": 1, "latency_ms": 500,
         "route": "agent_loop"},
    ]
    agg = aggregate(results)
    assert check("答案分均值 0.833", agg["answer_score_avg"] == 0.833)
    assert check("路由准确率均值 0.667", agg["route_accuracy_avg"] == 0.667)
    assert check("工具准确率均值 0.833", agg["tool_accuracy_avg"] == 0.833)
    assert check("参数准确率均值 0.667", agg["param_accuracy_avg"] == 0.667)
    assert check("平均轮次 2.33", agg["turns_avg"] == 2.33)
    assert check("平均延迟 300ms", agg["latency_ms_avg"] == 300)
    assert check("计数 3", agg["count"] == 3)
    # 按路由分组：整体均值 300ms 掩盖了"直答 100ms / 循环 400ms"的差别
    assert check("分路由延迟：直答 100ms",
                 agg["latency_by_route"]["rag_direct"] == 100)
    assert check("分路由延迟：循环 (300+500)/2=400ms",
                 agg["latency_by_route"]["agent_loop"] == 400)

    # 缺 route 字段（旧数据）不应崩，落到 unknown 组
    agg_old = aggregate([{"answer_score": 1.0, "tool_accuracy": 1.0,
                          "param_accuracy": 1.0, "route_accuracy": 1.0,
                          "turns": 0, "latency_ms": 50}])
    assert check("缺 route 字段不崩，归入 unknown",
                 agg_old["latency_by_route"] == {"unknown": 50})


def test_stability():
    """多次运行的均值与极差（--repeat N 用）。"""
    runs = [
        {"count": 20, "answer_score_avg": 1.0, "route_accuracy_avg": 1.0,
         "tool_accuracy_avg": 1.0, "param_accuracy_avg": 1.0,
         "turns_avg": 0.8, "latency_ms_avg": 3200,
         "latency_by_route": {"rag_direct": 1500, "agent_loop": 4000}},
        {"count": 20, "answer_score_avg": 0.95, "route_accuracy_avg": 1.0,
         "tool_accuracy_avg": 1.0, "param_accuracy_avg": 0.95,
         "turns_avg": 0.9, "latency_ms_avg": 3400,
         "latency_by_route": {"rag_direct": 1700, "agent_loop": 3600}},
    ]
    m = mean_aggregate(runs)
    assert check("均值：答案分 0.975", m["answer_score_avg"] == 0.975)
    assert check("均值：count 取第一次（20）", m["count"] == 20)
    assert check("均值：分路由延迟逐键取均值",
                 m["latency_by_route"] == {"rag_direct": 1600, "agent_loop": 3800})

    st = stability(runs)
    assert check("稳定性：运行次数 2", st["runs"] == 2)
    assert check("稳定性：答案分 min/max 正确",
                 st["answer_score_avg"] == {"mean": 0.975, "min": 0.95, "max": 1.0})
    assert check("稳定性：路由准确率极差为 0（完全稳定）",
                 st["route_accuracy_avg"]["min"] == st["route_accuracy_avg"]["max"] == 1.0)

    # 单次运行不产出稳定性块
    assert check("单次运行不产出稳定性", stability([runs[0]]) == {"runs": 1})


if __name__ == "__main__":
    for fn in [test_score_answer, test_score_tool_accuracy, test_aggregate,
               test_stability]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n全部通过：评估评分逻辑正确。")
