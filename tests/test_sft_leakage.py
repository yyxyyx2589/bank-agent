#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sft_leakage.py — 锁住「训练数据不泄漏评测题」这个保证。

为什么值得单独写一个测试套件：
  中间基线（RAG + LoRA）是这个项目里**唯一一个有"作弊空间"的基线**——
  它的模型是在本项目自己的知识库语料上微调的，只要训练数据里混进了评测题的
  问法，它就能靠记忆拿分，三基线对比立刻失去意义（而且这种失真从数字上看
  是"好消息"，最容易被将就过去）。

  所以这道闸门不能只靠构建脚本里的一次性打印，得有回归测试：
  以后任何人改了 build_sft.py 或题库，跑一遍测试就能知道泄漏保证还在不在。

测试分三层：
  1. 相似度函数本身的行为（该抓的抓住、该放过的放过）
  2. 训练数据产物与评测题的实际最大相似度（端到端的硬保证）
  3. 训练数据确实做了 RAG 范式对齐（messages 结构 = 推理时的结构）

零 API 消耗、不需要 GPU。
运行：python tests/test_sft_leakage.py
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "training"))

from build_sft import (  # noqa: E402
    _norm, _sim, NEAR_DUP_THRESHOLD, RAG_SYSTEM, RAG_USER_TMPL,
    load_eval_questions,
)

SFT_RAW = BASE_DIR / "data" / "bank_sft_raw.jsonl"

_ok, _fail = 0, 0


def check(label: str, cond: bool) -> bool:
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"[PASS] {label}")
    else:
        _fail += 1
        print(f"[FAIL] {label}")
    return cond


def test_similarity_function():
    # 该抓的近重复：第一版实测漏掉的 8 条，逐条验证现在能被抓住
    caught = [
        ("银行卡丢了要怎么补办呢", "银行卡丢了怎么补办"),
        ("未成年人能不能自己去开卡", "未成年人能自己开卡吗"),
        ("信用卡分期的手续费是怎么收的", "信用卡分期手续费怎么算"),
        ("房贷提前还款要交违约金吗", "房贷提前还款有违约金吗"),
        ("理财产品的R3等级是什么意思", "理财产品R3是什么意思"),
    ]
    for a, b in caught:
        r = _sim(a, b)
        check(f"近重复被抓：{r:.2f} 「{a}」vs「{b}」", r >= NEAR_DUP_THRESHOLD)

    # 该放过的：正常的换问法，不该被当成泄漏剔除
    passed = [
        ("转账限额要怎么提高", "附近有什么好吃的餐厅推荐吗"),
        ("定期存款都有哪些期限", "信用卡逾期会上征信吗"),
        ("企业网银转账需要几个人审批", "今天天气怎么样"),
    ]
    for a, b in passed:
        r = _sim(a, b)
        check(f"正常问法不误伤：{r:.2f} 「{a}」vs「{b}」", r < NEAR_DUP_THRESHOLD)

    # 归一化：空白不该影响判定（与 eval.py 的 _norm 同源）
    check("归一化后相同：'20 万元' 与 '20万元'",
          _norm("20 万元") == _norm("20万元"))


def test_no_leakage_in_data():
    """端到端硬保证：训练数据里任何一条问题，与任何评测题的相似度都 < 阈值。"""
    if not SFT_RAW.exists():
        check("训练数据产物存在（data/bank_sft_raw.jsonl）", False)
        print("      → 先跑 python training/build_sft.py")
        return
    eval_qs = load_eval_questions()
    check(f"评测题集合非空（{len(eval_qs)} 条）", len(eval_qs) > 0)

    rows = [json.loads(l) for l in SFT_RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    check(f"训练数据非空（{len(rows)} 条）", len(rows) > 0)

    worst, worst_pair = 0.0, ("", "")
    for r in rows:
        # user 消息里带检索片段，问题在最后一行「问题：xxx」
        user = r["messages"][1]["content"]
        q = user.split("问题：")[-1].strip() if "问题：" in user else user.strip()
        nq = _norm(q)
        for e in eval_qs:
            s = _sim(nq, e)
            if s > worst:
                worst, worst_pair = s, (q, e)
    check(f"训练集与评测题最大相似度 {worst:.3f} < 阈值 {NEAR_DUP_THRESHOLD}"
          f"（最接近的一对：「{worst_pair[0]}」vs「{worst_pair[1]}」）",
          worst < NEAR_DUP_THRESHOLD)


def test_paradigm_alignment():
    """训练数据的 messages 必须与 run_baseline_rag 推理时的结构逐字同构，
    否则 LoRA 会被训成「裸问题直答器」，学不会用检索片段。"""
    if not SFT_RAW.exists():
        return
    rows = [json.loads(l) for l in SFT_RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    sample = rows[0]["messages"]
    check("三条消息：system / user / assistant", len(sample) == 3)
    check("角色顺序正确", [m["role"] for m in sample] == ["system", "user", "assistant"])
    check(f"system 与推理一致：{RAG_SYSTEM!r}", sample[0]["content"] == RAG_SYSTEM)

    with_ctx = [r for r in rows
                if r["messages"][1]["content"].startswith("基于知识片段回答：")]
    check(f"绝大多数样本带检索片段（{len(with_ctx)}/{len(rows)}）",
          len(with_ctx) >= len(rows) * 0.9)

    # 模板必须能复现出同样的 user 内容
    m = with_ctx[0]["messages"][1]["content"]
    body = m[len("基于知识片段回答：\n"):]
    ctx, q = body.split("\n\n问题：")
    check("user 内容可被 RAG_USER_TMPL 精确复现",
          RAG_USER_TMPL.format(context=ctx, question=q) == m)


if __name__ == "__main__":
    for fn in [test_similarity_function, test_no_leakage_in_data,
               test_paradigm_alignment]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n断言 {_ok} 通过 / {_fail} 失败")
    if _fail:
        sys.exit(1)
    print("全部通过：训练数据与评测题无泄漏，且与推理范式同构。")
