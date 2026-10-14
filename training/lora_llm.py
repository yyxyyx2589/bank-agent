#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lora_llm.py — 本地 Qwen2.5-1.5B + 银行域 LoRA 的推理客户端。

存在意义：给中间基线 baseline_rag_lora 换上**真的本地模型**。
在此之前 eval.py 里那一行是

    "lora_llm": llm,     # 注释写着"预留基线，复用纯 RAG 链路"

也就是说「RAG+LoRA」这一列其实跑的是 DeepSeek —— 基线名与实际不符，
报告里那一列数据是假的。本模块把它换成基座 + bank_lora adapter 的本地推理。

接口契约：必须与 llm_client.LLMClient.chat() **完全一致**（返回对象要有
`.choices[0].message.content`），否则 eval.py 的 run_baseline_rag 拿不到答案。

一个刻意的选择：**贪心解码（do_sample=False）**。
  · 小模型采样解码很容易跑偏、跑重复，用贪心能拿到它最稳的那条回答；
  · 更重要的是**可复现**：`eval.py --repeat 3` 要报稳定性极差，
    如果解码带随机性，极差里混进了「采样噪声」，就说不清波动来自模型能力
    还是来自随机种子。贪心解码让「稳定性」这个指标变干净。

用法（一般不用手动跑，eval.py 会自动加载）：
    python training/lora_llm.py --adapter training/out/bank_lora   # 冒烟自测
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "D:/yy/model/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = BASE_DIR / "training" / "out" / "bank_lora"


def adapter_available(adapter_dir: Path) -> bool:
    """adapter 是否存在且完整（有配置文件 + 权重）。"""
    if not adapter_dir.is_dir():
        return False
    has_cfg = (adapter_dir / "adapter_config.json").exists()
    has_w = any(adapter_dir.glob("adapter_model.*"))
    return has_cfg and has_w


class LocalLoRAClient:
    """本地推理客户端，接口与 LLMClient 对齐。

    adapter_dir=None 时**只加载基座、不挂 adapter** —— 这是消融实验要用的：
    同一条检索链路，唯一的差别就是「有没有微调过」。没有这一档，
    「我训了 LoRA」就只是一句做了什么事，而不是「LoRA 带来了多少提升」。
    """

    def __init__(self, adapter_dir=DEFAULT_ADAPTER, model_path=DEFAULT_MODEL,
                 max_new_tokens: int = 200, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.adapter_dir = str(adapter_dir) if adapter_dir else None
        self.max_new_tokens = max_new_tokens

        if not torch.cuda.is_available() and device == "cuda":
            print("[!] 无可用 CUDA，退回 CPU（会非常慢，仅用于验证链路）")
            device = "cpu"
        self.device = device

        print(f"[*] 加载基座 {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True)
        if self.adapter_dir:
            from peft import PeftModel
            print(f"[*] 挂载银行域 adapter {self.adapter_dir}")
            self.model = PeftModel.from_pretrained(base, self.adapter_dir)
            self.tag = "qwen2.5-1.5b+bank_lora"
        else:
            print("[*] 消融档：不挂 adapter（纯基座）")
            self.model = base
            self.tag = "qwen2.5-1.5b(基座)"
        self.model.eval()
        if device == "cuda":
            self.model = self.model.cuda()

    def chat(self, messages: list, tools: list | None = None,
             stream: bool = False, tool_choice=None) -> object:
        """与 OpenAI SDK 同形的返回：.choices[0].message.content。

        tools / tool_choice 参数仅为接口兼容而保留 —— 1.5B 基座不做
        Function Calling，中间基线也不需要工具（它测的就是「只有检索、
        没有编排」这条路径）。传了会被忽略，但不会报错。
        """
        torch = self.torch
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,           # 贪心：确定性 + 可复现
                repetition_penalty=1.05,   # 小模型容易复读，轻微惩罚
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        content = self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        msg = SimpleNamespace(content=content, tool_calls=None, role="assistant")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=None, model=self.tag)


def main():
    import json
    import sys
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--no-adapter", action="store_true",
                    help="只加载基座、不挂 adapter（消融实验：验证 LoRA 到底带来了什么）")
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--demo", action="store_true", help="跑真实样例看输出")
    args = ap.parse_args()
    sys.path.insert(0, str(BASE_DIR))

    if not args.no_adapter and not adapter_available(Path(args.adapter)):
        print(f"[x] adapter 不存在或不完整: {args.adapter}")
        print("    先跑 python training/train_lora.py（或加 --no-adapter 只跑基座）")
        return

    t0 = time.time()
    llm = LocalLoRAClient(None if args.no_adapter else args.adapter, args.model_path)
    print(f"[✓] 模型就绪，加载耗时 {time.time()-t0:.1f}s")

    if args.demo:
        from db.init_chroma import KB_DOCUMENTS
        kb = {d["id"]: d for d in KB_DOCUMENTS}
        cases = [("kb-003", "银行卡丢了要怎么补办"), ("kb-016", "R3等级的理财风险大吗")]
        for kid, q in cases:
            user = (f"基于知识片段回答：\n{kb[kid]['text']}\n\n问题：{q}")
            t = time.time()
            resp = llm.chat([
                {"role": "system", "content": "你是银行坐席知识助手。"},
                {"role": "user", "content": user}])
            print(f"\n--- {q} ({time.time()-t:.1f}s) ---")
            print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
