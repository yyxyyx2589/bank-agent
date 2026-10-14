#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_lora.py — 银行域 LoRA 微调（中间基线 baseline_rag_lora 的模型侧）。

训练对象：Qwen2.5-1.5B-Instruct（本地路径 D:/yy/model/Qwen2.5-1.5B-Instruct）
训练数据：data/bank_sft.jsonl（由 training/build_sft.py 生成，RAG 范式对齐 +
          completion-only loss mask）
产出：    training/out/bank_lora/ —— 独立的银行域 adapter

为什么是独立 adapter，而不是复用答辩项目那个 qwen-lora：
  那个 adapter 是**运维域**微调的（酷哇科技实训数据）。拿它答银行业务，
  属于跨域，答出来的东西不能代表「银行域微调」的效果，会污染中间基线。
  所以这里从头训一个银行域 adapter，两个项目的模型产物互不干扰。

超参不是拍脑袋的，每一处都对应本机的实际约束（RTX 4060 Laptop / 8GB）：
  · max_seq_length=512   —— 实测训练数据 token 长度中位 257、最长 309
                            （单个 KB 片段 + 一个问 + 一段答复），
                            用 2048 纯属浪费显存和算力
  · batch=4 × accum=4    —— 等效 batch 16，8GB 显存下 fp16 跑得动
  · lr=2e-4 / cosine     —— LoRA 微调常用档，r=16 时不必再小
  · 5 epochs             —— 样本量小（约 300 条），靠多跑几轮把域风格压进去；
                            配合 dropout=0.05 抗过拟合

用法：
  python training/train_lora.py                 # 完整训练
  python training/train_lora.py --max-steps 20  # 冒烟测试（验证链路能跑通）
  python training/train_lora.py --quantize 4bit # 显存不够时 4bit 量化基座
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "D:/yy/model/Qwen2.5-1.5B-Instruct"
DEFAULT_DATA = BASE_DIR / "data" / "bank_sft.jsonl"
DEFAULT_OUT = BASE_DIR / "training" / "out" / "bank_lora"
TRAIN_REPORT = BASE_DIR / "reports" / "lora_train.json"

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 512  # 实测数据最长 309 token，512 有富余


class MaskedDataCollator:
    """预 tokenized + completion-only mask 数据的 padding / 截断。

    输入每条带 {input_ids, labels}：labels 里 prompt 段已是 -100，
    只有 assistant 回复段保留真实 token id —— 这样模型只学「怎么答」，
    不去学「怎么复述提示词」。

    两个细节：
      · pad 的 labels 填 -100（不是 pad_token_id）：若填 pad id，
        模型会被训练去预测一串 pad，污染 loss
      · 超长时**从左截**：completion 在序列尾部，从右截会把要学的部分切掉
    """

    def __init__(self, pad_token_id: int, max_len: int = MAX_SEQ_LENGTH):
        self.pad_token_id = pad_token_id
        self.max_len = max_len

    def __call__(self, features):
        import torch
        input_ids, labels, attn = [], [], []
        for f in features:
            ids, lab = list(f["input_ids"]), list(f["labels"])
            if len(ids) > self.max_len:
                ids, lab = ids[-self.max_len:], lab[-self.max_len:]
            input_ids.append(ids)
            labels.append(lab)
            attn.append([1] * len(ids))
        m = max(len(x) for x in input_ids)
        for i in range(len(input_ids)):
            p = m - len(input_ids[i])
            input_ids[i] += [self.pad_token_id] * p
            labels[i] += [-100] * p
            attn[i] += [0] * p
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser(description="Qwen2.5 银行域 LoRA 微调")
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help="0=完整训练，调试用")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--quantize", choices=["4bit"], default=None,
                    help="4bit 量化加载基座（显存不足时）")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        print("[x] 未检测到可用 CUDA —— 1.5B 的 LoRA 微调放 CPU 上不现实，已中止")
        return
    print(f"[*] GPU: {torch.cuda.get_device_name(0)} "
          f"| 显存 {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ---------- 1. 基座 ----------
    print(f"[*] 加载基座: {args.model_path}")
    load_kwargs = {}
    if args.quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, **load_kwargs)

    # ---------- 2. LoRA ----------
    print(f"[*] LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    # ---------- 3. 数据 ----------
    dataset = load_dataset("json", data_files=str(args.data), split="train")
    print(f"[✓] 训练样本 {len(dataset)} 条（已预 tokenized + completion-only mask）")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="no",          # adapter 体积小，训练结束统一保存即可
        max_length=MAX_SEQ_LENGTH,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],
        seed=42,                     # 固定种子：让「可复现」不是一句空话
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=dataset,
                         data_collator=MaskedDataCollator(tokenizer.pad_token_id))

    # ---------- 4. 训练 ----------
    print("[*] 开始训练 …")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # ---------- 5. 训练记录（写进报告，可被追问） ----------
    logs = [h for h in trainer.state.log_history if "loss" in h]
    first = logs[0]["loss"] if logs else None
    last = logs[-1]["loss"] if logs else None
    record = {
        "base_model": args.model_path,
        "data": str(args.data),
        "adapter_out": str(out_dir),
        "samples": len(dataset),
        "epochs": args.epochs,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "batch": args.batch_size, "grad_accum": args.grad_accum,
        "lr": args.lr, "seed": 42, "quantize": args.quantize or "none",
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_sec": round(elapsed, 1),
        "loss_first": round(first, 4) if first is not None else None,
        "loss_last": round(last, 4) if last is not None else None,
        "loss_curve": [{"step": h.get("step"), "loss": round(h["loss"], 4)} for h in logs],
    }
    TRAIN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_REPORT.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print("=" * 60)
    print(f"[✓] 训练完成，耗时 {elapsed/60:.1f} 分钟")
    print(f"    loss {record['loss_first']} → {record['loss_last']}")
    print(f"    adapter: {out_dir}")
    print(f"    训练记录: {TRAIN_REPORT}")


if __name__ == "__main__":
    main()
