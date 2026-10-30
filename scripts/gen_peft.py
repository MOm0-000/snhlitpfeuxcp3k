# -*- coding: utf-8 -*-
"""批量生成：Qwen3-4B 4-bit 基座，可选加载 PEFT/LoRA adapter。

输出每行 {"src": [...], "tgt": [...], "pred": "..."}，可直接喂
evaluate_medical.py / evaluate_cmmlu.py。
"""
import argparse
import json
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="none", help="PEFT adapter 目录；none 表示只用基座")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0=全量")
    ap.add_argument("--batch-size", type=int, default=1, help="生成批大小；8GB 建议 4")
    ap.add_argument("--system", "--system-prompt", dest="system", default="", help="系统提示词；CMMLU 用它约束只输出字母")
    return ap.parse_args()


def build_prompt(tokenizer, text, system=""):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def main():
    args = parse_args()
    torch.manual_seed(23)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if args.adapter and args.adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    n = 0
    batch_size = max(1, args.batch_size)
    with open(args.output, "w", encoding="utf-8", newline="\n") as out:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = [build_prompt(tokenizer, (r.get("src") or [""])[0], args.system) for r in batch]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            for j, row in enumerate(batch):
                src = (row.get("src") or [""])[0]
                tgt = row.get("tgt") or [""]
                new_tokens = generated[j, prompt_len:]
                pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                pred = re.sub(r"<think>.*?</think>", "", pred, flags=re.S).strip()
                out.write(json.dumps({"src": [src], "tgt": tgt, "pred": pred}, ensure_ascii=False) + "\n")
                out.flush()
                n += 1
            done = min(start + batch_size, len(rows))
            if done % 10 == 0 or done == len(rows):
                print(f"[{done}/{len(rows)}] done", flush=True)

    print(f"wrote {n} rows -> {args.output}")


if __name__ == "__main__":
    main()
