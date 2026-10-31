#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QLoRA alpha=24 + learning-rate local sweep.

Run from repository root:
    python scripts/run_alpha24_lr_sweep.py

Optional:
    python scripts/run_alpha24_lr_sweep.py --only lr9e5 lr9p5e5
    python scripts/run_alpha24_lr_sweep.py --force-train
"""

import argparse
import copy
import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/qwen3_4b_qlora_align100.yaml"
GENERATED_CONFIG_DIR = ROOT / "configs/sweep_alpha24_lr"
RUNS_DIR = ROOT / "runs"
EVAL_DIR = ROOT / "eval"
MODEL_DIR = ROOT / "models/Qwen3-4B-Instruct-2507"
MEDICAL_INPUT = EVAL_DIR / "medical_100.jsonl"
CMMLU_INPUT = EVAL_DIR / "cmmlu_computer_science.jsonl"

CMMLU_SYSTEM = (
    "你是一名计算机科学考试助手。"
    "只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。"
)

ORIGINAL_QLORA = {
    "rouge_1": 0.22521800360198024,
    "rouge_2": 0.03910942421783961,
    "rouge_l": 0.1718939515510848,
    "cmmlu_accuracy": 0.7794117647058824,
    "cmmlu_correct": 159,
    "cmmlu_total": 204,
}

EXPERIMENTS = [
    {"name": "lr9e5", "learning_rate": 9.0e-5},
    {"name": "lr9p5e5", "learning_rate": 9.5e-5},
    {"name": "lr1e4", "learning_rate": 1.0e-4, "is_control": True},
    {"name": "lr1p05e4", "learning_rate": 1.05e-4},
    {"name": "lr1p1e4", "learning_rate": 1.10e-4},
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    return ap.parse_args()


def run(cmd):
    cmd = [str(x) for x in cmd]
    print("\n$ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_adapter(output_dir):
    output_dir = Path(output_dir)
    if (output_dir / "adapter_model.safetensors").exists():
        return output_dir
    ckpt100 = output_dir / "checkpoint-100"
    if (ckpt100 / "adapter_model.safetensors").exists():
        return ckpt100
    candidates = []
    if output_dir.exists():
        for p in output_dir.glob("checkpoint-*"):
            if not (p / "adapter_model.safetensors").exists():
                continue
            try:
                step = int(p.name.split("-")[-1])
            except ValueError:
                continue
            candidates.append((step, p))
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    return None


def prepare_config(base_cfg, exp):
    cfg = copy.deepcopy(base_cfg)
    cfg["do_train"] = True
    cfg["do_eval"] = True
    cfg["lora_rank"] = 8
    cfg["lora_alpha"] = 24
    cfg["lora_dropout"] = 0.0
    cfg["learning_rate"] = exp["learning_rate"]
    cfg["eval_dataset"] = "med_val_1k"
    cfg["per_device_eval_batch_size"] = 1
    cfg["eval_strategy"] = "steps"
    cfg["eval_steps"] = 50

    if exp.get("is_control"):
        output_dir = RUNS_DIR / "qwen3-4b-qlora-alpha24-align100"
    else:
        output_dir = RUNS_DIR / f"qwen3-4b-qlora-alpha24-{exp['name']}-align100"

    cfg["output_dir"] = str(output_dir.relative_to(ROOT))
    cfg["overwrite_output_dir"] = False

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = GENERATED_CONFIG_DIR / f"qwen3_4b_qlora_alpha24_{exp['name']}_align100.yaml"
    with yaml_path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return yaml_path, output_dir


def train_if_needed(yaml_path, output_dir, force_train=False):
    adapter = find_adapter(output_dir)
    if adapter is not None and not force_train:
        print(f"[SKIP TRAIN] 已检测到完整 adapter：{adapter}")
        return adapter

    if force_train and output_dir.exists():
        print(f"[FORCE TRAIN] 删除旧目录：{output_dir}")
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()) and adapter is None:
        print(f"[INCOMPLETE RUN] {output_dir} 不完整，删除后重新训练")
        shutil.rmtree(output_dir)

    print(f"[TRAIN] 开始训练：{yaml_path.relative_to(ROOT)}")
    run(["llamafactory-cli", "train", yaml_path])
    adapter = find_adapter(output_dir)
    if adapter is None:
        raise RuntimeError(f"训练结束但未找到 adapter：{output_dir}")
    return adapter


def run_medical(name, adapter, batch_size):
    pred = EVAL_DIR / f"medical_100_alpha24_{name}.jsonl"
    metric = EVAL_DIR / f"medical_alpha24_{name}_rouge.json"
    run([
        sys.executable, "scripts/gen_peft.py",
        "--model", MODEL_DIR,
        "--adapter", adapter,
        "--input", MEDICAL_INPUT,
        "--output", pred,
        "--max-new-tokens", "256",
        "--batch-size", str(batch_size),
    ])
    run([
        sys.executable, "scripts/evaluate_medical.py",
        "--pred", pred,
        "--save", metric,
    ])
    return load_json(metric)


def run_cmmlu(name, adapter, batch_size):
    pred = EVAL_DIR / f"cmmlu_alpha24_{name}.jsonl"
    metric = EVAL_DIR / f"cmmlu_alpha24_{name}_metrics.json"
    wrong = EVAL_DIR / f"cmmlu_alpha24_{name}_wrong.jsonl"
    run([
        sys.executable, "scripts/gen_peft.py",
        "--model", MODEL_DIR,
        "--adapter", adapter,
        "--input", CMMLU_INPUT,
        "--output", pred,
        "--max-new-tokens", "32",
        "--batch-size", str(batch_size),
        "--system-prompt", CMMLU_SYSTEM,
    ])
    run([
        sys.executable, "scripts/evaluate_cmmlu.py",
        "--pred", pred,
        "--save", metric,
        "--wrong-file", wrong,
    ])
    return load_json(metric)


def get_train_eval_metrics(output_dir):
    eval_results = load_json(Path(output_dir) / "eval_results.json")
    train_results = load_json(Path(output_dir) / "train_results.json")
    return {
        "eval_loss": eval_results.get("eval_loss", ""),
        "train_loss": train_results.get("train_loss", ""),
    }


def classify_result(row):
    try:
        r1 = float(row["rouge_1"])
        r2 = float(row["rouge_2"])
        rl = float(row["rouge_l"])
        correct = int(row["cmmlu_correct"])
    except (TypeError, ValueError):
        return ""

    passes = [
        r1 >= ORIGINAL_QLORA["rouge_1"],
        r2 >= ORIGINAL_QLORA["rouge_2"],
        rl >= ORIGINAL_QLORA["rouge_l"],
    ]
    if all(passes):
        if correct >= 159:
            return "A: 三项ROUGE全过 + CMMLU不降"
        if correct >= 157:
            return "B: 三项ROUGE全过 + CMMLU轻微下降"
        return "C: 三项ROUGE全过，但CMMLU下降较多"
    return f"未达目标: {sum(passes)}/3 ROUGE过线"


def enrich(row):
    if row.get("rouge_1") != "":
        row["r1_vs_qlora"] = float(row["rouge_1"]) - ORIGINAL_QLORA["rouge_1"]
    if row.get("rouge_2") != "":
        row["r2_vs_qlora"] = float(row["rouge_2"]) - ORIGINAL_QLORA["rouge_2"]
    if row.get("rouge_l") != "":
        row["rl_vs_qlora"] = float(row["rouge_l"]) - ORIGINAL_QLORA["rouge_l"]
    if row.get("cmmlu_accuracy") != "":
        row["cmmlu_vs_qlora"] = float(row["cmmlu_accuracy"]) - ORIGINAL_QLORA["cmmlu_accuracy"]
    row["classification"] = classify_result(row)


def save_summary(rows):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_DIR / "qlora_alpha24_lr_sweep_summary.csv"
    json_path = EVAL_DIR / "qlora_alpha24_lr_sweep_summary.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fields = [
        "experiment", "lora_rank", "lora_alpha", "lora_dropout", "learning_rate",
        "eval_loss", "train_loss", "rouge_1", "rouge_2", "rouge_l",
        "cmmlu_accuracy", "cmmlu_correct", "cmmlu_total",
        "r1_vs_qlora", "r2_vs_qlora", "rl_vs_qlora", "cmmlu_vs_qlora",
        "classification", "status",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    print(f"\n汇总已更新：\n  {csv_path}\n  {json_path}", flush=True)


def main():
    args = parse_args()

    for p in [BASE_CONFIG, MEDICAL_INPUT, CMMLU_INPUT]:
        if not p.exists():
            raise FileNotFoundError(f"缺少必要文件：{p}")

    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    experiments = EXPERIMENTS
    if args.only:
        known = {x["name"] for x in EXPERIMENTS}
        unknown = set(args.only) - known
        if unknown:
            raise ValueError(f"未知实验：{sorted(unknown)}；可选：{sorted(known)}")
        experiments = [x for x in EXPERIMENTS if x["name"] in args.only]

    print("alpha24 + Learning Rate 局部 Sweep")
    print(
        f"原始 QLoRA 门槛：R1>={ORIGINAL_QLORA['rouge_1']:.6f}, "
        f"R2>={ORIGINAL_QLORA['rouge_2']:.6f}, "
        f"RL>={ORIGINAL_QLORA['rouge_l']:.6f}, "
        f"CMMLU最好>={ORIGINAL_QLORA['cmmlu_correct']}/{ORIGINAL_QLORA['cmmlu_total']}"
    )
    print("固定：rank=8, alpha=24, dropout=0.0")
    for exp in experiments:
        print(f"  - {exp['name']}: learning_rate={exp['learning_rate']}")

    rows = []
    for idx, exp in enumerate(experiments, 1):
        name = exp["name"]
        print("\n" + "=" * 76)
        print(f"[{idx}/{len(experiments)}] alpha24 + {name}")
        print("=" * 76)

        row = {
            "experiment": name,
            "lora_rank": 8,
            "lora_alpha": 24,
            "lora_dropout": 0.0,
            "learning_rate": exp["learning_rate"],
            "eval_loss": "",
            "train_loss": "",
            "rouge_1": "",
            "rouge_2": "",
            "rouge_l": "",
            "cmmlu_accuracy": "",
            "cmmlu_correct": "",
            "cmmlu_total": "",
            "r1_vs_qlora": "",
            "r2_vs_qlora": "",
            "rl_vs_qlora": "",
            "cmmlu_vs_qlora": "",
            "classification": "",
            "status": "running",
        }

        try:
            yaml_path, output_dir = prepare_config(base_cfg, exp)
            print(f"生成 YAML：{yaml_path.relative_to(ROOT)}")
            adapter = train_if_needed(yaml_path, output_dir, args.force_train)
            row.update(get_train_eval_metrics(output_dir))

            if not args.skip_tests:
                med = run_medical(name, adapter, args.batch_size)
                cmmlu = run_cmmlu(name, adapter, args.batch_size)
                row["rouge_1"] = med.get("rouge-1", "")
                row["rouge_2"] = med.get("rouge-2", "")
                row["rouge_l"] = med.get("rouge-l", "")
                row["cmmlu_accuracy"] = cmmlu.get("accuracy", "")
                row["cmmlu_correct"] = cmmlu.get("correct", "")
                row["cmmlu_total"] = cmmlu.get("total", "")
                enrich(row)

            row["status"] = "ok"
        except KeyboardInterrupt:
            row["status"] = "interrupted"
            rows.append(row)
            save_summary(rows)
            raise
        except Exception as e:
            row["status"] = f"failed: {type(e).__name__}: {e}"
            print(f"\n[ERROR] {name}: {e}", file=sys.stderr)
            print("继续下一个实验。", file=sys.stderr)

        rows.append(row)
        save_summary(rows)

    print("\n" + "=" * 76)
    print("alpha24 + LR Sweep 完成")
    print("=" * 76)
    print("结果：")
    print("  eval/qlora_alpha24_lr_sweep_summary.csv")
    print("  eval/qlora_alpha24_lr_sweep_summary.json")

    successful = [r for r in rows if r.get("status") == "ok" and r.get("rouge_1") != ""]
    if successful:
        print("\n最终测试结果：")
        for r in successful:
            print(
                f"  {r['experiment']}: "
                f"R1={float(r['rouge_1']):.6f} "
                f"R2={float(r['rouge_2']):.6f} "
                f"RL={float(r['rouge_l']):.6f} "
                f"CMMLU={r['cmmlu_correct']}/{r['cmmlu_total']} "
                f"-> {r['classification']}"
            )

        winners = [
            r for r in successful
            if str(r.get("classification", "")).startswith(("A:", "B:"))
        ]
        if winners:
            print("\n*** 找到满足目标的候选 ***")
            for r in winners:
                print(f"  {r['experiment']} -> {r['classification']}")
        else:
            print("\n本轮暂无 R1/R2/RL 三项全部超过原始 QLoRA 的配置。")


if __name__ == "__main__":
    main()
