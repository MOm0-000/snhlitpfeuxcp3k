#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA 单因素批量实验脚本（OFAT: One-Factor-At-A-Time）

功能：
1. 读取同一个 QLoRA 基线 YAML；
2. 每个实验只修改一个超参数；
3. 自动生成独立 YAML；
4. 顺序执行 LLaMA-Factory 训练；
5. 自动跑 medical_100 + CMMLU；
6. 自动汇总 eval_loss / ROUGE / CMMLU 到 CSV 和 JSON。

从仓库根目录运行：
    python scripts/run_qlora_sweep.py

只跑指定实验：
    python scripts/run_qlora_sweep.py --only lr5e5 lr7p5e5

若某实验已经完成（output_dir 下已有 adapter_model.safetensors 或 checkpoint-100），
默认跳过训练，直接复用模型并执行/刷新测试。
"""

import argparse
import copy
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/qwen3_4b_qlora_align100.yaml"
GENERATED_CONFIG_DIR = ROOT / "configs/sweep"
RUNS_DIR = ROOT / "runs"
EVAL_DIR = ROOT / "eval"
MODEL_DIR = ROOT / "models/Qwen3-4B-Instruct-2507"

MEDICAL_INPUT = EVAL_DIR / "medical_100.jsonl"
CMMLU_INPUT = EVAL_DIR / "cmmlu_computer_science.jsonl"

CMMLU_SYSTEM = "你是一名计算机科学考试助手。只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。"

# ------------------------------------------------------------
# 实验列表：每一项必须只有一个 change。
# 想增加实验，只需继续往这里加一项。
# ------------------------------------------------------------
EXPERIMENTS = [
    {
        "name": "lr5e5",
        "change": {"learning_rate": 5.0e-5},
    },
    {
        "name": "lr7p5e5",
        "change": {"learning_rate": 7.5e-5},
    },
    {
        "name": "lr1p5e4",
        "change": {"learning_rate": 1.5e-4},
    },
    {
        "name": "rank4",
        "change": {"lora_rank": 4},
    },
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只运行指定实验，例如 --only lr5e5 rank4",
    )
    ap.add_argument(
        "--force-train",
        action="store_true",
        help="即使检测到已有训练产物，也重新训练。注意：会删除对应 runs 目录。",
    )
    ap.add_argument(
        "--skip-tests",
        action="store_true",
        help="只训练和验证，不跑 medical/CMMLU。",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    return ap.parse_args()


def run(cmd):
    print("\n$ " + shlex.join([str(x) for x in cmd]), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_summary(rows):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DIR / "qlora_sweep_summary.json"
    csv_path = EVAL_DIR / "qlora_sweep_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fields = [
        "experiment",
        "changed_param",
        "changed_value",
        "eval_loss",
        "train_loss",
        "rouge_1",
        "rouge_2",
        "rouge_l",
        "cmmlu_accuracy",
        "cmmlu_correct",
        "cmmlu_total",
        "status",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    print(f"\n汇总已更新：\n  {csv_path}\n  {json_path}", flush=True)


def find_adapter(output_dir):
    output_dir = Path(output_dir)
    # LLaMA-Factory 最终输出目录本身通常就包含最终 adapter。
    if (output_dir / "adapter_model.safetensors").exists():
        return output_dir
    ckpt100 = output_dir / "checkpoint-100"
    if (ckpt100 / "adapter_model.safetensors").exists():
        return ckpt100
    return None


def prepare_config(base_cfg, exp):
    change = exp["change"]
    if len(change) != 1:
        raise ValueError(f"{exp['name']} 不是单因素实验：change 必须恰好只有 1 个键")

    cfg = copy.deepcopy(base_cfg)

    # 所有实验共同启用同一个验证集；这不是实验变量。
    cfg["do_train"] = True
    cfg["do_eval"] = True
    cfg["eval_dataset"] = "med_val_1k"
    cfg["per_device_eval_batch_size"] = 1
    cfg["eval_strategy"] = "steps"
    cfg["eval_steps"] = 50

    # 每个实验只改变一个真正的实验变量。
    key, value = next(iter(change.items()))
    if key not in base_cfg:
        raise KeyError(f"基线 YAML 中不存在参数 {key!r}，拒绝静默新增，避免实验失控。")
    cfg[key] = value

    output_dir = RUNS_DIR / f"qwen3-4b-qlora-{exp['name']}-align100"
    cfg["output_dir"] = str(output_dir.relative_to(ROOT))
    cfg["overwrite_output_dir"] = False

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = GENERATED_CONFIG_DIR / f"qwen3_4b_qlora_{exp['name']}_align100.yaml"
    with yaml_path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    return yaml_path, output_dir, key, value


def remove_dir(path):
    import shutil
    if Path(path).exists():
        shutil.rmtree(path)


def train_if_needed(yaml_path, output_dir, force_train=False):
    adapter = find_adapter(output_dir)
    if adapter is not None and not force_train:
        print(f"[SKIP TRAIN] 已检测到完整 adapter：{adapter}")
        return adapter

    if force_train:
        remove_dir(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"{output_dir} 已存在但未检测到完整 adapter。\n"
            "为避免覆盖半成品，请检查目录；确认要重跑时使用 --force-train。"
        )

    run(["llamafactory-cli", "train", yaml_path])
    adapter = find_adapter(output_dir)
    if adapter is None:
        raise RuntimeError(f"训练结束但未找到 adapter：{output_dir}")
    return adapter


def run_medical(exp_name, adapter, batch_size):
    pred = EVAL_DIR / f"medical_100_{exp_name}.jsonl"
    metric = EVAL_DIR / f"medical_{exp_name}_rouge.json"

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


def run_cmmlu(exp_name, adapter, batch_size):
    pred = EVAL_DIR / f"cmmlu_{exp_name}.jsonl"
    metric = EVAL_DIR / f"cmmlu_{exp_name}_metrics.json"
    wrong = EVAL_DIR / f"cmmlu_{exp_name}_wrong.jsonl"

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


def main():
    args = parse_args()

    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"找不到基线配置：{BASE_CONFIG}")
    if not MEDICAL_INPUT.exists():
        raise FileNotFoundError(f"找不到医疗测试集：{MEDICAL_INPUT}")
    if not CMMLU_INPUT.exists():
        raise FileNotFoundError(f"找不到 CMMLU 测试集：{CMMLU_INPUT}")

    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    experiments = EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        known = {x["name"] for x in EXPERIMENTS}
        unknown = wanted - known
        if unknown:
            raise ValueError(f"未知实验：{sorted(unknown)}；可选：{sorted(known)}")
        experiments = [x for x in EXPERIMENTS if x["name"] in wanted]

    print("将运行以下单因素实验：")
    for exp in experiments:
        k, v = next(iter(exp["change"].items()))
        print(f"  - {exp['name']}: {k} = {v!r}")

    rows = []

    for idx, exp in enumerate(experiments, 1):
        name = exp["name"]
        print("\n" + "=" * 72)
        print(f"[{idx}/{len(experiments)}] 实验：{name}")
        print("=" * 72)

        row = {
            "experiment": name,
            "changed_param": "",
            "changed_value": "",
            "eval_loss": "",
            "train_loss": "",
            "rouge_1": "",
            "rouge_2": "",
            "rouge_l": "",
            "cmmlu_accuracy": "",
            "cmmlu_correct": "",
            "cmmlu_total": "",
            "status": "running",
        }

        try:
            yaml_path, output_dir, key, value = prepare_config(base_cfg, exp)
            row["changed_param"] = key
            row["changed_value"] = value
            print(f"生成 YAML：{yaml_path.relative_to(ROOT)}")

            adapter = train_if_needed(
                yaml_path, output_dir, force_train=args.force_train
            )

            eval_results = load_json(output_dir / "eval_results.json")
            train_results = load_json(output_dir / "train_results.json")
            row["eval_loss"] = eval_results.get("eval_loss", "")
            row["train_loss"] = train_results.get("train_loss", "")

            if not args.skip_tests:
                med = run_medical(name, adapter, args.batch_size)
                cmmlu = run_cmmlu(name, adapter, args.batch_size)
                row["rouge_1"] = med.get("rouge-1", "")
                row["rouge_2"] = med.get("rouge-2", "")
                row["rouge_l"] = med.get("rouge-l", "")
                row["cmmlu_accuracy"] = cmmlu.get("accuracy", "")
                row["cmmlu_correct"] = cmmlu.get("correct", "")
                row["cmmlu_total"] = cmmlu.get("total", "")

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

    print("\n" + "=" * 72)
    print("全部实验结束")
    print("=" * 72)
    print(f"结果表：{(EVAL_DIR / 'qlora_sweep_summary.csv').relative_to(ROOT)}")

    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("eval_loss") != ""]
    if ok_rows:
        best = min(ok_rows, key=lambda r: float(r["eval_loss"]))
        print(
            f"当前最低 eval_loss：{best['experiment']} "
            f"({best['changed_param']}={best['changed_value']}) -> {best['eval_loss']}"
        )


if __name__ == "__main__":
    main()
