#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA 批量实验 + 补评测脚本 v2

功能：
1. 对已有但尚未跑最终测试的实验：
   - alpha24
   - alpha32_dropout005
   直接复用已有 adapter，跑 Medical + CMMLU，不重新训练。

2. 对学习率 sweep：
   - lr5e5
   - lr6e5
   - lr6p5e5
   - lr7e5
   - lr7p5e5
   - lr1p5e4
   每次只改 learning_rate，其余参数保持基线一致。

3. 自动：
   - 生成 YAML
   - 顺序训练
   - 跑 med_val_1k eval
   - 跑 medical_100
   - 跑 CMMLU
   - 汇总 CSV / JSON

从仓库根目录运行：
    python scripts/run_qlora_sweep_v2.py

只跑指定实验：
    python scripts/run_qlora_sweep_v2.py --only alpha24 alpha32_dropout005 lr6p5e5

只补已有实验的最终评测：
    python scripts/run_qlora_sweep_v2.py --only alpha24 alpha32_dropout005

只跑新的窄区间学习率：
    python scripts/run_qlora_sweep_v2.py --only lr6e5 lr6p5e5 lr7e5
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
GENERATED_CONFIG_DIR = ROOT / "configs/sweep"
RUNS_DIR = ROOT / "runs"
EVAL_DIR = ROOT / "eval"
MODEL_DIR = ROOT / "models/Qwen3-4B-Instruct-2507"

MEDICAL_INPUT = EVAL_DIR / "medical_100.jsonl"
CMMLU_INPUT = EVAL_DIR / "cmmlu_computer_science.jsonl"

CMMLU_SYSTEM = (
    "你是一名计算机科学考试助手。"
    "只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。"
)

# ============================================================
# 已有实验：不重新训练，直接找已有 adapter 做最终测试
# ============================================================
EXISTING_EXPERIMENTS = [
    {
        "name": "alpha24",
        "output_dir": "runs/qwen3-4b-qlora-alpha24-align100",
        "changed_param": "lora_alpha",
        "changed_value": 24,
    },
    {
        "name": "alpha32_dropout005",
        "output_dir": "runs/qwen3-4b-qlora-alpha32-dropout005-align100",
        "changed_param": "lora_alpha+lora_dropout",
        "changed_value": "32+0.05",
    },
]

# ============================================================
# 学习率 sweep：全部从同一基线出发，只改 learning_rate
# ============================================================
LR_EXPERIMENTS = [
    {"name": "lr5e5",    "change": {"learning_rate": 5.0e-5}},
    {"name": "lr6e5",    "change": {"learning_rate": 6.0e-5}},
    {"name": "lr6p5e5",  "change": {"learning_rate": 6.5e-5}},
    {"name": "lr7e5",    "change": {"learning_rate": 7.0e-5}},
    {"name": "lr7p5e5",  "change": {"learning_rate": 7.5e-5}},
    {"name": "lr1p5e4",  "change": {"learning_rate": 1.5e-4}},
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只运行指定实验",
    )
    ap.add_argument(
        "--force-train",
        action="store_true",
        help="对学习率实验强制重新训练；会删除对应 runs 目录",
    )
    ap.add_argument(
        "--skip-tests",
        action="store_true",
        help="学习率实验只训练+验证，不跑 Medical/CMMLU；已有实验也只读取已有结果",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="gen_peft 推理 batch size",
    )
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

    # LLaMA-Factory 最终目录
    if (output_dir / "adapter_model.safetensors").exists():
        return output_dir

    # 优先 checkpoint-100
    ckpt100 = output_dir / "checkpoint-100"
    if (ckpt100 / "adapter_model.safetensors").exists():
        return ckpt100

    # 兜底：找编号最大的 checkpoint
    candidates = []
    for p in output_dir.glob("checkpoint-*"):
        if (p / "adapter_model.safetensors").exists():
            try:
                step = int(p.name.split("-")[-1])
            except ValueError:
                continue
            candidates.append((step, p))

    if candidates:
        candidates.sort()
        return candidates[-1][1]

    return None


def extract_train_eval_metrics(output_dir):
    output_dir = Path(output_dir)
    eval_results = load_json(output_dir / "eval_results.json")
    train_results = load_json(output_dir / "train_results.json")

    return {
        "eval_loss": eval_results.get("eval_loss", ""),
        "train_loss": train_results.get("train_loss", ""),
    }


def prepare_lr_config(base_cfg, exp):
    change = exp["change"]

    if len(change) != 1:
        raise ValueError(
            f"{exp['name']} 不是单因素实验：change 必须恰好包含 1 个参数"
        )

    cfg = copy.deepcopy(base_cfg)

    # 共同设置：所有 LR 实验都启用同一验证集
    cfg["do_train"] = True
    cfg["do_eval"] = True
    cfg["eval_dataset"] = "med_val_1k"
    cfg["per_device_eval_batch_size"] = 1
    cfg["eval_strategy"] = "steps"
    cfg["eval_steps"] = 50

    key, value = next(iter(change.items()))

    if key not in base_cfg:
        raise KeyError(
            f"基线 YAML 中不存在参数 {key!r}，拒绝静默新增"
        )

    cfg[key] = value

    output_dir = RUNS_DIR / f"qwen3-4b-qlora-{exp['name']}-align100"
    cfg["output_dir"] = str(output_dir.relative_to(ROOT))
    cfg["overwrite_output_dir"] = False

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = (
        GENERATED_CONFIG_DIR
        / f"qwen3_4b_qlora_{exp['name']}_align100.yaml"
    )

    with yaml_path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(
            cfg,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    return yaml_path, output_dir, key, value


def train_if_needed(yaml_path, output_dir, force_train=False):
    output_dir = Path(output_dir)
    adapter = find_adapter(output_dir)

    if adapter is not None and not force_train:
        print(f"[SKIP TRAIN] 已检测到 adapter：{adapter}")
        return adapter

    if force_train and output_dir.exists():
        shutil.rmtree(output_dir)

    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"{output_dir} 已存在但没有检测到完整 adapter。\n"
            "为了避免误覆盖，请先检查目录；确认重跑时加 --force-train。"
        )

    run([
        "llamafactory-cli",
        "train",
        yaml_path,
    ])

    adapter = find_adapter(output_dir)
    if adapter is None:
        raise RuntimeError(
            f"训练结束后仍未找到 adapter：{output_dir}"
        )

    return adapter


def run_medical(exp_name, adapter, batch_size):
    pred = EVAL_DIR / f"medical_100_{exp_name}.jsonl"
    metric = EVAL_DIR / f"medical_{exp_name}_rouge.json"

    run([
        sys.executable,
        "scripts/gen_peft.py",
        "--model",
        MODEL_DIR,
        "--adapter",
        adapter,
        "--input",
        MEDICAL_INPUT,
        "--output",
        pred,
        "--max-new-tokens",
        "256",
        "--batch-size",
        str(batch_size),
    ])

    run([
        sys.executable,
        "scripts/evaluate_medical.py",
        "--pred",
        pred,
        "--save",
        metric,
    ])

    return load_json(metric)


def run_cmmlu(exp_name, adapter, batch_size):
    pred = EVAL_DIR / f"cmmlu_{exp_name}.jsonl"
    metric = EVAL_DIR / f"cmmlu_{exp_name}_metrics.json"
    wrong = EVAL_DIR / f"cmmlu_{exp_name}_wrong.jsonl"

    run([
        sys.executable,
        "scripts/gen_peft.py",
        "--model",
        MODEL_DIR,
        "--adapter",
        adapter,
        "--input",
        CMMLU_INPUT,
        "--output",
        pred,
        "--max-new-tokens",
        "32",
        "--batch-size",
        str(batch_size),
        "--system-prompt",
        CMMLU_SYSTEM,
    ])

    run([
        sys.executable,
        "scripts/evaluate_cmmlu.py",
        "--pred",
        pred,
        "--save",
        metric,
        "--wrong-file",
        wrong,
    ])

    return load_json(metric)


def fill_test_metrics(row, med, cmmlu):
    row["rouge_1"] = med.get("rouge-1", "")
    row["rouge_2"] = med.get("rouge-2", "")
    row["rouge_l"] = med.get("rouge-l", "")

    row["cmmlu_accuracy"] = cmmlu.get("accuracy", "")
    row["cmmlu_correct"] = cmmlu.get("correct", "")
    row["cmmlu_total"] = cmmlu.get("total", "")


def save_summary(rows):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    json_path = EVAL_DIR / "qlora_sweep_summary_v2.json"
    csv_path = EVAL_DIR / "qlora_sweep_summary_v2.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fields = [
        "experiment",
        "experiment_type",
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

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({
                k: row.get(k, "")
                for k in fields
            })

    print(
        "\n汇总已更新：\n"
        f"  {csv_path}\n"
        f"  {json_path}",
        flush=True,
    )


def make_empty_row(
    name,
    experiment_type,
    changed_param,
    changed_value,
):
    return {
        "experiment": name,
        "experiment_type": experiment_type,
        "changed_param": changed_param,
        "changed_value": changed_value,
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


def run_existing_experiment(exp, args):
    name = exp["name"]
    output_dir = ROOT / exp["output_dir"]

    row = make_empty_row(
        name,
        "existing_eval_only",
        exp["changed_param"],
        exp["changed_value"],
    )

    adapter = find_adapter(output_dir)

    if adapter is None:
        raise RuntimeError(
            f"{name} 找不到已有 adapter：{output_dir}"
        )

    print(f"[REUSE] {name} 使用已有 adapter：{adapter}")

    metrics = extract_train_eval_metrics(output_dir)
    row.update(metrics)

    if not args.skip_tests:
        med = run_medical(
            name,
            adapter,
            args.batch_size,
        )
        cmmlu = run_cmmlu(
            name,
            adapter,
            args.batch_size,
        )
        fill_test_metrics(row, med, cmmlu)

    row["status"] = "ok"
    return row


def run_lr_experiment(exp, base_cfg, args):
    name = exp["name"]
    key, value = next(iter(exp["change"].items()))

    row = make_empty_row(
        name,
        "lr_sweep",
        key,
        value,
    )

    yaml_path, output_dir, key, value = prepare_lr_config(
        base_cfg,
        exp,
    )

    print(
        f"生成 YAML：{yaml_path.relative_to(ROOT)}\n"
        f"唯一变化：{key} = {value}"
    )

    adapter = train_if_needed(
        yaml_path,
        output_dir,
        force_train=args.force_train,
    )

    metrics = extract_train_eval_metrics(output_dir)
    row.update(metrics)

    if not args.skip_tests:
        med = run_medical(
            name,
            adapter,
            args.batch_size,
        )
        cmmlu = run_cmmlu(
            name,
            adapter,
            args.batch_size,
        )
        fill_test_metrics(row, med, cmmlu)

    row["status"] = "ok"
    return row


def main():
    args = parse_args()

    if not BASE_CONFIG.exists():
        raise FileNotFoundError(
            f"找不到基线配置：{BASE_CONFIG}"
        )

    if not MEDICAL_INPUT.exists():
        raise FileNotFoundError(
            f"找不到医疗测试集：{MEDICAL_INPUT}"
        )

    if not CMMLU_INPUT.exists():
        raise FileNotFoundError(
            f"找不到 CMMLU 测试集：{CMMLU_INPUT}"
        )

    with BASE_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as f:
        base_cfg = yaml.safe_load(f)

    all_names = {
        x["name"]
        for x in EXISTING_EXPERIMENTS + LR_EXPERIMENTS
    }

    if args.only:
        unknown = set(args.only) - all_names
        if unknown:
            raise ValueError(
                f"未知实验：{sorted(unknown)}\n"
                f"可选：{sorted(all_names)}"
            )

    def selected(name):
        return (
            args.only is None
            or name in args.only
        )

    jobs = []

    for exp in EXISTING_EXPERIMENTS:
        if selected(exp["name"]):
            jobs.append(
                ("existing", exp)
            )

    for exp in LR_EXPERIMENTS:
        if selected(exp["name"]):
            jobs.append(
                ("lr", exp)
            )

    print("将运行以下任务：")
    for kind, exp in jobs:
        if kind == "existing":
            print(
                f"  - {exp['name']}: "
                "复用已有模型，只做最终评测"
            )
        else:
            k, v = next(iter(exp["change"].items()))
            print(
                f"  - {exp['name']}: "
                f"只修改 {k}={v}"
            )

    rows = []

    for idx, (kind, exp) in enumerate(jobs, 1):
        name = exp["name"]

        print("\n" + "=" * 76)
        print(
            f"[{idx}/{len(jobs)}] {name}"
        )
        print("=" * 76)

        try:
            if kind == "existing":
                row = run_existing_experiment(
                    exp,
                    args,
                )
            else:
                row = run_lr_experiment(
                    exp,
                    base_cfg,
                    args,
                )

        except KeyboardInterrupt:
            print("\n用户中断。")
            save_summary(rows)
            raise

        except Exception as e:
            print(
                f"\n[ERROR] {name}: {e}",
                file=sys.stderr,
            )
            row = make_empty_row(
                name,
                (
                    "existing_eval_only"
                    if kind == "existing"
                    else "lr_sweep"
                ),
                (
                    exp.get("changed_param")
                    if kind == "existing"
                    else next(
                        iter(exp["change"].keys())
                    )
                ),
                (
                    exp.get("changed_value")
                    if kind == "existing"
                    else next(
                        iter(exp["change"].values())
                    )
                ),
            )
            row["status"] = (
                f"failed: {type(e).__name__}: {e}"
            )

        rows.append(row)
        save_summary(rows)

    print("\n" + "=" * 76)
    print("全部任务结束")
    print("=" * 76)

    print(
        "结果：\n"
        "  eval/qlora_sweep_summary_v2.csv\n"
        "  eval/qlora_sweep_summary_v2.json"
    )

    tested = [
        r
        for r in rows
        if r.get("status") == "ok"
        and r.get("cmmlu_accuracy") != ""
    ]

    if tested:
        print("\n最终测试已完成的实验：")
        for r in tested:
            print(
                f"  {r['experiment']}: "
                f"R1={r['rouge_1']} "
                f"R2={r['rouge_2']} "
                f"RL={r['rouge_l']} "
                f"CMMLU={r['cmmlu_accuracy']}"
            )


if __name__ == "__main__":
    main()
