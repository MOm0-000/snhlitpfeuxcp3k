#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA Alpha 单变量 Sweep

目标：
- 以原始 QLoRA 为基线
- 每次只修改 lora_alpha
- 自动生成 YAML
- 若 run 不存在则自动训练
- 若 run 已存在则复用
- 自动跑 medical_100 + CMMLU
- 自动汇总 eval_loss / ROUGE / CMMLU

默认 alpha:
    18, 20, 22, 24, 28

从仓库根目录运行：
    python scripts/run_alpha_sweep.py

只跑部分：
    python scripts/run_alpha_sweep.py --only alpha20 alpha22 alpha24

强制重训：
    python scripts/run_alpha_sweep.py --force-train
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
GENERATED_CONFIG_DIR = ROOT / "configs/sweep_alpha"
RUNS_DIR = ROOT / "runs"
EVAL_DIR = ROOT / "eval"
MODEL_DIR = ROOT / "models/Qwen3-4B-Instruct-2507"

MEDICAL_INPUT = EVAL_DIR / "medical_100.jsonl"
CMMLU_INPUT = EVAL_DIR / "cmmlu_computer_science.jsonl"

CMMLU_SYSTEM = (
    "你是一名计算机科学考试助手。"
    "只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。"
)

ALPHAS = [18, 20, 22, 24, 28]

BASELINE = {
    "rouge_1": 0.22521800360198024,
    "rouge_2": 0.03910942421783961,
    "rouge_l": 0.1718939515510848,
    "cmmlu_accuracy": 0.7794117647058824,
    "cmmlu_correct": 159,
    "cmmlu_total": 204,
}


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只运行指定实验，如 --only alpha20 alpha22",
    )

    ap.add_argument(
        "--force-train",
        action="store_true",
        help="即使已有完整 adapter，也删除对应 run 后重新训练",
    )

    ap.add_argument(
        "--skip-tests",
        action="store_true",
        help="只训练+验证，不跑 Medical/CMMLU",
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="推理 batch size，默认 4",
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

    # 最终目录本身
    if (output_dir / "adapter_model.safetensors").exists():
        return output_dir

    # 优先 checkpoint-100
    ckpt100 = output_dir / "checkpoint-100"
    if (ckpt100 / "adapter_model.safetensors").exists():
        return ckpt100

    # 兜底：取最大 checkpoint
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


def prepare_config(base_cfg, alpha):
    cfg = copy.deepcopy(base_cfg)

    # 统一验证设置，不作为实验变量
    cfg["do_train"] = True
    cfg["do_eval"] = True
    cfg["eval_dataset"] = "med_val_1k"
    cfg["per_device_eval_batch_size"] = 1
    cfg["eval_strategy"] = "steps"
    cfg["eval_steps"] = 50

    # 单变量：只修改 alpha
    cfg["lora_alpha"] = alpha

    name = f"alpha{alpha}"

    output_dir = (
        RUNS_DIR /
        f"qwen3-4b-qlora-{name}-align100"
    )

    cfg["output_dir"] = str(
        output_dir.relative_to(ROOT)
    )

    cfg["overwrite_output_dir"] = False

    GENERATED_CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    yaml_path = (
        GENERATED_CONFIG_DIR /
        f"qwen3_4b_qlora_{name}_align100.yaml"
    )

    with yaml_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        yaml.safe_dump(
            cfg,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    return name, yaml_path, output_dir


def train_if_needed(
    yaml_path,
    output_dir,
    force_train=False,
):
    adapter = find_adapter(output_dir)

    if adapter is not None and not force_train:
        print(
            f"[SKIP TRAIN] 已检测到完整 adapter：{adapter}"
        )
        return adapter

    if force_train and output_dir.exists():
        print(
            f"[FORCE TRAIN] 删除旧目录：{output_dir}"
        )
        shutil.rmtree(output_dir)

    elif (
        output_dir.exists()
        and any(output_dir.iterdir())
        and adapter is None
    ):
        print(
            f"[INCOMPLETE RUN] 检测到不完整目录：{output_dir}"
        )
        print(
            "未找到完整 adapter，将删除该目录并重新训练。"
        )
        shutil.rmtree(output_dir)

    print(
        f"[TRAIN] 未找到可复用 adapter，开始训练："
        f"{yaml_path.relative_to(ROOT)}"
    )

    run([
        "llamafactory-cli",
        "train",
        yaml_path,
    ])

    adapter = find_adapter(output_dir)

    if adapter is None:
        raise RuntimeError(
            f"训练完成但未找到 adapter：{output_dir}"
        )

    return adapter


def run_medical(name, adapter, batch_size):
    pred = (
        EVAL_DIR /
        f"medical_100_{name}.jsonl"
    )

    metric = (
        EVAL_DIR /
        f"medical_{name}_rouge.json"
    )

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


def run_cmmlu(name, adapter, batch_size):
    pred = (
        EVAL_DIR /
        f"cmmlu_{name}.jsonl"
    )

    metric = (
        EVAL_DIR /
        f"cmmlu_{name}_metrics.json"
    )

    wrong = (
        EVAL_DIR /
        f"cmmlu_{name}_wrong.jsonl"
    )

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


def get_train_eval_metrics(output_dir):
    eval_results = load_json(
        Path(output_dir) / "eval_results.json"
    )

    train_results = load_json(
        Path(output_dir) / "train_results.json"
    )

    return {
        "eval_loss": eval_results.get(
            "eval_loss",
            "",
        ),
        "train_loss": train_results.get(
            "train_loss",
            "",
        ),
    }


def classify_result(row):
    """
    医疗优先：
    1. 三个 ROUGE 是否全部 >= 原始 QLoRA
    2. CMMLU 是否 >= baseline
    3. CMMLU 是否只轻微下降
    """

    try:
        r1 = float(row["rouge_1"])
        r2 = float(row["rouge_2"])
        rl = float(row["rouge_l"])
        acc = float(row["cmmlu_accuracy"])
    except (TypeError, ValueError):
        return ""

    all_rouge_pass = (
        r1 >= BASELINE["rouge_1"]
        and r2 >= BASELINE["rouge_2"]
        and rl >= BASELINE["rouge_l"]
    )

    if all_rouge_pass:
        if acc >= BASELINE["cmmlu_accuracy"]:
            return "A: 三项ROUGE全过 + CMMLU不降"

        correct = round(
            acc * BASELINE["cmmlu_total"]
        )

        if correct >= 157:
            return "B: 三项ROUGE全过 + CMMLU轻微下降"

        return "C: 三项ROUGE全过，但CMMLU下降较多"

    passed = sum([
        r1 >= BASELINE["rouge_1"],
        r2 >= BASELINE["rouge_2"],
        rl >= BASELINE["rouge_l"],
    ])

    return f"未达目标: {passed}/3 ROUGE过线"


def save_summary(rows):
    EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        EVAL_DIR /
        "qlora_alpha_sweep_summary.csv"
    )

    json_path = (
        EVAL_DIR /
        "qlora_alpha_sweep_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rows,
            f,
            ensure_ascii=False,
            indent=2,
        )

    fields = [
        "experiment",
        "lora_alpha",
        "eval_loss",
        "train_loss",
        "rouge_1",
        "rouge_2",
        "rouge_l",
        "cmmlu_accuracy",
        "cmmlu_correct",
        "cmmlu_total",
        "r1_vs_baseline",
        "r2_vs_baseline",
        "rl_vs_baseline",
        "cmmlu_vs_baseline",
        "classification",
        "status",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                k: row.get(k, "")
                for k in fields
            })

    print(
        "\n汇总已更新："
        f"\n  {csv_path}"
        f"\n  {json_path}",
        flush=True,
    )


def enrich_comparison(row):
    if row.get("rouge_1") != "":
        row["r1_vs_baseline"] = (
            float(row["rouge_1"])
            - BASELINE["rouge_1"]
        )

    if row.get("rouge_2") != "":
        row["r2_vs_baseline"] = (
            float(row["rouge_2"])
            - BASELINE["rouge_2"]
        )

    if row.get("rouge_l") != "":
        row["rl_vs_baseline"] = (
            float(row["rouge_l"])
            - BASELINE["rouge_l"]
        )

    if row.get("cmmlu_accuracy") != "":
        row["cmmlu_vs_baseline"] = (
            float(row["cmmlu_accuracy"])
            - BASELINE["cmmlu_accuracy"]
        )

    row["classification"] = classify_result(
        row
    )


def main():
    args = parse_args()

    for p in [
        BASE_CONFIG,
        MEDICAL_INPUT,
        CMMLU_INPUT,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"缺少必要文件：{p}"
            )

    with BASE_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as f:
        base_cfg = yaml.safe_load(f)

    experiments = [
        f"alpha{x}"
        for x in ALPHAS
    ]

    if args.only:
        unknown = (
            set(args.only)
            - set(experiments)
        )

        if unknown:
            raise ValueError(
                f"未知实验：{sorted(unknown)}\n"
                f"可选：{experiments}"
            )

        experiments = [
            x
            for x in experiments
            if x in args.only
        ]

    print("Alpha 单变量 Sweep")
    print("基线：")
    print(
        f"  R1={BASELINE['rouge_1']:.6f}"
        f"  R2={BASELINE['rouge_2']:.6f}"
        f"  RL={BASELINE['rouge_l']:.6f}"
        f"  CMMLU={BASELINE['cmmlu_correct']}/"
        f"{BASELINE['cmmlu_total']}"
    )

    print("\n将运行：")

    for name in experiments:
        print(
            f"  - {name}: "
            f"lora_alpha={name.replace('alpha', '')}"
        )

    rows = []

    for idx, name in enumerate(
        experiments,
        1,
    ):
        alpha = int(
            name.replace(
                "alpha",
                "",
            )
        )

        print("\n" + "=" * 76)
        print(
            f"[{idx}/{len(experiments)}] "
            f"{name}"
        )
        print("=" * 76)

        row = {
            "experiment": name,
            "lora_alpha": alpha,
            "eval_loss": "",
            "train_loss": "",
            "rouge_1": "",
            "rouge_2": "",
            "rouge_l": "",
            "cmmlu_accuracy": "",
            "cmmlu_correct": "",
            "cmmlu_total": "",
            "r1_vs_baseline": "",
            "r2_vs_baseline": "",
            "rl_vs_baseline": "",
            "cmmlu_vs_baseline": "",
            "classification": "",
            "status": "running",
        }

        try:
            (
                _,
                yaml_path,
                output_dir,
            ) = prepare_config(
                base_cfg,
                alpha,
            )

            print(
                f"生成 YAML："
                f"{yaml_path.relative_to(ROOT)}"
            )

            adapter = train_if_needed(
                yaml_path,
                output_dir,
                args.force_train,
            )

            row.update(
                get_train_eval_metrics(
                    output_dir
                )
            )

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

                row["rouge_1"] = med.get(
                    "rouge-1",
                    "",
                )

                row["rouge_2"] = med.get(
                    "rouge-2",
                    "",
                )

                row["rouge_l"] = med.get(
                    "rouge-l",
                    "",
                )

                row["cmmlu_accuracy"] = (
                    cmmlu.get(
                        "accuracy",
                        "",
                    )
                )

                row["cmmlu_correct"] = (
                    cmmlu.get(
                        "correct",
                        "",
                    )
                )

                row["cmmlu_total"] = (
                    cmmlu.get(
                        "total",
                        "",
                    )
                )

                enrich_comparison(row)

            row["status"] = "ok"

        except KeyboardInterrupt:
            row["status"] = "interrupted"
            rows.append(row)
            save_summary(rows)
            raise

        except Exception as e:
            row["status"] = (
                f"failed: "
                f"{type(e).__name__}: {e}"
            )

            print(
                f"\n[ERROR] {name}: {e}",
                file=sys.stderr,
            )

            print(
                "继续下一个实验。",
                file=sys.stderr,
            )

        rows.append(row)
        save_summary(rows)

    print("\n" + "=" * 76)
    print("Alpha Sweep 完成")
    print("=" * 76)

    print(
        "结果："
        "\n  eval/qlora_alpha_sweep_summary.csv"
        "\n  eval/qlora_alpha_sweep_summary.json"
    )

    successful = [
        r
        for r in rows
        if r.get("status") == "ok"
        and r.get("rouge_1") != ""
    ]

    if successful:
        print("\n最终测试结果：")

        for r in successful:
            print(
                f"  {r['experiment']}: "
                f"R1={float(r['rouge_1']):.6f} "
                f"R2={float(r['rouge_2']):.6f} "
                f"RL={float(r['rouge_l']):.6f} "
                f"CMMLU="
                f"{r['cmmlu_correct']}/"
                f"{r['cmmlu_total']} "
                f"-> {r['classification']}"
            )

        winners = [
            r
            for r in successful
            if str(
                r.get(
                    "classification",
                    "",
                )
            ).startswith(
                ("A:", "B:")
            )
        ]

        if winners:
            print(
                "\n发现满足目标的候选："
            )

            for r in winners:
                print(
                    f"  *** "
                    f"{r['experiment']} "
                    f"-> "
                    f"{r['classification']}"
                )

        else:
            print(
                "\n本轮暂无 "
                "R1/R2/RL 三项全部过线的配置。"
            )


if __name__ == "__main__":
    main()
