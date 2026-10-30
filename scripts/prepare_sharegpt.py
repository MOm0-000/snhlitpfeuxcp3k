# -*- coding: utf-8 -*-
"""DISC-Med-SFT 原始多轮对话 -> LLaMA-Factory ShareGPT 数据。

直接产出：
  val_1k_sharegpt.json      1,000 条验证样本
  train_2k_sharegpt.json    2,000 条训练子集（供 100 步实验）
  dataset_info.json         两个数据集在 LLaMA-Factory 中的注册信息

训练池先抽 20,000 条作为 2,000 条子集的抽取池，只在内存中使用，不写文件。
不再生成 train_srctgt.jsonl / val_srctgt.jsonl 等 src-tgt 中间文件。
"""
import argparse
import json
import os
import random


def to_single_turn(convs):
    """多轮对话 -> 单轮 ShareGPT 样本；每条 assistant 回复配紧邻的前一条 user 提问。"""
    items = []
    for d in convs:
        prev_role, prev_content = None, ""
        for turn in (d.get("conversation") or []):
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role == "assistant" and prev_role == "user" and prev_content and content:
                items.append({
                    "conversations": [
                        {"from": "human", "value": prev_content},
                        {"from": "gpt", "value": content},
                    ]
                })
            prev_role, prev_content = role, content
    return items


def write_json(path, items):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"{os.path.basename(path)}: {len(items):,} 条 -> {path}")


def dataset_info():
    tags = {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
    }
    def entry(file_name):
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": tags,
        }
    return {
        "med_train_2k": entry("train_2k_sharegpt.json"),
        "med_val_1k": entry("val_1k_sharegpt.json"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="DISC-Med-SFT_released.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-n", type=int, default=20000, help="训练池抽样条数，仅作子集抽取池，不写文件")
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--subset-n", type=int, default=2000)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--sample-seed", type=int, default=23)
    args = ap.parse_args()

    convs = []
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            c = d.get("conversation")
            if isinstance(c, list) and len(c) >= 2:
                convs.append(d)
    print(f"读入 {len(convs):,} 段对话（过滤不足 2 轮的记录）")

    # 对话级 9:1 切分；同一段对话不会同时出现在训练池和验证池
    random.Random(args.split_seed).shuffle(convs)
    split_idx = int(len(convs) * (1 - args.val_ratio))
    train_pool = to_single_turn(convs[:split_idx])
    val_pool = to_single_turn(convs[split_idx:])
    print(f"单轮化：训练池 {len(train_pool):,} 条 / 验证池 {len(val_pool):,} 条")

    # 训练池先抽 20,000 条，作为 2,000 条子集的抽取池；这个中间池不写文件
    rng = random.Random(args.sample_seed)
    train_sample = rng.sample(train_pool, min(args.train_n, len(train_pool)))
    val_items = rng.sample(val_pool, min(args.val_n, len(val_pool)))
    subset_items = random.Random(args.sample_seed).sample(
        train_sample, min(args.subset_n, len(train_sample))
    )

    os.makedirs(args.out_dir, exist_ok=True)
    write_json(os.path.join(args.out_dir, "val_1k_sharegpt.json"), val_items)
    write_json(os.path.join(args.out_dir, "train_2k_sharegpt.json"), subset_items)

    info_path = os.path.join(args.out_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(dataset_info(), f, ensure_ascii=False, indent=2)
    print(f"dataset_info.json: 2 个数据集 -> {info_path}")


if __name__ == "__main__":
    main()
