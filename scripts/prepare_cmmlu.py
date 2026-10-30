# -*- coding: utf-8 -*-
"""CMMLU computer_science.csv → 评测数据（src/tgt 格式，与训练数据一致）

- src = 题干与四个选项拼成的整段问题，tgt = 正确选项字母（用于判分）；
- 原始文件首列为无名行号列，按列名读取、自动忽略；

产出（UTF-8，每行一条）：
  cmmlu_computer_science.jsonl
  {"src": ["题干\\nA. …\\nB. …\\nC. …\\nD. …"], "tgt": ["B"]}

用法：
  python prepare_cmmlu.py \
      --src /home/aistudio/work/computer_science.csv \
      --out-dir /home/aistudio/work/eval
"""
import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='computer_science.csv 路径')
    ap.add_argument('--out-dir', default='/home/aistudio/work/eval')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.src, encoding='utf-8-sig', newline='')))
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'cmmlu_computer_science.jsonl')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            q = (f"{r['Question'].strip()}\n"
                 f"A. {r['A'].strip()}\n"
                 f"B. {r['B'].strip()}\n"
                 f"C. {r['C'].strip()}\n"
                 f"D. {r['D'].strip()}")
            f.write(json.dumps({'src': [q], 'tgt': [r['Answer'].strip()]},
                               ensure_ascii=False) + '\n')
    print(f'cmmlu_computer_science.jsonl: {len(rows)} 条 -> {out}')


if __name__ == '__main__':
    main()
