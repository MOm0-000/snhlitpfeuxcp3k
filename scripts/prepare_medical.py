# -*- coding: utf-8 -*-
"""Chinese-medical-dialogue（Alpaca 格式）→ 评测数据（src/tgt 格式，与训练数据一致）

- 取前 100 条；
- src = input（患者提问原文），tgt = output（参考回答）；

产出（UTF-8，每行一条）：
  medical_100.jsonl   {"src": ["患者提问"], "tgt": ["参考回答"]}

用法：
  python prepare_medical.py \
      --src /home/aistudio/work/train_0001_of_0001.json \
      --out-dir /home/aistudio/work/eval
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='train_0001_of_0001.json 路径')
    ap.add_argument('--out-dir', default='/home/aistudio/work/eval')
    ap.add_argument('--n', type=int, default=100)
    args = ap.parse_args()

    dec = json.JSONDecoder()
    with open(args.src, encoding='utf-8') as f:
        buf = f.read(2 << 20)                      # 先读 2MB，足够取前 n 条
    i = buf.index('[') + 1
    items = []
    while len(items) < args.n:
        if i > len(buf) - 200000:
            raise SystemExit('读入缓冲不足（--src 指向的文件格式不符？）')
        while buf[i] in ' \n\r\t,':                # 跳过记录间分隔符
            i += 1
        obj, i = dec.raw_decode(buf, i)
        q = (obj.get('input') or '').strip()
        a = (obj.get('output') or '').strip()
        if q and a:                                # 跳过空记录
            items.append({'src': [q], 'tgt': [a]})

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'medical_100.jsonl')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
    print(f'medical_100.jsonl: {len(items)} 条 -> {out}')


if __name__ == '__main__':
    main()
