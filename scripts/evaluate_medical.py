# -*- coding: utf-8 -*-
"""医疗对话自动评测（ROUGE-1 / ROUGE-2 / ROUGE-L）

读 gen_batch.py 的输出（每行 {"src","tgt","pred"}），把 pred（模型生成回答）
与 tgt（参考回答）分词后用空格连接，计算 ROUGE-1/2/L 并取全量平均（f1）。

依赖：
    pip install rouge-chinese jieba

用法：
    python evaluate_medical.py --pred /home/aistudio/work/eval/medical_100_merged.jsonl
    python evaluate_medical.py --pred .../medical_100_base.jsonl --save base_rouge.json
"""
import argparse
import json
import os

import jieba
from rouge_chinese import Rouge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='gen_batch.py 输出文件（{"src","tgt","pred"}）')
    ap.add_argument('--save', default=None, help='可选：指标另存为 json')
    args = ap.parse_args()

    hyps, refs = [], []
    n_total = 0
    for line in open(args.pred, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        n_total += 1
        pred = (o.get('pred') or '').strip()
        ref = ((o.get('tgt') or [''])[0] or '').strip()
        if not pred:                        # 空输出无法计算，按 0 分计入平均
            continue
        hyps.append(' '.join(jieba.cut(pred)))      # 必须先分词、空格连接
        refs.append(' '.join(jieba.cut(ref)))

    out = {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    if hyps:
        scores = Rouge().get_scores(hyps, refs, avg=True)   # 非空条目的平均
        ratio = len(hyps) / max(n_total, 1)                 # 空输出条数按 0 折算
        for k in out:
            out[k] = scores[k]['f'] * ratio

    print(f'共 {n_total} 条（有效 {len(hyps)} 条）—— {args.pred}')
    for k, v in out.items():
        print(f'  {k:<8} f1 = {v:.4f}')
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print('已保存 ->', args.save)


if __name__ == '__main__':
    main()
