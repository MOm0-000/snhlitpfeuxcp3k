# -*- coding: utf-8 -*-
"""CMMLU computer_science 正确率评测

读 gen_batch.py 的输出（每行 {"src","tgt","pred"}），从 pred（模型生成回答）中
解析选项字母 A/B/C/D，与 tgt（正确选项字母）比对，统计正确率。

解析顺序：
    ① 整条回答就是单个字母（可带括号/点号），如 "B"、"（C）"、"D."；
    ② 显式表述："答案：B"、"答案是B"、"选 B"、"选项C" 等；
    ③ 以字母开头的回答，如 "B. 256个"、"C、16个"。

输出：
    - 汇总（总数 / 正确 / 错误 / 未解析，正确率）；
    - 全部错例明细（含未解析条），按“答错（选了别的）”与“未解析（没选出字母）”分组，
      逐条打印：题号、标准答案、解析结果、题干首行、回答片段。

用法：
    python evaluate_cmmlu.py --pred /home/aistudio/work/eval/cmmlu_merged.jsonl
    python evaluate_cmmlu.py --pred .../cmmlu_base.jsonl --save base_cmmlu.json
    python evaluate_cmmlu.py --pred .../cmmlu_merged.jsonl --wrong-file wrong.jsonl   # 错例另存
"""
import argparse
import json
import os
import re

# ① 整条回答就是单个字母
RE_ALONE = re.compile(r'^\s*[（(]?\s*([ABCD])\s*[）)\.．、]?\s*$')
# ② 「答案/选项/选择/选 … X」
RE_EXPLICIT = re.compile(r'(?:答案|选项|选择|选)\s*(?:是|为)?\s*[:：]?\s*[（(]?\s*([ABCD])(?![A-Za-z])')
# ③ 以选项字母开头
RE_LEADING = re.compile(r'^\s*[（(]?\s*([ABCD])(?![A-Za-z])')


def parse_choice(text):
    t = (text or '').strip()
    m = RE_ALONE.match(t)
    if m:
        return m.group(1)
    m = RE_EXPLICIT.search(t)
    if m:
        return m.group(1)
    m = RE_LEADING.match(t)
    if m:
        return m.group(1)
    return None


def brief(text, n=80):
    t = (text or '').replace('\n', ' ').strip()
    return t[:n] + ('…' if len(t) > n else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='gen_batch.py 输出文件（{"src","tgt","pred"}）')
    ap.add_argument('--save', default=None, help='可选：指标另存为 json')
    ap.add_argument('--wrong-file', default=None, help='可选：错例（含未解析）另存为 jsonl')
    args = ap.parse_args()

    total = correct = 0
    wrong, unparsed = [], []          # wrong=选了别的字母；unparsed=没解析出字母
    for idx, line in enumerate(open(args.pred, encoding='utf-8'), 1):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        total += 1
        gold = ((o.get('tgt') or [''])[0] or '').strip().upper()
        pred = (o.get('pred') or '').strip()
        choice = parse_choice(pred)
        if choice == gold:
            correct += 1
        elif choice is None:
            unparsed.append((idx, gold, pred))
        else:
            wrong.append((idx, gold, choice, pred))

    acc = correct / max(total, 1)
    print(f'共 {total} 题：正确 {correct}，答错 {len(wrong)}，未解析 {len(unparsed)}'
          f'（均按错误计）→ 正确率 {acc:.2%}  —— {args.pred}\n')

    print(f'—— 答错（选了别的字母）{len(wrong)} 条 ——')
    for idx, gold, choice, pred in wrong:
        print(f'[{idx}] 标准 {gold}｜解析 {choice}｜回答：{brief(pred)}')

    print(f'\n—— 未解析（没选出字母，含截断）{len(unparsed)} 条 ——')
    for idx, gold, pred in unparsed:
        print(f'[{idx}] 标准 {gold}｜回答：{brief(pred)}')

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump({'total': total, 'correct': correct,
                       'wrong': len(wrong), 'unparsed': len(unparsed),
                       'accuracy': acc}, f, ensure_ascii=False, indent=2)
        print('\n指标已保存 ->', args.save)
    if args.wrong_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.wrong_file)), exist_ok=True)
        with open(args.wrong_file, 'w', encoding='utf-8') as f:
            for idx, gold, choice, pred in wrong:
                f.write(json.dumps({'idx': idx, 'gold': gold, 'choice': choice,
                                    'pred': pred}, ensure_ascii=False) + '\n')
            for idx, gold, pred in unparsed:
                f.write(json.dumps({'idx': idx, 'gold': gold, 'choice': None,
                                    'pred': pred}, ensure_ascii=False) + '\n')
        print('错例已保存 ->', args.wrong_file)


if __name__ == '__main__':
    main()
