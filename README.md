# 实验 III：Qwen3-4B 中文医疗对话参数高效微调

本仓库用于完成《自然语言处理 C》实验 III：**利用对话数据微调基础大模型——基于 Qwen3-4B-Instruct-2507 的中文医疗对话本地微调**。

实验以 **Qwen3-4B-Instruct-2507** 为基座模型，使用 **LLaMA-Factory** 完成 4-bit 参数高效微调，并使用 **Transformers + PEFT** 完成统一批量推理。当前仓库包含三组主实验，以及围绕 QLoRA 超参数的多组补充实验：

- **QLoRA**：4-bit NF4 + LoRA；
- **RS-LoRA**：在 QLoRA 基础上启用 Rank-Stabilized LoRA；
- **QOFT**：作为非 LoRA 的参数高效微调对照；
- **QLoRA Sweep**：学习率、LoRA rank、LoRA alpha 等单变量实验。

评测包含两部分：

- **Chinese-medical-dialogue**：100 条医疗问答，使用 ROUGE-1 / ROUGE-2 / ROUGE-L；
- **CMMLU Computer Science**：204 道选择题，使用 Accuracy。

---

## 1. 实验环境

本仓库当前 `requirements.txt` 对应的主要环境为：

```text
Python: 3.11
PyTorch: 2.11.0+cu128
Transformers: 5.8.0
PEFT: 0.18.1
Accelerate: 1.11.0
bitsandbytes: 0.50.2
jieba: 0.42.1
rouge-chinese: 1.0.3
```

LLaMA-Factory 固定到 Git commit：

```text
47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

> `requirements.txt` 使用 PyTorch CUDA 12.8 wheel 源。请先确认本机 NVIDIA 驱动和 GPU 环境能够运行对应版本的 PyTorch。

---

## 2. 项目结构

下面只列出当前仓库中最重要的文件和目录；`eval/`、`runs/` 与 sweep 配置目录中还包含多组已生成结果。

```text
.
├── configs/
│   ├── qwen3_4b_qlora_align100.yaml
│   ├── qwen3_4b_qrslora_align100.yaml
│   ├── qwen3_4b_qoft_align100.yaml
│   ├── qwen3_4b_qlora_alpha24_align100.yaml
│   ├── qwen3_4b_qlora_alpha32_align100.yaml
│   ├── qwen3_4b_qlora_dropout005_align100.yaml
│   ├── qwen3_4b_qlora_alpha32_dropout005_align100.yaml
│   ├── sweep/
│   ├── sweep_alpha/
│   └── sweep_alpha24_lr/
│
├── data/
│   ├── raw/
│   │   ├── computer_science.csv
│   │   ├── DISC-Med-SFT/
│   │   └── Chinese-medical-dialogue/
│   └── processed/
│       ├── dataset_info.json
│       ├── train_2k_sharegpt.json
│       └── val_1k_sharegpt.json
│
├── eval/
│   ├── medical_100.jsonl
│   ├── cmmlu_computer_science.jsonl
│   ├── medical_*_rouge.json
│   ├── cmmlu_*_metrics.json
│   ├── cmmlu_*_wrong.jsonl
│   ├── qlora_sweep_summary.json
│   ├── qlora_alpha_sweep_summary.json
│   └── qlora_alpha24_lr_sweep_summary.json
│
├── models/
│   └── Qwen3-4B-Instruct-2507/
│
├── runs/
│   ├── qwen3-4b-qlora-align100/
│   ├── qwen3-4b-qrslora-align100/
│   ├── qwen3-4b-qoft-align100/
│   └── qwen3-4b-qlora-*/
│
├── scripts/
│   ├── prepare_sharegpt.py
│   ├── prepare_medical.py
│   ├── prepare_cmmlu.py
│   ├── gen_peft.py
│   ├── evaluate_medical.py
│   ├── evaluate_cmmlu.py
│   ├── run_qlora_sweep.py
│   ├── run_qlora_sweep_v2.py
│   ├── run_alpha_sweep.py
│   └── run_alpha24_lr_sweep.py
│
├── requirements.txt
├── qoft_align100.log
├── 实验指导书III-本地版.docx
└── 实验指导书III-本地版.pdf
```

### 脚本职责

| 脚本 | 作用 |
|---|---|
| `prepare_sharegpt.py` | DISC-Med-SFT 多轮对话切分、单轮化并转换为 LLaMA-Factory ShareGPT 格式 |
| `prepare_medical.py` | Chinese-medical-dialogue → 100 条医疗评测 JSONL |
| `prepare_cmmlu.py` | CMMLU `computer_science.csv` → 统一 `src/tgt` JSONL |
| `gen_peft.py` | 4-bit Qwen3 批量推理，可选加载 PEFT adapter |
| `evaluate_medical.py` | 对已经生成的 `pred` 计算 ROUGE-1/2/L |
| `evaluate_cmmlu.py` | 从已经生成的 `pred` 解析 A/B/C/D 并计算准确率 |
| `run_qlora_sweep.py` | QLoRA 单因素学习率 / rank sweep，自动训练、推理、评测和汇总 |
| `run_qlora_sweep_v2.py` | QLoRA 扩展 sweep 与已有实验补评测 |
| `run_alpha_sweep.py` | `lora_alpha` 单变量 sweep |
| `run_alpha24_lr_sweep.py` | 固定 `alpha=24` 后做局部 learning-rate sweep |

---

## 3. 基座模型

实验使用：

```text
Qwen/Qwen3-4B-Instruct-2507
```

建议下载到仓库约定路径：

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/Qwen3-4B-Instruct-2507
```

训练 YAML 和推理脚本默认都使用：

```text
models/Qwen3-4B-Instruct-2507
```

仓库 `.gitignore` 已忽略部分大型模型权重和原始数据文件，因此重新克隆仓库后，应先确认本地模型与原始数据是否完整。

---

## 4. 数据准备

### 4.1 DISC-Med-SFT：训练集与验证集

将原始数据放到：

```text
data/raw/DISC-Med-SFT/DISC-Med-SFT_released.jsonl
```

然后运行：

```bash
python scripts/prepare_sharegpt.py \
  --src data/raw/DISC-Med-SFT/DISC-Med-SFT_released.jsonl \
  --out-dir data/processed \
  --train-n 20000 \
  --val-n 1000 \
  --subset-n 2000
```

脚本会：

1. 过滤不足两轮的无效对话；
2. 按完整对话进行训练 / 验证切分，避免同一段对话泄漏到两侧；
3. 将多轮对话转换为单轮 `human -> gpt` 样本；
4. 从训练池抽取 2,000 条用于本实验；
5. 从验证池抽取 1,000 条；
6. 自动生成 LLaMA-Factory 使用的 `dataset_info.json`。

输出：

```text
data/processed/train_2k_sharegpt.json
data/processed/val_1k_sharegpt.json
data/processed/dataset_info.json
```

其中注册的数据集名为：

```text
med_train_2k
med_val_1k
```

### 4.2 Chinese-medical-dialogue：医疗测试集

将原始测试数据准备为：

```text
data/raw/Chinese-medical-dialogue/data/train_0001_of_0001.json
```

生成前 100 条有效评测样本：

```bash
python scripts/prepare_medical.py \
  --src data/raw/Chinese-medical-dialogue/data/train_0001_of_0001.json \
  --out-dir eval \
  --n 100
```

输出：

```text
eval/medical_100.jsonl
```

单行格式：

```json
{"src": ["患者提问"], "tgt": ["参考回答"]}
```

### 4.3 CMMLU Computer Science

仓库中使用：

```text
data/raw/computer_science.csv
```

转换：

```bash
python scripts/prepare_cmmlu.py \
  --src data/raw/computer_science.csv \
  --out-dir eval
```

输出：

```text
eval/cmmlu_computer_science.jsonl
```

当前评测集共 **204 道题**。每行格式为：

```json
{
  "src": ["题干\nA. ...\nB. ...\nC. ...\nD. ..."],
  "tgt": ["B"]
}
```

---

## 5. 主实验配置

三组主实验使用同一个基座模型、同一份 2,000 条训练数据和相同训练预算。

### 5.1 公共设置

| 参数 | 取值 |
|---|---:|
| Base Model | Qwen3-4B-Instruct-2507 |
| Quantization | bitsandbytes 4-bit |
| Quantization Type | NF4 |
| Double Quantization | true |
| Dataset | `med_train_2k` |
| cutoff_len | 1024 |
| max_samples | 2000 |
| Batch Size | 1 |
| Gradient Accumulation | 16 |
| Effective Batch Size | 16 |
| Learning Rate | 1e-4 |
| Max Steps | 100 |
| Warmup Steps | 20 |
| Scheduler | cosine |
| Optimizer | paged_adamw_8bit |
| BF16 | true |
| Gradient Checkpointing | true |
| Seed | 23 |
| Save Steps | 50 |

### 5.2 方法差异

| 方法 | 关键配置 |
|---|---|
| QLoRA | `finetuning_type: lora`, `lora_target: all`, `rank=8`, `alpha=16`, `dropout=0.0` |
| RS-LoRA | 与 QLoRA 相同，并增加 `use_rslora: true` |
| QOFT | `finetuning_type: oft`, `oft_block_size: 32`, `oft_target: all`, `module_dropout: 0.0` |

配置文件分别为：

```text
configs/qwen3_4b_qlora_align100.yaml
configs/qwen3_4b_qrslora_align100.yaml
configs/qwen3_4b_qoft_align100.yaml
```

---

## 6. 训练

### 6.1 QLoRA

```bash
llamafactory-cli train configs/qwen3_4b_qlora_align100.yaml
```

输出：

```text
runs/qwen3-4b-qlora-align100
```

最终评测使用：

```text
runs/qwen3-4b-qlora-align100/checkpoint-100
```

仓库现有训练结果：

```text
Epoch:         0.8
Train loss:    1.7281
Runtime:       535.3 s
```

### 6.2 RS-LoRA

```bash
llamafactory-cli train configs/qwen3_4b_qrslora_align100.yaml
```

输出：

```text
runs/qwen3-4b-qrslora-align100
```

仓库现有训练结果：

```text
Epoch:         0.8
Train loss:    1.6248
Runtime:       535.1 s
```

### 6.3 QOFT

```bash
llamafactory-cli train configs/qwen3_4b_qoft_align100.yaml
```

输出：

```text
runs/qwen3-4b-qoft-align100
```

仓库现有训练结果：

```text
Epoch:         0.8
Train loss:    1.6102
Runtime:       591.5 s
```

---

## 7. 统一推理接口

当前仓库统一使用：

```text
scripts/gen_peft.py
```

它负责真正的模型推理，并将结果写成：

```json
{"src": ["..."], "tgt": ["..."], "pred": "模型生成结果"}
```

之后再把该文件交给 `evaluate_medical.py` 或 `evaluate_cmmlu.py` 评分。

主要参数：

```text
--model             基座模型目录，必填
--adapter           PEFT adapter；none 表示只用 Base
--input             src/tgt JSONL，必填
--output            生成结果 JSONL，必填
--max-new-tokens    最大新生成 token 数，默认 256
--limit             只生成前 N 条；0 表示全量
--batch-size        推理 batch size，默认 1
--system-prompt     可选 system prompt
```

推理统一使用：

```text
4-bit NF4
BF16 compute
do_sample = false
Qwen3 enable_thinking = false（支持时）
```

---

## 8. 医疗问答评测

### 8.1 Base

生成：

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter none \
  --input eval/medical_100.jsonl \
  --output eval/medical_100_base.jsonl \
  --max-new-tokens 256 \
  --batch-size 4
```

评分：

```bash
python scripts/evaluate_medical.py \
  --pred eval/medical_100_base.jsonl \
  --save eval/medical_base_rouge.json
```

### 8.2 QLoRA

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qlora-align100/checkpoint-100 \
  --input eval/medical_100.jsonl \
  --output eval/medical_100_lora.jsonl \
  --max-new-tokens 256 \
  --batch-size 4

python scripts/evaluate_medical.py \
  --pred eval/medical_100_lora.jsonl \
  --save eval/medical_lora_rouge.json
```

### 8.3 RS-LoRA

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qrslora-align100/checkpoint-100 \
  --input eval/medical_100.jsonl \
  --output eval/medical_100_rslora.jsonl \
  --max-new-tokens 256 \
  --batch-size 4

python scripts/evaluate_medical.py \
  --pred eval/medical_100_rslora.jsonl \
  --save eval/medical_rslora_rouge.json
```

### 8.4 QOFT

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qoft-align100/checkpoint-100 \
  --input eval/medical_100.jsonl \
  --output eval/medical_100_qoft.jsonl \
  --max-new-tokens 256 \
  --batch-size 4

python scripts/evaluate_medical.py \
  --pred eval/medical_100_qoft.jsonl \
  --save eval/medical_qoft_rouge.json
```

`evaluate_medical.py` **只负责读取已经生成的预测结果并计算指标，不负责加载模型**。它对中文文本使用 `jieba` 分词，再使用 `rouge-chinese` 计算 ROUGE-1 / ROUGE-2 / ROUGE-L F1。

---

## 9. CMMLU Computer Science 评测

统一 System Prompt：

```text
你是一名计算机科学考试助手。只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。
```

Base 示例：

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter none \
  --input eval/cmmlu_computer_science.jsonl \
  --output eval/cmmlu_base.jsonl \
  --max-new-tokens 32 \
  --batch-size 4 \
  --system-prompt "你是一名计算机科学考试助手。只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。"

python scripts/evaluate_cmmlu.py \
  --pred eval/cmmlu_base.jsonl \
  --save eval/cmmlu_base_metrics.json \
  --wrong-file eval/cmmlu_base_wrong.jsonl
```

评测其他 adapter 时只需替换 `--adapter` 和输出文件名。例如 QLoRA：

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qlora-align100/checkpoint-100 \
  --input eval/cmmlu_computer_science.jsonl \
  --output eval/cmmlu_lora.jsonl \
  --max-new-tokens 32 \
  --batch-size 4 \
  --system-prompt "你是一名计算机科学考试助手。只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。"

python scripts/evaluate_cmmlu.py \
  --pred eval/cmmlu_lora.jsonl \
  --save eval/cmmlu_lora_metrics.json \
  --wrong-file eval/cmmlu_lora_wrong.jsonl
```

`evaluate_cmmlu.py` 会按以下顺序从 `pred` 中解析选项：

1. 整条回答就是 `A/B/C/D`；
2. `答案：B`、`选 C`、`选项D` 等显式表述；
3. 以选项字母开头的回答，如 `B. ...`。

未解析出的回答也按错误计入准确率。

---

## 10. 主实验结果

结果直接对应仓库当前 `eval/*_rouge.json`、`eval/*_metrics.json` 与 `runs/*/train_results.json`。

| 模型 | Train Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | CMMLU |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-4B | — | 0.1578 | 0.0131 | 0.1062 | **79.90%** (163/204) |
| QLoRA | 1.7281 | **0.2252** | 0.0391 | 0.1719 | 77.94% (159/204) |
| RS-LoRA | 1.6248 | 0.2227 | **0.0410** | **0.1804** | 75.00% (153/204) |
| QOFT | **1.6102** | 0.2119 | 0.0393 | 0.1772 | 79.41% (162/204) |

可以看到：

- 三种微调方法在医疗问答 ROUGE 上均明显优于 Base；
- QLoRA 的 ROUGE-1 最高；
- RS-LoRA 的 ROUGE-2 / ROUGE-L 最高，但 CMMLU 下降最多；
- QOFT 的医疗指标明显优于 Base，同时 CMMLU 只比 Base 少答对 1 题；
- 更低的训练 loss 并不意味着所有下游指标都会同步提高。

---

## 11. QLoRA 单因素 Sweep

### 11.1 基础 OFAT Sweep

运行：

```bash
python scripts/run_qlora_sweep.py
```

只运行指定实验：

```bash
python scripts/run_qlora_sweep.py --only lr5e5 lr7p5e5 rank4
```

强制重训：

```bash
python scripts/run_qlora_sweep.py --force-train
```

脚本会从 `configs/qwen3_4b_qlora_align100.yaml` 出发，每次只修改一个实验变量，并自动：

```text
生成 YAML
-> LLaMA-Factory 训练 / 复用已有 adapter
-> Medical 100 推理 + ROUGE
-> CMMLU 推理 + Accuracy
-> 写出 CSV / JSON 汇总
```

当前汇总文件：

```text
eval/qlora_sweep_summary.json
eval/qlora_sweep_summary.csv
```

当前结果：

| 实验 | 变量 | ROUGE-1 | ROUGE-2 | ROUGE-L | CMMLU |
|---|---|---:|---:|---:|---:|
| `lr5e5` | lr = 5e-5 | 0.2204 | **0.0423** | **0.1779** | 76.47% |
| `lr7p5e5` | lr = 7.5e-5 | 0.2223 | 0.0359 | 0.1741 | **78.92%** |
| `lr1p5e4` | lr = 1.5e-4 | 0.2180 | 0.0379 | 0.1718 | 76.96% |
| `rank4` | rank = 4 | 0.2227 | 0.0381 | 0.1733 | 77.94% |

其中 `lr=5e-5` 提高了 ROUGE-2 / ROUGE-L，但牺牲了 ROUGE-1 与 CMMLU；`lr=7.5e-5` 的 CMMLU 更接近 Base，但医疗指标并未全面超过原始 QLoRA。

### 11.2 Alpha Sweep

运行全部 alpha：

```bash
python scripts/run_alpha_sweep.py
```

只运行部分：

```bash
python scripts/run_alpha_sweep.py --only alpha20 alpha22 alpha24
```

默认测试：

```text
alpha = 18, 20, 22, 24, 28
```

其余核心参数保持 QLoRA 基线不变，即 `rank=8`、`lr=1e-4`、`dropout=0.0`。

汇总：

```text
eval/qlora_alpha_sweep_summary.json
eval/qlora_alpha_sweep_summary.csv
```

当前结果：

| alpha | Eval Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | CMMLU |
|---:|---:|---:|---:|---:|---:|
| 18 | 1.3603 | 0.2227 | 0.0377 | 0.1724 | 77.94% |
| 20 | 1.3564 | 0.2212 | **0.0396** | 0.1734 | 77.45% |
| 22 | 1.3528 | 0.2202 | 0.0388 | 0.1737 | 76.96% |
| 24 | 1.3495 | **0.2227** | 0.0393 | 0.1741 | 77.45% |
| 28 | **1.3443** | 0.2222 | 0.0393 | **0.1750** | 76.47% |

对比原始 QLoRA（`alpha=16`）：

```text
ROUGE-1 = 0.2252
ROUGE-2 = 0.0391
ROUGE-L = 0.1719
CMMLU   = 77.94%
```

随着 alpha 增大，验证 loss 持续下降，ROUGE-L 总体上升，但 ROUGE-1 和 CMMLU 没有同步改善。因此，仅以验证 loss 选择 alpha 会与最终多指标目标存在偏差。

### 11.3 固定 alpha=24 的局部学习率 Sweep

运行：

```bash
python scripts/run_alpha24_lr_sweep.py
```

只运行部分：

```bash
python scripts/run_alpha24_lr_sweep.py --only lr9e5 lr9p5e5
```

固定：

```text
rank = 8
alpha = 24
dropout = 0.0
```

测试：

```text
lr = 9e-5, 9.5e-5, 1e-4, 1.05e-4, 1.1e-4
```

汇总：

```text
eval/qlora_alpha24_lr_sweep_summary.json
eval/qlora_alpha24_lr_sweep_summary.csv
```

当前结果：

| Learning Rate | Eval Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | CMMLU |
|---:|---:|---:|---:|---:|---:|
| 9e-5 | 1.3553 | 0.2197 | 0.0381 | 0.1729 | 77.45% |
| 9.5e-5 | 1.3522 | 0.2185 | 0.0383 | 0.1735 | **77.94%** |
| 1e-4 | 1.3495 | **0.2227** | 0.0393 | 0.1741 | 77.45% |
| 1.05e-4 | 1.3471 | 0.2202 | **0.0399** | 0.1746 | 75.98% |
| 1.1e-4 | **1.3448** | 0.2210 | 0.0391 | **0.1754** | 75.98% |

这组结果同样没有找到在三个 ROUGE 与 CMMLU 上同时全面超过原始 QLoRA 的组合。

### 11.4 `run_qlora_sweep_v2.py`

该脚本用于：

- 对已有的 `alpha24`、`alpha32_dropout005` 复用 adapter 并补最终评测；
- 扩展更窄区间的 learning-rate sweep；
- 自动生成配置、训练、推理、评测和汇总。

示例：

```bash
python scripts/run_qlora_sweep_v2.py \
  --only alpha24 alpha32_dropout005 lr6p5e5
```

只补已有实验最终评测：

```bash
python scripts/run_qlora_sweep_v2.py \
  --only alpha24 alpha32_dropout005
```

---

## 12. 结果文件说明

主实验主要结果：

```text
eval/medical_base_rouge.json
eval/medical_lora_rouge.json
eval/medical_rslora_rouge.json
eval/medical_qoft_rouge.json

eval/cmmlu_base_metrics.json
eval/cmmlu_lora_metrics.json
eval/cmmlu_rslora_metrics.json
eval/cmmlu_qoft_metrics.json
```

CMMLU 错例：

```text
eval/cmmlu_*_wrong.jsonl
```

Sweep 汇总：

```text
eval/qlora_sweep_summary.json
eval/qlora_sweep_summary.csv

eval/qlora_alpha_sweep_summary.json
eval/qlora_alpha_sweep_summary.csv

eval/qlora_alpha24_lr_sweep_summary.json
eval/qlora_alpha24_lr_sweep_summary.csv
```

训练目录中的常见文件：

```text
adapter_config.json
adapter_model.safetensors
trainer_state.json
train_results.json
eval_results.json          # 启用了验证的 sweep 实验
training_loss.png
checkpoint-50/
checkpoint-100/
```

不同实验目录不一定同时包含上述全部文件，以实际运行方式和 LLaMA-Factory 输出为准。

---

## 13. 从零复现实验的最短流程

```bash
# 1. 安装依赖
python -m pip install -r requirements.txt

# 2. 下载基座模型
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/Qwen3-4B-Instruct-2507

# 3. 准备训练 / 验证数据
python scripts/prepare_sharegpt.py \
  --src data/raw/DISC-Med-SFT/DISC-Med-SFT_released.jsonl \
  --out-dir data/processed \
  --train-n 20000 \
  --val-n 1000 \
  --subset-n 2000

# 4. 准备医疗评测集
python scripts/prepare_medical.py \
  --src data/raw/Chinese-medical-dialogue/data/train_0001_of_0001.json \
  --out-dir eval \
  --n 100

# 5. 准备 CMMLU
python scripts/prepare_cmmlu.py \
  --src data/raw/computer_science.csv \
  --out-dir eval

# 6. 训练 QLoRA
llamafactory-cli train configs/qwen3_4b_qlora_align100.yaml

# 7. 医疗评测：生成
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qlora-align100/checkpoint-100 \
  --input eval/medical_100.jsonl \
  --output eval/medical_100_lora.jsonl \
  --max-new-tokens 256 \
  --batch-size 4

# 8. 医疗评测：评分
python scripts/evaluate_medical.py \
  --pred eval/medical_100_lora.jsonl \
  --save eval/medical_lora_rouge.json

# 9. CMMLU：生成
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qlora-align100/checkpoint-100 \
  --input eval/cmmlu_computer_science.jsonl \
  --output eval/cmmlu_lora.jsonl \
  --max-new-tokens 32 \
  --batch-size 4 \
  --system-prompt "你是一名计算机科学考试助手。只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。"

# 10. CMMLU：评分
python scripts/evaluate_cmmlu.py \
  --pred eval/cmmlu_lora.jsonl \
  --save eval/cmmlu_lora_metrics.json \
  --wrong-file eval/cmmlu_lora_wrong.jsonl
```

---

## 14. 复现实验时的注意事项

1. **生成和评分是两步。** `gen_peft.py` 负责加载模型并生成 `pred`；`evaluate_medical.py` / `evaluate_cmmlu.py` 只读取生成结果并计算指标。
2. **从仓库根目录运行脚本。** Sweep 脚本使用相对路径定位 `configs/`、`runs/`、`eval/` 和 `models/`。
3. **不要把测试集用于训练。** `medical_100.jsonl` 与 `cmmlu_computer_science.jsonl` 只用于最终测试。
4. **大文件不一定随仓库完整分发。** 原始 DISC-Med-SFT、Chinese-medical-dialogue 和部分基座模型权重已在 `.gitignore` 中排除，需要本地补齐。
5. **8 GB 显存机器建议从 `--batch-size 4` 或更小开始。** 如果推理 OOM，可降到 2 或 1；这不会改变评测文件格式。
6. **Sweep 脚本默认会复用已有完整 adapter。** 如需强制重新训练，使用对应脚本的 `--force-train`。

---

## 15. 实验结论

在当前数据规模与 100-step 训练预算下：

- 参数高效微调能够显著提高 Qwen3-4B 在中文医疗问答上的 ROUGE 指标；
- QLoRA 在本实验中取得最高 ROUGE-1；
- RS-LoRA 取得最高 ROUGE-2 / ROUGE-L，但对 CMMLU 的影响最大；
- QOFT 在医疗指标提升与 CMMLU 保持之间表现更均衡；
- QLoRA 的 alpha / learning-rate sweep 显示，验证 loss 降低并不保证医疗 ROUGE 与通用能力同时提高；
- 因此超参数选择应同时观察领域任务指标和通用能力指标，而不能只按训练 loss 或 eval loss 排序。

---

## 16. 实验材料

仓库根目录保留：

```text
实验指导书III-本地版.docx
实验指导书III-本地版.pdf
```

可结合指导书中的实验要求、仓库配置文件与 `eval/` 中的已落盘结果复现实验。
