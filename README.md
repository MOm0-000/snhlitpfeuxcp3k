# 实验 III：Qwen3-4B 医疗领域参数高效微调

本仓库用于完成 NLP 实验 III，基于 **Qwen3-4B-Instruct-2507**，使用 LLaMA-Factory 进行 4-bit QLoRA / LoRA 微调，并在医疗问答数据与 CMMLU Computer Science 数据上进行微调前后对比评测。

## 1. 实验环境

本次实验使用的主要环境如下：

```text
OS: Ubuntu
Python: 3.11

GPU: NVIDIA GeForce RTX 5070 Laptop GPU
GPU Memory: 8 GB
NVIDIA Driver: 595.99.02
System CUDA Toolkit: 13.2

PyTorch: 2.11.0+cu128
torch CUDA: 12.8
CUDA Compute Capability: 12.0
BF16: Supported
```

主要 Python 依赖版本见：

```text
requirements.txt
```

其中 LLaMA-Factory 固定到以下 Git commit：

```text
47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

---

## 2. 项目结构

```text
.
├── configs/
│   └── qwen3_4b_qlora_align100.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── dataset_info.json
│       ├── train_2k_sharegpt.json
│       └── val_1k_sharegpt.json
│
├── eval/
│   ├── medical_100.jsonl
│   ├── cmmlu_computer_science.jsonl
│   ├── medical_100_base.jsonl
│   ├── medical_100_lora.jsonl
│   ├── medical_base_rouge.json
│   ├── medical_lora_rouge.json
│   ├── cmmlu_base.jsonl
│   ├── cmmlu_lora.jsonl
│   ├── cmmlu_base_metrics.json
│   ├── cmmlu_lora_metrics.json
│   ├── cmmlu_base_wrong.jsonl
│   └── cmmlu_lora_wrong.jsonl
│
├── models/
│   └── Qwen3-4B-Instruct-2507/
│
├── runs/
│   └── qwen3-4b-qlora-align100/
│
├── scripts/
│   ├── evaluate_cmmlu.py
│   ├── evaluate_medical.py
│   ├── gen_peft.py
│   ├── prepare_cmmlu.py
│   ├── prepare_medical.py
│   └── prepare_sharegpt.py
│
├── requirements.txt
├── 实验指导书III-本地版.docx
└── 实验指导书III-本地版.pdf
```

---

## 3. 基座模型

实验使用：

```text
Qwen/Qwen3-4B-Instruct-2507
```

下载：

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/Qwen3-4B-Instruct-2507
```

由于模型权重较大，GitHub 仓库中不完整保存基座模型的 `.safetensors` 权重文件。

完整模型应自行使用上述命令下载。

配置中使用的本地模型路径为：

```text
models/Qwen3-4B-Instruct-2507
```

---

## 4. 数据准备

### 4.1 医疗训练数据

训练数据来自：

```text
Flmc/DISC-Med-SFT
```

下载：

```bash
hf download Flmc/DISC-Med-SFT \
  DISC-Med-SFT_released.jsonl \
  --repo-type dataset \
  --local-dir data/raw/DISC-Med-SFT
```

转换为 LLaMA-Factory ShareGPT 格式：

```bash
python scripts/prepare_sharegpt.py \
  --src data/raw/DISC-Med-SFT/DISC-Med-SFT_released.jsonl \
  --out-dir data/processed
```

生成：

```text
data/processed/train_2k_sharegpt.json
data/processed/val_1k_sharegpt.json
data/processed/dataset_info.json
```

训练集使用 2000 条样本。

---

### 4.2 医疗评测数据

医疗评测数据来自：

```text
ticoAg/Chinese-medical-dialogue
```

下载：

```bash
hf download ticoAg/Chinese-medical-dialogue \
  data/train_0001_of_0001.json \
  --repo-type dataset \
  --local-dir data/raw/Chinese-medical-dialogue
```

生成 100 条医疗问答评测数据：

```bash
python scripts/prepare_medical.py \
  --src data/raw/Chinese-medical-dialogue/data/train_0001_of_0001.json \
  --out-dir eval \
  --n 100
```

生成：

```text
eval/medical_100.jsonl
```

---

### 4.3 CMMLU Computer Science

使用 CMMLU 的：

```text
computer_science
```

作为非医疗领域能力测试。

生成：

```text
eval/cmmlu_computer_science.jsonl
```

本次评测共：

```text
204
```

道选择题。

---

## 5. 微调前基座评测

### 5.1 医疗问答

生成：

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter none \
  --input eval/medical_100.jsonl \
  --output eval/medical_100_base.jsonl \
  --batch-size 4 \
  --max-new-tokens 256
```

计算 ROUGE：

```bash
python scripts/evaluate_medical.py \
  --pred eval/medical_100_base.jsonl \
  --save eval/medical_base_rouge.json
```

结果：

```text
ROUGE-1 F1 = 0.1578
ROUGE-2 F1 = 0.0131
ROUGE-L F1 = 0.1062
```

---

### 5.2 CMMLU

生成：

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter none \
  --input eval/cmmlu_computer_science.jsonl \
  --output eval/cmmlu_base.jsonl \
  --batch-size 4 \
  --max-new-tokens 16 \
  --system "这是单项选择题。请只输出正确答案的一个字母 A、B、C 或 D，不要解释。"
```

评测：

```bash
python scripts/evaluate_cmmlu.py \
  --pred eval/cmmlu_base.jsonl \
  --save eval/cmmlu_base_metrics.json \
  --wrong-file eval/cmmlu_base_wrong.jsonl
```

结果：

```text
正确：163 / 204
错误：41
未解析：0

Accuracy = 79.90%
```

---

## 6. QLoRA / LoRA 微调

训练配置：

```text
configs/qwen3_4b_qlora_align100.yaml
```

主要参数：

```text
Base Model: Qwen3-4B-Instruct-2507

Quantization:
  4-bit
  bitsandbytes
  NF4
  double quantization

Fine-tuning:
  SFT
  LoRA
  lora_target = all
  rank = 8
  alpha = 16
  dropout = 0.0

Data:
  med_train_2k
  max_samples = 2000
  cutoff_len = 1024

Training:
  batch_size = 1
  gradient_accumulation_steps = 16
  learning_rate = 1e-4
  max_steps = 100
  warmup_steps = 20
  lr_scheduler = cosine
  bf16 = true
  gradient_checkpointing = true
  seed = 23
```

开始训练：

```bash
llamafactory-cli train configs/qwen3_4b_qlora_align100.yaml
```

输出目录：

```text
runs/qwen3-4b-qlora-align100
```

训练结果：

```text
Total optimization steps = 100
Epoch = 0.8
Train loss = 1.7281
Training runtime ≈ 8 min 55 s
```

训练 loss 曲线：

```text
runs/qwen3-4b-qlora-align100/training_loss.png
```

训练日志及状态：

```text
runs/qwen3-4b-qlora-align100/trainer_log.jsonl
runs/qwen3-4b-qlora-align100/trainer_state.json
runs/qwen3-4b-qlora-align100/train_results.json
```

最终 LoRA adapter：

```text
runs/qwen3-4b-qlora-align100/adapter_model.safetensors
runs/qwen3-4b-qlora-align100/adapter_config.json
```

---

## 7. LoRA 微调后评测

### 7.1 医疗问答

生成：

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qlora-align100 \
  --input eval/medical_100.jsonl \
  --output eval/medical_100_lora.jsonl \
  --batch-size 4 \
  --max-new-tokens 256
```

评测：

```bash
python scripts/evaluate_medical.py \
  --pred eval/medical_100_lora.jsonl \
  --save eval/medical_lora_rouge.json
```

结果：

```text
ROUGE-1 F1 = 0.2252
ROUGE-2 F1 = 0.0391
ROUGE-L F1 = 0.1719
```

---

### 7.2 CMMLU

生成：

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qlora-align100 \
  --input eval/cmmlu_computer_science.jsonl \
  --output eval/cmmlu_lora.jsonl \
  --batch-size 4 \
  --max-new-tokens 16 \
  --system "这是单项选择题。请只输出正确答案的一个字母 A、B、C 或 D，不要解释。"
```

评测：

```bash
python scripts/evaluate_cmmlu.py \
  --pred eval/cmmlu_lora.jsonl \
  --save eval/cmmlu_lora_metrics.json \
  --wrong-file eval/cmmlu_lora_wrong.jsonl
```

结果：

```text
正确：163 / 204
错误：41
未解析：0

Accuracy = 79.90%
```

---

## 8. 微调前后结果比较

| 模型 | ROUGE-1 | ROUGE-2 | ROUGE-L | CMMLU |
|---|---:|---:|---:|---:|
| Base Qwen3-4B | 0.1578 | 0.0131 | 0.1062 | 79.90% |
| QLoRA / LoRA | 0.2252 | 0.0391 | 0.1719 | 79.90% |

医疗问答指标变化：

```text
ROUGE-1:
0.1578 -> 0.2252
约提升 42.7%

ROUGE-2:
0.0131 -> 0.0391
约提升 198.5%

ROUGE-L:
0.1062 -> 0.1719
约提升 61.9%
```

结果表明，经过医疗领域数据微调后，模型在医疗问答上的文本匹配指标明显提高。

与此同时：

```text
CMMLU Computer Science:
79.90% -> 79.90%
```

总体准确率保持不变。

虽然准确率相同，但微调前后的具体错题并非完全一致，说明 LoRA 微调确实改变了部分样本上的模型决策，只是最终正确题数仍为 163 / 204。

因此，本轮实验表现为：

```text
医疗领域能力明显提升
+
CMMLU Computer Science 总体准确率基本保持
```

---

## 9. 追加实验：其他 LoRA 变种

除实验指导书中的 QLoRA / LoRA 示例外，还将根据课程追加要求实验其他参数高效微调方法，例如：

```text
RS-LoRA
```

实验时保持基座模型、训练数据、训练步数和主要超参数尽量一致，以便和普通 LoRA 做公平对比。

对应实验结果将在后续补充。

---

## 10. 大文件说明

以下文件由于体积较大，不要求完整保存在 GitHub 中：

```text
Qwen3-4B-Instruct-2507 模型权重

DISC-Med-SFT 原始数据

Chinese-medical-dialogue 原始数据
```

这些文件可通过本文前面的下载命令重新获取。

GitHub 仓库主要保存：

```text
训练配置
数据处理脚本
处理后的实验数据
评测数据
模型生成结果
评测指标
训练日志
训练曲线
LoRA adapter
环境依赖
```

从而保证实验过程和结果能够复现。
