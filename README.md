# 实验 III：Qwen3-4B 中文医疗对话 QLoRA 本地微调

本仓库用于完成《自然语言处理 C》实验 III：**利用对话数据微调基础大模型——基于 Qwen3-4B-Instruct-2507 的中文医疗对话 QLoRA 本地微调**。

实验以 **Qwen3-4B-Instruct-2507** 为基座模型，使用 **LLaMA-Factory** 完成 4-bit QLoRA 微调，并使用 **Transformers + PEFT** 对微调前后的模型进行统一推理。

实验包括：

- DISC-Med-SFT 医疗对话数据处理；
- 对话级训练集 / 验证集划分；
- 多轮对话单轮化与 ShareGPT 格式转换；
- Qwen3-4B 4-bit NF4 QLoRA 微调；
- Base 与 QLoRA 模型医疗问答对比；
- Chinese-medical-dialogue ROUGE-1 / ROUGE-2 / ROUGE-L 评测；
- CMMLU Computer Science 204 道题通用能力评测；
- 追加 RS-LoRA 对比实验。

---

## 1. 实验环境

本次实验实际使用环境如下：

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

主要 Python 依赖见：

```text
requirements.txt
```

本实验使用的主要依赖包括：

```text
torch
transformers
peft
accelerate
bitsandbytes
LLaMA-Factory
jieba
rouge-chinese
```

其中 LLaMA-Factory 固定到 Git commit：

```text
47d407d2aabbd52e11d4f34c7a06b9de3e2f0a77
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

RTX 5070 Laptop GPU 的 CUDA Compute Capability 为 12.0（sm120），因此本实验使用 CUDA 12.8 构建的 PyTorch。

---

## 2. 项目结构

```text
.
├── configs/
│   ├── qwen3_4b_qlora_align100.yaml
│   └── qwen3_4b_qrslora_align100.yaml
│
├── data/
│   ├── raw/
│   │   └── computer_science.csv
│   └── processed/
│       ├── dataset_info.json
│       ├── train_2k_sharegpt.json
│       └── val_1k_sharegpt.json
│
├── eval/
│   ├── medical_100.jsonl
│   ├── cmmlu_computer_science.jsonl
│   │
│   ├── medical_100_base.jsonl
│   ├── medical_100_lora.jsonl
│   ├── medical_100_rslora.jsonl
│   │
│   ├── medical_base_rouge.json
│   ├── medical_lora_rouge.json
│   ├── medical_rslora_rouge.json
│   │
│   ├── cmmlu_base.jsonl
│   ├── cmmlu_lora.jsonl
│   ├── cmmlu_rslora.jsonl
│   │
│   ├── cmmlu_base_metrics.json
│   ├── cmmlu_lora_metrics.json
│   ├── cmmlu_rslora_metrics.json
│   │
│   ├── cmmlu_base_wrong.jsonl
│   ├── cmmlu_lora_wrong.jsonl
│   └── cmmlu_rslora_wrong.jsonl
│
├── models/
│   └── Qwen3-4B-Instruct-2507/
│
├── runs/
│   ├── qwen3-4b-qlora-align100/
│   │   ├── checkpoint-50/
│   │   └── checkpoint-100/
│   └── qwen3-4b-qrslora-align100/
│       ├── checkpoint-50/
│       └── checkpoint-100/
│
├── scripts/
│   ├── prepare_sharegpt.py
│   ├── prepare_medical.py
│   ├── prepare_cmmlu.py
│   ├── gen_peft.py
│   ├── evaluate_medical.py
│   └── evaluate_cmmlu.py
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

下载模型：

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/Qwen3-4B-Instruct-2507
```

模型本地路径：

```text
models/Qwen3-4B-Instruct-2507
```

完整模型目录应包含：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
```

由于基座模型权重文件体积较大，本 GitHub 仓库未完整保存所有 `.safetensors` 分片，完整模型可使用上述命令重新下载。

---

## 4. 数据准备

### 4.1 DISC-Med-SFT 训练与验证数据

医疗训练数据使用：

```text
DISC-Med-SFT
```

原始数据下载后放置于：

```text
data/raw/DISC-Med-SFT/DISC-Med-SFT_released.jsonl
```

运行：

```bash
python scripts/prepare_sharegpt.py \
  --src data/raw/DISC-Med-SFT/DISC-Med-SFT_released.jsonl \
  --out-dir data/processed \
  --train-n 20000 \
  --val-n 1000 \
  --subset-n 2000
```

脚本主要完成以下处理：

1. 过滤无效或轮数不足的对话；
2. 按完整对话进行 9:1 训练 / 验证划分；
3. 将多轮医疗对话转换为单轮样本；
4. 每条医生回复与紧邻的前一条患者提问构成一条训练样本；
5. 转换为 LLaMA-Factory 所需 ShareGPT 格式；
6. 从训练池中抽取 2,000 条用于本实验的 100 步训练；
7. 输出 1,000 条验证样本；
8. 自动生成 `dataset_info.json`。

输出：

```text
data/processed/train_2k_sharegpt.json
data/processed/val_1k_sharegpt.json
data/processed/dataset_info.json
```

实际数据规模：

```text
train_2k_sharegpt.json: 2000 条
val_1k_sharegpt.json:   1000 条
```

单条 ShareGPT 数据格式：

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "患者提问"
    },
    {
      "from": "gpt",
      "value": "医生回复"
    }
  ]
}
```

---

### 4.2 Chinese-medical-dialogue 测试集

医疗对话评测使用：

```text
Chinese-medical-dialogue
```

按照实验指导书要求取前 100 条有效数据。

运行：

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

共：

```text
100 条
```

测试数据不参与模型训练。

---

### 4.3 CMMLU Computer Science 测试集

通用知识能力评测使用：

```text
CMMLU computer_science
```

运行：

```bash
python scripts/prepare_cmmlu.py \
  --src data/raw/computer_science.csv \
  --out-dir eval
```

输出：

```text
eval/cmmlu_computer_science.jsonl
```

实际题目数量：

```text
204
```

每道题由题干、A/B/C/D 四个选项以及标准答案组成。

CMMLU 测试集不参与训练。

---

## 5. QLoRA 主实验

### 5.1 训练配置

QLoRA 配置文件：

```text
configs/qwen3_4b_qlora_align100.yaml
```

主要参数如下：

| 参数 | 取值 |
|---|---:|
| Base Model | Qwen3-4B-Instruct-2507 |
| Quantization | 4-bit |
| Quantization Method | bitsandbytes |
| Quantization Type | NF4 |
| Double Quantization | true |
| Fine-tuning | LoRA |
| LoRA Target | all |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.0 |
| cutoff_len | 1024 |
| Training Samples | 2000 |
| Batch Size | 1 |
| Gradient Accumulation | 16 |
| Effective Batch Size | 16 |
| Learning Rate | 1e-4 |
| Max Steps | 100 |
| Warmup Steps | 20 |
| LR Scheduler | cosine |
| Optimizer | paged_adamw_8bit |
| BF16 | true |
| FP16 | false |
| Gradient Checkpointing | true |
| Seed | 23 |
| Save Steps | 50 |

训练时冻结 4-bit 量化后的基座模型参数，仅训练 LoRA 低秩适配器。

---

### 5.2 启动训练

```bash
llamafactory-cli train configs/qwen3_4b_qlora_align100.yaml
```

训练输出目录：

```text
runs/qwen3-4b-qlora-align100
```

保存：

```text
checkpoint-50
checkpoint-100
```

最终实验统一使用：

```text
runs/qwen3-4b-qlora-align100/checkpoint-100
```

训练结果：

```text
Total optimization steps = 100
Epoch = 0.8
Train loss = 1.7281
Training runtime ≈ 535.3 s
```

约为：

```text
8 min 55 s
```

训练损失曲线：

```text
runs/qwen3-4b-qlora-align100/training_loss.png
```

训练日志：

```text
runs/qwen3-4b-qlora-align100/trainer_log.jsonl
runs/qwen3-4b-qlora-align100/trainer_state.json
runs/qwen3-4b-qlora-align100/train_results.json
```

---

## 6. Base 与 QLoRA 医疗问答评测

Base 和 QLoRA 使用相同的：

- Qwen3-4B 4-bit NF4 基座；
- 测试数据；
- Chat Template；
- 生成参数。

区别仅为是否加载 LoRA adapter。

### 6.1 Base

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter none \
  --input eval/medical_100.jsonl \
  --output eval/medical_100_base.jsonl \
  --max-new-tokens 256 \
  --batch-size 4
```

评测：

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

### 6.2 QLoRA

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qlora-align100/checkpoint-100 \
  --input eval/medical_100.jsonl \
  --output eval/medical_100_lora.jsonl \
  --max-new-tokens 256 \
  --batch-size 4
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

## 7. CMMLU Computer Science 统一评测

按照实验指导书要求，Base、QLoRA 和追加实验 RS-LoRA 均使用完全相同的 CMMLU 推理设置。

统一 System Prompt：

```text
你是一名计算机科学考试助手。只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。
```

统一生成参数：

```text
max_new_tokens = 32
batch_size = 4
do_sample = false
```

`gen_peft.py` 使用贪心生成，因此三组模型的推理条件保持一致。

---

### 7.1 Base

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter none \
  --input eval/cmmlu_computer_science.jsonl \
  --output eval/cmmlu_base.jsonl \
  --max-new-tokens 32 \
  --batch-size 4 \
  --system-prompt "你是一名计算机科学考试助手。只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。"
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
Total:    204
Correct:  163
Wrong:    41
Unparsed: 0
Accuracy: 79.90%
```

---

### 7.2 QLoRA

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qlora-align100/checkpoint-100 \
  --input eval/cmmlu_computer_science.jsonl \
  --output eval/cmmlu_lora.jsonl \
  --max-new-tokens 32 \
  --batch-size 4 \
  --system-prompt "你是一名计算机科学考试助手。只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。"
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
Total:    204
Correct:  163
Wrong:    41
Unparsed: 0
Accuracy: 79.90%
```

---

## 8. QLoRA 主实验结果比较

| 模型 | ROUGE-1 | ROUGE-2 | ROUGE-L | CMMLU |
|---|---:|---:|---:|---:|
| Base Qwen3-4B | 0.1578 | 0.0131 | 0.1062 | 79.90% |
| QLoRA | **0.2252** | **0.0391** | **0.1719** | **79.90%** |

相对于 Base：

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

CMMLU Computer Science：

```text
Base:  79.90%
QLoRA: 79.90%
```

因此，本次 QLoRA 主实验表现为：

```text
医疗领域生成能力明显提升
+
CMMLU Computer Science 总体准确率保持不变
```

说明在当前数据量、训练步数和超参数设置下，医疗领域 QLoRA 微调明显改善了模型在医疗问答上的文本匹配表现，同时没有观察到 CMMLU Computer Science 总体正确率的下降。

需要注意的是，Base 与 QLoRA 虽然最终正确题数均为 163 / 204，但部分具体题目的模型决策可能发生变化，因此相同的总准确率并不代表两者输出完全相同。

---

## 9. 追加实验：RS-LoRA

在普通 QLoRA 基础上，本实验进一步进行了 RS-LoRA 对比实验。

配置文件：

```text
configs/qwen3_4b_qrslora_align100.yaml
```

RS-LoRA 与普通 QLoRA 保持以下条件一致：

```text
Base Model
Training Data
Training Samples
LoRA Rank
LoRA Alpha
Learning Rate
Batch Size
Gradient Accumulation
Training Steps
Warmup Steps
Seed
Quantization
```

主要区别为：

```yaml
use_rslora: true
```

因此该实验用于比较普通 LoRA 与 Rank-Stabilized LoRA 在相同训练条件下的表现。

---

### 9.1 RS-LoRA 训练

```bash
llamafactory-cli train configs/qwen3_4b_qrslora_align100.yaml
```

输出目录：

```text
runs/qwen3-4b-qrslora-align100
```

最终使用：

```text
runs/qwen3-4b-qrslora-align100/checkpoint-100
```

训练结果：

```text
Total optimization steps = 100
Epoch = 0.8
Train loss = 1.6248
Training runtime ≈ 535.1 s
```

普通 QLoRA：

```text
Train loss = 1.7281
```

RS-LoRA：

```text
Train loss = 1.6248
```

在相同的 100 步训练条件下，RS-LoRA 得到了更低的训练平均 loss。

---

### 9.2 RS-LoRA 医疗问答评测

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qrslora-align100/checkpoint-100 \
  --input eval/medical_100.jsonl \
  --output eval/medical_100_rslora.jsonl \
  --max-new-tokens 256 \
  --batch-size 4
```

评测：

```bash
python scripts/evaluate_medical.py \
  --pred eval/medical_100_rslora.jsonl \
  --save eval/medical_rslora_rouge.json
```

结果：

```text
ROUGE-1 F1 = 0.2227
ROUGE-2 F1 = 0.0410
ROUGE-L F1 = 0.1804
```

---

### 9.3 RS-LoRA CMMLU 评测

使用与 Base、QLoRA 完全相同的 System Prompt 与解码参数：

```bash
python scripts/gen_peft.py \
  --model models/Qwen3-4B-Instruct-2507 \
  --adapter runs/qwen3-4b-qrslora-align100/checkpoint-100 \
  --input eval/cmmlu_computer_science.jsonl \
  --output eval/cmmlu_rslora.jsonl \
  --max-new-tokens 32 \
  --batch-size 4 \
  --system-prompt "你是一名计算机科学考试助手。只输出正确选项的字母（A/B/C/D），不要解释，不要重复题干。"
```

评测：

```bash
python scripts/evaluate_cmmlu.py \
  --pred eval/cmmlu_rslora.jsonl \
  --save eval/cmmlu_rslora_metrics.json \
  --wrong-file eval/cmmlu_rslora_wrong.jsonl
```

最终结果：

```text
Total:    204
Correct:  153
Wrong:    51
Unparsed: 0
Accuracy: 75.00%
```

---

## 10. Base / QLoRA / RS-LoRA 总体比较

| 模型 | ROUGE-1 | ROUGE-2 | ROUGE-L | CMMLU |
|---|---:|---:|---:|---:|
| Base Qwen3-4B | 0.1578 | 0.0131 | 0.1062 | **79.90%** |
| QLoRA | **0.2252** | 0.0391 | 0.1719 | **79.90%** |
| RS-LoRA | 0.2227 | **0.0410** | **0.1804** | 75.00% |

从医疗测试结果看：

- 普通 QLoRA 的 ROUGE-1 最高；
- RS-LoRA 的 ROUGE-2 和 ROUGE-L 略高于普通 QLoRA；
- 两种 LoRA 方法均显著优于 Base。

从 CMMLU Computer Science 看：

```text
Base:     79.90%
QLoRA:    79.90%
RS-LoRA:  75.00%
```

普通 QLoRA 在提高医疗问答指标的同时保持了基座模型的 CMMLU 总体正确率。

RS-LoRA 虽然取得了较低的训练 loss，并在 ROUGE-2、ROUGE-L 上略高于普通 QLoRA，但 CMMLU Computer Science 正确率下降至 75.00%。

因此，当前实验结果表明：

```text
更低的训练 loss 并不必然对应所有下游任务指标同时提升。
```

在本实验设置下，普通 QLoRA 在医疗领域提升与通用能力保持之间表现得更加均衡。

RS-LoRA 在部分医疗生成指标上进一步改善，但也表现出更明显的通用知识能力下降。

---

## 11. 推理实现说明

批量推理脚本：

```text
scripts/gen_peft.py
```

推理时使用：

```text
4-bit Qwen3-4B-Instruct-2507
+
可选 PEFT adapter
```

Base：

```text
adapter = none
```

QLoRA：

```text
adapter = runs/qwen3-4b-qlora-align100/checkpoint-100
```

RS-LoRA：

```text
adapter = runs/qwen3-4b-qrslora-align100/checkpoint-100
```

模型生成采用：

```text
do_sample = false
```

即贪心解码。

Qwen3 Chat Template 调用时关闭 thinking：

```text
enable_thinking = false
```

从而保证 Base、QLoRA、RS-LoRA 使用统一的非思考模式完成评测。

---

## 12. 评测方法

### 12.1 医疗问答

评测脚本：

```text
scripts/evaluate_medical.py
```

模型预测回答和参考答案首先使用 `jieba` 分词，再通过：

```text
rouge-chinese
```

计算：

```text
ROUGE-1 F1
ROUGE-2 F1
ROUGE-L F1
```

最终报告 100 条医疗问答样本的平均指标。

---

### 12.2 CMMLU

评测脚本：

```text
scripts/evaluate_cmmlu.py
```

从模型输出中解析：

```text
A
B
C
D
```

并与标准答案比较。

统计：

```text
total
correct
wrong
unparsed
accuracy
```

错误样本同时保存至：

```text
*_wrong.jsonl
```

用于后续实验结果分析。

---

## 13. 实验结果文件

### Base

```text
eval/medical_100_base.jsonl
eval/medical_base_rouge.json

eval/cmmlu_base.jsonl
eval/cmmlu_base_metrics.json
eval/cmmlu_base_wrong.jsonl
```

### QLoRA

```text
eval/medical_100_lora.jsonl
eval/medical_lora_rouge.json

eval/cmmlu_lora.jsonl
eval/cmmlu_lora_metrics.json
eval/cmmlu_lora_wrong.jsonl
```

### RS-LoRA

```text
eval/medical_100_rslora.jsonl
eval/medical_rslora_rouge.json

eval/cmmlu_rslora.jsonl
eval/cmmlu_rslora_metrics.json
eval/cmmlu_rslora_wrong.jsonl
```

---

## 14. 大文件说明

以下文件体积较大，因此没有完整保存在 GitHub 中：

```text
Qwen3-4B-Instruct-2507 完整模型权重

DISC-Med-SFT 原始训练数据

Chinese-medical-dialogue 原始数据
```

这些文件均可通过数据集或模型下载命令重新获取。

仓库保留：

```text
训练配置
数据处理脚本
处理后的训练 / 验证数据
评测测试集
批量推理脚本
评测脚本
模型生成结果
评测指标
错误样本
训练日志
训练曲线
LoRA adapter
RS-LoRA adapter
环境依赖
```

用于保证实验过程及主要结果可以复现。

---

## 15. 实验流程

完整实验流程如下：

```text
环境配置
    ↓
下载 Qwen3-4B-Instruct-2507
    ↓
准备 DISC-Med-SFT
    ↓
对话级 9:1 切分
    ↓
多轮对话单轮化
    ↓
生成 ShareGPT 数据
    ↓
准备 Chinese-medical-dialogue / CMMLU
    ↓
Base 模型评测
    ↓
100 步 QLoRA 训练
    ↓
加载 checkpoint-100
    ↓
QLoRA 医疗 / CMMLU 评测
    ↓
RS-LoRA 追加实验
    ↓
统一指标比较与结果分析
```

本仓库中的实验报告将基于上述训练日志、推理结果、评测指标及实际生成案例进一步分析 QLoRA 微调前后的医疗领域能力与通用知识能力变化。
