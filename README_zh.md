# MemBuilder

[English](README.md) | 中文版 | [📄 论文](https://arxiv.org/abs/2601.05488)

**基于归因密集奖励的大模型长期记忆构建强化学习框架**

## MemBuilder是什么？

MemBuilder训练大语言模型从对话中构建**多维度长期记忆**。使用**ADRPO**（归因密集奖励策略优化）解决两个关键挑战：
- **稀疏轨迹级奖励**：通过合成会话级QA提供密集的中间奖励信号
- **多维记忆归因**：引入基于下游影响的贡献感知梯度加权机制

## 记忆架构

| 类型 | 存储内容 | 操作 | 示例 |
|------|----------|------|------|
| **Core** | 用户基本信息（持久化） | APPEND, REPLACE, REWRITE | "姓名：小明。职业：工程师。" |
| **Episodic** | 时间相关的事件记忆 | ADD, UPDATE, MERGE | "2024-03-15: 升职了" |
| **Semantic** | 实体和概念的知识 | ADD, UPDATE, SKIP | "Rust - 用户最喜欢的语言" |
| **Procedural** | 步骤化流程和工作习惯 | ADD, UPDATE | "晨间流程：1. 咖啡 2. 邮件 3. 站会" |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 安装LLaMA-Factory（用于SFT训练）
pip install llamafactory

# 安装veRL（用于RL训练）
pip install verl

# 设置API密钥
export OPENAI_API_KEY="your-key"
```

---

## 评测（先试试看）

在训练自己的模型之前，我们建议先使用强大的API模型（如Claude 4.5 Sonnet）尝试记忆构建框架，了解它的工作原理。

我们在三个基准数据集上评测：**LoCoMo**、**LongMemEval** 和 **PerLTQA**。

### 快速测试（单样本）

```bash
# 测试单个LoCoMo对话
python -m eval.runner --dataset locomo --conv-id conv-26 \
    --model claude-sonnet-4-5 --judge-model gpt-4.1

# 测试单个LongMemEval样本
python -m eval.runner --dataset longmemeval --sample-id e47becba \
    --model claude-sonnet-4-5 --judge-model gpt-4.1

# 测试单个PerLTQA人物
python -m eval.runner --dataset perltqa --character-id char_000 \
    --model claude-sonnet-4-5 --judge-model gpt-4.1
```

### 全量基准测试

```bash
# LoCoMo：全部10个对话（1,986个问题）
python -m eval.runner --dataset locomo --model claude-sonnet-4-5 --judge-model gpt-4.1

# LongMemEval：400个隔离测试样本（未用于训练）
python -m eval.runner --dataset longmemeval \
    --split test \
    --model claude-sonnet-4-5 --judge-model gpt-4.1

# PerLTQA：全部31个主角（8,316个问题）
python -m eval.runner --dataset perltqa --model claude-sonnet-4-5 --judge-model gpt-4.1
```

**关键选项：**
- `--mode build`：仅构建记忆（保存到磁盘）
- `--mode answer`：仅回答（加载已保存的记忆）
- `--mode full`：构建+回答（默认）
- `--sessions N`：限制前N个会话
- `--questions N`：限制前N个问题
- `--verbose`：显示详细输出

---

## 训练流程

如果你想训练自己的记忆构建模型，请按照以下步骤操作。

### 步骤0：生成专家轨迹

我们使用**LongMemEval**作为唯一训练数据源。数据划分定义在`data/longmemeval/splits/longmemeval_splits.json`中：
- **50个对话**（`sft`分割）用于SFT轨迹收集
- **50个对话**（`rl`分割）用于RL训练（带合成QA对）
- **400个对话**（`test`分割）用于隔离评测

使用Claude 4.5 Sonnet生成记忆构建轨迹：

```bash
# 为SFT生成专家轨迹（50个对话，约2,400个会话）
python scripts/generate_expert_trajectories.py \
    --dataset longmemeval \
    --split sft \
    --output-dir expert_trajectories/longmemeval_sft \
    --expert-model claude-sonnet-4-5

# 为RL生成专家轨迹（另外50个对话）
python scripts/generate_expert_trajectories.py \
    --dataset longmemeval \
    --split rl \
    --output-dir expert_trajectories/longmemeval_rl \
    --expert-model claude-sonnet-4-5

# 输出结构：
# expert_trajectories/{dataset}/{sample_id}/
# ├── states/          # 每个session前的记忆状态快照
# ├── agent_calls.jsonl # 4个agent的调用记录（Core, Episodic, Semantic, Procedural）
# └── metadata.json
```

> **注意**：SFT和RL使用**不同的**对话子集以避免数据泄漏。LoCoMo和PerLTQA作为分布外测试集。

---

### 阶段1：SFT（监督微调）

**目标**：训练模型模仿专家的记忆构建行为。

```bash
# 1. 将SFT轨迹转换为LLaMA-Factory格式
#    （约9,600个样本：2,400个会话 × 4种记忆类型）
python scripts/convert_trajectories_to_sft.py \
    --trajectory-dir expert_trajectories/longmemeval_sft \
    --output-file /path/to/LLaMA-Factory/data/memory_building_sft.json

# 2. 在LLaMA-Factory/data/dataset_info.json中注册数据集
#    添加："memory_building_sft": {"file_name": "memory_building_sft.json"}

# 3. 运行SFT训练
cd /path/to/LLaMA-Factory
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen3-4B --stage sft --do_train \
    --dataset memory_building_sft --template qwen3 \
    --cutoff_len 20000 --output_dir saves/membuilder-sft \
    --learning_rate 5e-7 --num_train_epochs 10 --bf16 \
    --deepspeed ds_z2_config.json
```

---

### 阶段2：ADRPO（Attributed Dense Reward Policy Optimization，归因密集奖励策略优化）

**目标**：使用密集QA奖励和基于归因的梯度加权进一步优化记忆构建。

```bash
# 1. 将RL轨迹转换为veRL格式（包含合成QA对）
#    每个会话生成5个QA对用于密集奖励计算
python scripts/prepare_rl_data.py \
    --trajectories-dir expert_trajectories/longmemeval_rl \
    --output-file data/memory_rl_train.parquet \
    --add-qa --qa-per-session 5

# 2. 启动奖励服务器
cd training/reward_server && ./start_server.sh
# 测试: curl http://localhost:8765/health

# 3. 运行veRL训练
MODEL_PATH=/path/to/sft-model \
TRAIN_DATA=data/memory_rl_train.parquet \
bash scripts/run_memory_grpo_multinode.sh
```

---

## 配置说明

### 奖励配置 (`training/reward_server/reward_config.json`)

```json
{
  "task_reward_mode": "api",           // "api" 或 "local"
  "memory_api_url": "http://localhost:8765",
  "enable_expert_length_penalty": true,
  "expert_length_penalty_weight": 0.8, // 论文中的λ
  "core_length_start_penalty": 150,    // θ_min
  "core_length_max_penalty": 400,      // θ_max
  "other_length_upper_tolerance": 1.3, // γ_u
  "other_length_lower_tolerance": 0.5, // γ_l
  "enable_attribution_weighting": true,
  "attribution_boost_factor": 4.0      // 论文中的α
}
```

### 模型配置 (`config.py`)

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `SFT_EXPERT_MODEL` | claude-sonnet-4-5 | 轨迹生成专家模型 |
| `QA_GENERATION_MODEL` | claude-opus-4-5 | 合成QA专家模型 |
| `ANSWER_MODEL` | gpt-4.1-mini | QA回答模型 |
| `JUDGE_MODEL` | gpt-4.1 | LLM Judge评测模型 |
| `EMBEDDING_MODEL` | text-embedding-3-small | 嵌入模型 |
| `CORE_MEMORY_CHAR_LIMIT` | 5000 | 核心记忆最大字符数 |

---

## 项目结构

```
MemBuilder/
├── config.py              # 所有配置常量
├── llm_client.py          # OpenAI兼容API客户端
├── memory_system.py       # 多维度记忆系统
├── prompts.py             # 智能体提示词模板
├── qa_generator.py        # 合成QA生成
│
├── scripts/
│   ├── generate_expert_trajectories.py  # 生成专家轨迹
│   ├── convert_trajectories_to_sft.py   # 转换为LLaMA-Factory格式
│   ├── prepare_rl_data.py               # 准备RL parquet数据
│   ├── process_locomo.py                # 处理LoCoMo数据集
│   ├── process_perltqa.py               # 处理PerLTQA数据集
│   ├── run_memory_grpo_multinode.sh     # 启动veRL训练
│   ├── convert_verl_to_hf.sh            # 转换veRL检查点为HF格式
│   └── launch_vllm_openai_server.sh     # 使用vLLM部署
│
├── data/
│   └── longmemeval/splits/              # 训练/测试数据划分
│
├── eval/
│   ├── runner.py          # 评测入口
│   ├── datasets.py        # 数据集加载器
│   ├── llm_judge.py       # LLM答案评估
│   └── metrics.py         # 准确率指标计算
│
└── training/
    ├── sft/
    │   ├── train_example.sh             # SFT训练脚本
    │   └── ds_z2_config.json            # DeepSpeed ZeRO-2配置
    ├── rl/
    │   └── adrpo.py                     # ADRPO算法实现
    └── reward_server/
        ├── server.py                    # 奖励API服务器
        ├── reward_function.py           # 奖励计算
        ├── reward_config.json           # 奖励超参数
        └── start_server.sh              # 服务器启动脚本
```

---

## 训练后处理

```bash
# 转换veRL checkpoint为HuggingFace格式（GPU数量需与训练时一致）
bash scripts/convert_verl_to_hf.sh checkpoints/global_step_100 models/hf_model

# 使用vLLM部署
bash scripts/launch_vllm_openai_server.sh models/hf_model 8000 1

# 注意: vLLM不支持embeddings，需单独配置：
export OPENAI_EMBEDDINGS_BASE_URL="https://api.openai.com/v1"
```

## 引用

如果这项工作对您有帮助，请引用：

```bibtex
@misc{shen2026membuilderreinforcingllmslongterm,
      title={MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards}, 
      author={Zhiyu Shen and Ziming Wu and Fuming Lai and Shaobing Lian and Yanghui Rao},
      year={2026},
      eprint={2601.05488},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.05488}
}
```

## 许可证

MIT License

Copyright (c) 2026 The MemBuilder Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
