# MemBuilder

[English](README.md) | 中文版

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
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

```python
from memory_system import MemorySystem
from llm_client import OpenAIClient

# 初始化记忆系统
memory = MemorySystem(llm_client=OpenAIClient(model="gpt-4"))

# 处理对话并构建记忆
session = [{"role": "user", "content": "我在谷歌工作"}]
memory.process_session(session, user_id="u1")

# 基于记忆回答问题
answer = memory.generate_answer("我在哪里工作？", user_id="u1")
print(answer)  # 输出: 根据记忆，你在谷歌工作。
```

---

## 训练流程

### 步骤0：生成专家轨迹（SFT和RL共用）

使用强专家模型（如Claude 4.5 Sonnet）为每个对话生成记忆构建轨迹。

```bash
# 生成LongMemEval数据集的专家轨迹
python scripts/generate_expert_trajectories.py \
    --dataset longmemeval \
    --output-dir expert_trajectories/longmemeval \
    --expert-model claude-4.5-sonnet

# 输出结构：
# expert_trajectories/longmemeval/{sample_id}/
# ├── states/          # 每个session前的记忆状态快照
# ├── agent_calls.jsonl # 4个agent的调用记录
# └── metadata.json
```

---

### 阶段1：SFT（监督微调）

**目标**：训练模型模仿专家的记忆构建行为。

```bash
# 1. 将专家轨迹转换为LLaMA-Factory格式
python scripts/convert_trajectories_to_sft.py \
    --trajectory-dir expert_trajectories/longmemeval \
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

**SFT数据格式** (`data/sft_example.json`)：
```json
{"instruction": "You are the Core Memory Manager...", "input": "", "output": "```json\n{\"operation\": \"APPEND\", \"content\": \"...\"}\n```"}
```

---

### 阶段2：ADRPO（Attributed Dense Reward Policy Optimization，归因密集奖励策略优化）

**目标**：使用密集QA奖励和基于归因的梯度加权进一步优化记忆构建。

```bash
# 1. 将专家轨迹转换为RL训练数据（包含QA pairs用于奖励计算）
python scripts/prepare_rl_data.py \
    --trajectories-dir expert_trajectories/longmemeval \
    --output-file data/memory_rl_train.parquet

# 2. 启动奖励服务器
cd training/reward_server && ./start_server.sh
# 测试: curl http://localhost:8765/health

# 3. 运行veRL训练
MODEL_PATH=/path/to/sft-model \
TRAIN_DATA=data/memory_rl_train.parquet \
bash scripts/run_memory_grpo_multinode.sh
```

**RL数据格式** (`data/verl_training_example.parquet`)：
| prompt | ability | reward_model | data_source | meta |
|--------|---------|--------------|-------------|------|
| 系统提示 + 消息 | core/episodic/... | membuilder | longmemeval | {questions, expert_actions, ...} |

---

## 配置说明

### 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | 是 | LLM调用的API密钥 |
| `OPENAI_BASE_URL` | 否 | 自定义API端点 |
| `MODEL_PATH` | RL训练 | SFT模型路径 |
| `TRAIN_DATA` | RL训练 | parquet文件路径 |
| `MEMBUILDER_REWARD_CONFIG_PATH` | 否 | 自定义奖励配置 |

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
| `SFT_EXPERT_MODEL` | claude-4.5-sonnet | 轨迹生成专家模型 |
| `QA_GENERATION_MODEL` | claude-4.5-opus | 合成QA专家模型 |
| `ANSWER_MODEL` | gpt-4.1-mini | QA回答模型 |
| `JUDGE_MODEL` | gpt-4.1-mini | LLM Judge评测模型 |
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
├── evaluation.py          # LLM Judge评测
│
├── scripts/
│   ├── generate_expert_trajectories.py  # 步骤1: 生成SFT数据
│   ├── convert_trajectories_to_sft.py   # 步骤2: 转换为LLaMA-Factory格式
│   ├── prepare_rl_data.py               # 步骤3: 准备RL parquet
│   ├── run_memory_grpo_multinode.sh     # 步骤4: 启动veRL
│   ├── convert_verl_to_hf.sh            # 后处理: 转换checkpoint
│   └── launch_vllm_openai_server.sh     # 后处理: vLLM部署
│
├── data/
│   ├── sft_example.json             # SFT格式示例（4种智能体）
│   ├── rl_training_example.jsonl    # RL中间格式
│   ├── verl_training_example.parquet # veRL最终格式
│   └── evaluation_sample.json       # 评测数据格式
│
└── training/
    ├── sft/train_example.sh         # 完整SFT训练脚本
    └── reward_server/
        ├── server.py                # Flask奖励API
        ├── reward_function.py       # 奖励计算
        ├── reward_config.json       # 奖励超参数
        └── start_server.sh          # 服务器启动脚本
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

## 评测

```python
from evaluation import evaluate_answer, compute_accuracy

results = []
for qa in questions:
    answer = memory.generate_answer(qa['question'], user_id=conv_id)
    correct = evaluate_answer(qa['question'], qa['answer'], answer, client)
    results.append({'correct': correct, 'type': qa['type']})

print(compute_accuracy(results))  # {'overall': 0.85, 'single_hop': 0.90, ...}
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

Copyright (c) 2023 MemBuilder Authors

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
