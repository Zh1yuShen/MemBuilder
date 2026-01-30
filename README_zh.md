# MemBuilder

[English](README.md) | 中文版

**基于归因密集奖励的长期记忆构建强化学习框架**

## MemBuilder是什么？

MemBuilder训练大语言模型从对话中构建**多维度长期记忆**。使用**ADRPO**（归因密集奖励策略优化）解决两个问题：
- 稀疏奖励 → 密集的会话级QA奖励
- 多智能体归因 → 贡献感知的梯度加权

## 记忆架构

| 类型 | 用途 | 操作 | 示例 |
|------|------|------|------|
| **Core** | 用户画像（持久） | APPEND, REPLACE, REWRITE | "姓名：小明。职业：工程师。" |
| **Episodic** | 带时间戳的事件 | ADD, UPDATE, MERGE | "2024-03-15: 升职了" |
| **Semantic** | 实体相关事实 | ADD, UPDATE, SKIP | "Rust - 用户最喜欢的语言" |
| **Procedural** | 偏好和工作流 | ADD, UPDATE | "喜欢简短回复" |

## 快速开始

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

```python
from memory_system import MemorySystem
from llm_client import OpenAIClient

memory = MemorySystem(llm_client=OpenAIClient(model="gpt-4"))
memory.add([{"role": "user", "content": "我在谷歌工作"}], user_id="u1")
print(memory.generate_answer("我在哪里工作？", user_id="u1"))
```

---

## 训练流程

### 数据生成流程

SFT和RL训练共用同一个专家轨迹生成代码：

```
generate_expert_trajectories.py (共用)
         │
         │  --dataset locomo/longmemeval/perltqa
         │
         ▼
   expert_trajectories/{dataset}/{conv_id}/
         │
    ┌────┴────┐
    ▼         ▼
 SFT数据    RL数据
 (JSON)    (Parquet)
```

### 阶段1：SFT（监督微调）

**目标**：使用LLaMA-Factory用专家行为初始化模型。

```bash
# 1. 生成专家轨迹（使用claude-4.5-sonnet）
python scripts/generate_expert_trajectories.py \
    --dataset longmemeval --output-dir expert_trajectories/longmemeval

# 2. 转换为LLaMA-Factory格式
python scripts/convert_trajectories_to_sft.py \
    --trajectory-dir expert_trajectories/longmemeval \
    --output-file /path/to/LLaMA-Factory/data/memory_building_sft.json

# 3. 在LLaMA-Factory/data/dataset_info.json中注册数据集：
#    "memory_building_sft": {"file_name": "memory_building_sft.json"}

# 4. 训练
cd /path/to/LLaMA-Factory
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen3-4B --stage sft --do_train \
    --dataset memory_building_sft --template qwen3 \
    --cutoff_len 20000 --output_dir saves/membuilder-sft \
    --learning_rate 5e-7 --num_train_epochs 3 --bf16 \
    --deepspeed ds_z2_config.json
```

**SFT数据格式** (`data/sft_example.json`)：
```json
{"instruction": "You are the Core Memory Manager...", "input": "", "output": "```json\n{\"operation\": \"APPEND\", \"content\": \"...\"}\n```"}
```

### 阶段2：ADRPO（veRL强化学习）

**目标**：通过归因密集奖励优化记忆构建。

```bash
# 1. 准备RL数据
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

```bibtex
@article{membuilder2025,
  title={MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Reward},
  author={Anonymous},
  year={2025}
}
```

## 许可证

MIT
