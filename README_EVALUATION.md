# MemBuilder 评估系统使用指南

## 概述

MemBuilder 评估系统支持三个数据集的记忆构建和问答评估：
- **LOCOMO**: 长对话记忆问答
- **LongMemEval**: 长期记忆评估
- **PerLTQA**: 个人长期问答

系统支持记忆持久化，可以先构建记忆保存到磁盘，然后用不同模型进行问答测试。

---

## 命令行参数详解

### 数据集选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | `locomo` | 数据集类型，可选：`locomo`, `longmemeval`, `perltqa` |
| `--conv-id` | str | `None` | LOCOMO 对话 ID（如 `conv-26`） |
| `--sample-id` | str | `None` | LongMemEval 样本 ID |
| `--character-id` | str | `None` | PerLTQA 人物 ID |
| `--subset-file` | str | `None` | 子集配置文件路径（用于固定子集测试） |

### 运行模式

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | `full` | 运行模式：<br>• `build`: 仅构建记忆并保存<br>• `answer`: 仅回答（加载已保存记忆）<br>• `full`: 构建+回答 |

### 限制选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sessions` | int | `None` | 限制构建的会话数（`None` = 全部会话） |
| `--questions` | int | `None` | 限制测试的问题数（`None` = 全部问题） |

### 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `claude-4.5-sonnet` | 用于记忆构建和问题回答的 LLM 模型 |
| `--judge-model` | str | `gpt-4.1` | 用于评判答案正确性的 LLM 模型 |

**支持的模型名称：**
- **MetaAI Provider**: `gpt-4o`, `gpt-4o-mini`
- **Yunwu Provider**: `claude-sonnet-4-5-20250929`, `gpt-4o-mini`, `gpt-4.1`
- **OpenAI Provider**: 任何 OpenAI 兼容模型
- **vLLM Provider**: 本地部署的模型

### Provider 选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--provider` | str | `metaai` | LLM 提供商，可选：<br>• `metaai`: MetaAI 内部 API<br>• `yunwu`: 云雾 API（支持 Claude）<br>• `openai`: OpenAI 官方 API<br>• `vllm`: 本地 vLLM 服务 |
| `--base-url` | str | `None` | OpenAI API base URL（仅 `openai` provider） |
| `--api-key` | str | `None` | OpenAI API key（仅 `openai` provider） |
| `--vllm-url` | str | `http://localhost:8000/v1` | vLLM 服务器 URL（仅 `vllm` provider） |

### 检索选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--top-k` | int | `10` | 回答问题时检索的记忆数量（来自 `config.QA_ANSWERING_TOP_K`） |

### 持久化选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--db-path` | str | `None` | 自定义数据库路径<br>• 不指定：自动生成路径（推荐）<br>• 指定：使用自定义路径 |
| `--vector-store` | str | `faiss` | 向量存储类型（目前仅支持 `faiss`） |

**自动生成的路径格式：**
```
./{vector_store}_data/{dataset}/{model}/{id}/
```

**示例：**
- LOCOMO: `./faiss_data/locomo/claude-sonnet-4-5-20250929/conv-26/`
- LongMemEval: `./faiss_data/longmemeval/gpt-4o-mini/sample_001/`
- PerLTQA: `./faiss_data/perltqa/gpt-4o/char_alice/`

### 并发选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--concurrency` | int | `1` | LOCOMO 问题回答的并发数（串行=1） |
| `--parallel` | flag | `False` | 启用 LongMemEval 样本并行处理 |
| `--workers` | int | `4` | LongMemEval 并行处理的进程数 |

**注意：**
- 记忆构建时，4个 agent（Core/Episodic/Semantic/Procedural）始终并发执行
- `--concurrency` 仅影响问题回答阶段
- `--parallel` 用于 LongMemEval 多样本并行处理

### 输出选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output-dir` | str | `logs` | 输出目录（结果 JSON 文件保存位置） |
| `--verbose` | flag | `False` | 详细输出（显示每个问题的详细信息） |

---

## 使用示例

### 1. LOCOMO 数据集

#### 示例 1：完整流程（构建+回答）
```bash
# 使用 Claude Sonnet 4.5 构建记忆并回答全部 199 个问题
python -m eval.runner \
  --dataset locomo \
  --conv-id conv-26 \
  --provider yunwu \
  --model claude-sonnet-4-5-20250929 \
  --judge-model gpt-4.1 \
  --mode full
```

#### 示例 2：分步执行（先构建，后回答）
```bash
# 步骤 1：构建记忆（使用 Claude Sonnet）
python -m eval.runner \
  --dataset locomo \
  --conv-id conv-26 \
  --provider yunwu \
  --model claude-sonnet-4-5-20250929 \
  --mode build

# 步骤 2：用不同模型回答（使用 GPT-4o-mini）
python -m eval.runner \
  --dataset locomo \
  --conv-id conv-26 \
  --provider metaai \
  --model gpt-4o-mini \
  --judge-model gpt-4.1 \
  --mode answer \
  --db-path ./faiss_data/locomo/claude-sonnet-4-5-20250929
```

#### 示例 3：快速测试（限制会话和问题数）
```bash
# 仅构建前 3 个会话，回答前 10 个问题
python -m eval.runner \
  --dataset locomo \
  --conv-id conv-26 \
  --provider yunwu \
  --model claude-sonnet-4-5-20250929 \
  --judge-model gpt-4.1 \
  --sessions 3 \
  --questions 10 \
  --mode full \
  --verbose
```

#### 示例 4：并发回答
```bash
# 使用 5 个并发 worker 回答问题
python -m eval.runner \
  --dataset locomo \
  --conv-id conv-26 \
  --provider metaai \
  --model gpt-4o \
  --judge-model gpt-4.1 \
  --mode answer \
  --concurrency 5
```

### 2. LongMemEval 数据集

#### 示例 1：串行处理单个样本
```bash
python -m eval.runner \
  --dataset longmemeval \
  --sample-id sample_001 \
  --provider yunwu \
  --model claude-sonnet-4-5-20250929 \
  --judge-model gpt-4.1 \
  --mode full
```

#### 示例 2：并行处理多个样本
```bash
# 使用 8 个进程并行处理多个样本
python -m eval.runner \
  --dataset longmemeval \
  --provider yunwu \
  --model claude-sonnet-4-5-20250929 \
  --judge-model gpt-4.1 \
  --parallel \
  --workers 8 \
  --mode full
```

#### 示例 3：使用子集配置
```bash
python -m eval.runner \
  --dataset longmemeval \
  --subset-file dataset/longmemeval/longmemeval_s50.json \
  --provider yunwu \
  --model claude-sonnet-4-5-20250929 \
  --judge-model gpt-4.1 \
  --mode full
```

### 3. PerLTQA 数据集

#### 示例 1：测试单个人物
```bash
python -m eval.runner \
  --dataset perltqa \
  --character-id char_alice \
  --provider yunwu \
  --model claude-sonnet-4-5-20250929 \
  --judge-model gpt-4.1 \
  --mode full
```

#### 示例 2：测试全部人物
```bash
python -m eval.runner \
  --dataset perltqa \
  --provider yunwu \
  --model claude-sonnet-4-5-20250929 \
  --judge-model gpt-4.1 \
  --mode full \
  --verbose
```

### 4. 使用本地 vLLM

```bash
# 启动 vLLM 服务（另一个终端）
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/model \
  --port 8000

# 运行评估
python -m eval.runner \
  --dataset locomo \
  --conv-id conv-26 \
  --provider vllm \
  --model your-model-name \
  --vllm-url http://localhost:8000/v1 \
  --judge-model gpt-4.1 \
  --mode full
```

### 5. 使用 OpenAI API

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"

python -m eval.runner \
  --dataset locomo \
  --conv-id conv-26 \
  --provider openai \
  --model gpt-4o \
  --judge-model gpt-4o-mini \
  --mode full
```

---

## 记忆持久化工作流

### 典型工作流

```bash
# 1. 用强模型构建记忆（一次性）
python -m eval.runner \
  --dataset locomo --conv-id conv-26 \
  --provider yunwu \
  --model claude-sonnet-4-5-20250929 \
  --mode build

# 2. 用不同模型测试回答（可多次）
python -m eval.runner \
  --dataset locomo --conv-id conv-26 \
  --provider metaai \
  --model gpt-4o-mini \
  --judge-model gpt-4.1 \
  --mode answer \
  --db-path ./faiss_data/locomo/claude-sonnet-4-5-20250929

# 3. 再用另一个模型测试
python -m eval.runner \
  --dataset locomo --conv-id conv-26 \
  --provider metaai \
  --model gpt-4o \
  --judge-model gpt-4.1 \
  --mode answer \
  --db-path ./faiss_data/locomo/claude-sonnet-4-5-20250929
```

### 持久化文件结构

```
faiss_data/
└── locomo/
    └── claude-sonnet-4-5-20250929/
        └── conv-26/
            ├── index.faiss          # FAISS 向量索引
            ├── payload.json         # 记忆文本和元数据
            └── core_memory.json     # 核心记忆（用户画像）
```

---

## 输出结果

### 结果文件

评估结果保存在 `--output-dir` 指定的目录（默认 `logs/`）：

```
logs/
└── locomo_20260127_162345/
    ├── conv-26_results.json    # 单个对话的详细结果
    └── summary.json            # 汇总统计
```

### 结果格式

**单个对话结果 (`conv-26_results.json`)：**
```json
{
  "conversation_id": "conv-26",
  "questions": 199,
  "correct": 145,
  "accuracy": 72.86,
  "results": [
    {
      "question_id": "conv-26_q1",
      "question": "...",
      "answer": "...",
      "generated_answer": "...",
      "llm_score": 1,
      "search_time": 0.5,
      "generation_time": 2.3,
      "judge_time": 1.2
    },
    ...
  ]
}
```

**汇总结果 (`summary.json`)：**
```json
{
  "timestamp": "20260127_162345",
  "dataset": "locomo",
  "model": "claude-sonnet-4-5-20250929",
  "judge_model": "gpt-4.1",
  "total_conversations": 1,
  "total_questions": 199,
  "total_correct": 145,
  "overall_accuracy": "72.86%",
  "by_conversation": {
    "conv-26": "72.86%"
  }
}
```

---

## 常见问题

### Q1: 如何选择 provider 和 model？

**Provider 选择：**
- `metaai`: 内部 API，支持 `gpt-4o`, `gpt-4o-mini`
- `yunwu`: 云雾 API，支持 Claude 和 GPT 系列
- `openai`: OpenAI 官方 API
- `vllm`: 本地部署模型

**Model 选择：**
- 记忆构建：推荐强模型（`claude-sonnet-4-5-20250929`, `gpt-4o`）
- 问题回答：可用弱模型测试（`gpt-4o-mini`）
- Judge：推荐 `gpt-4.1`（参考老代码默认值）

### Q2: 如何加速评估？

1. **记忆构建阶段**：4个 agent 已自动并发，无需额外配置
2. **问题回答阶段**：
   - LOCOMO: 使用 `--concurrency 5`（或更高）
   - LongMemEval: 使用 `--parallel --workers 8`

### Q3: 如何复用已构建的记忆？

使用 `--mode answer` 和 `--db-path` 参数：

```bash
# 指定已保存记忆的路径
python -m eval.runner \
  --dataset locomo --conv-id conv-26 \
  --mode answer \
  --db-path ./faiss_data/locomo/claude-sonnet-4-5-20250929 \
  --provider metaai \
  --model gpt-4o-mini \
  --judge-model gpt-4.1
```

### Q4: 如何调试？

使用 `--verbose` 查看详细输出：

```bash
python -m eval.runner \
  --dataset locomo --conv-id conv-26 \
  --provider yunwu \
  --model claude-sonnet-4-5-20250929 \
  --judge-model gpt-4.1 \
  --questions 5 \
  --verbose
```

### Q5: 遇到 HTTP 404 错误怎么办？

检查 model 名称是否正确：
- ❌ `--model metaai`（错误，这是 provider 名）
- ✅ `--model gpt-4o-mini`（正确，这是模型名）

---

## 技术细节

### 记忆构建流程

1. **并发执行 4 个 Agent**（`ThreadPoolExecutor`, `max_workers=4`）：
   - Core Agent: 更新用户画像
   - Episodic Agent: 提取时间事件
   - Semantic Agent: 提取实体知识
   - Procedural Agent: 提取流程步骤

2. **向量化存储**：
   - 使用 FAISS 构建向量索引
   - Embedding 模型：`text-embedding-3-small`

3. **持久化**：
   - 保存 FAISS 索引、记忆文本、核心记忆

### 问题回答流程

1. **检索**：从向量数据库检索 top-k 相关记忆
2. **生成**：使用 LLM 根据记忆生成答案
3. **评判**：使用 LLM Judge 评估答案正确性

### 默认配置（来自 `config.py`）

```python
ANSWER_MODEL = "claude-4.5-sonnet"
JUDGE_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-small"
QA_ANSWERING_TOP_K = 10
MEMORY_CONSTRUCTION_TOP_K = 20
CORE_MEMORY_CHAR_LIMIT = 5000
```

---

## 参考

- 数据集路径：`eval/datasets.py`
- LLM 客户端：`llm_client.py`
- 记忆系统：`memory_system.py`
- 评判系统：`eval/llm_judge.py`
