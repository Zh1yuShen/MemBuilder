# MemBuilder

[中文版](README_zh.md) | English | [📄 Paper](https://arxiv.org/abs/2601.05488)

**Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards**

## What is MemBuilder?

MemBuilder trains LLMs to build **multi-dimensional long-term memory** from conversations. It uses **ADRPO** (Attributed Dense Reward Policy Optimization) to solve two key challenges:
- **Sparse Trajectory-Level Rewards**: We employ synthetic session-level QA to provide dense intermediate rewards
- **Multi-Dimensional Memory Attribution**: We introduce contribution-aware gradient weighting based on each component's downstream impact

## Memory Architecture

| Type | Storage Content | Actions | Example |
|------|-----------------|---------|---------|
| **Core** | User basic information (persistent) | APPEND, REPLACE, REWRITE | "Name: Sarah. Job: Engineer." |
| **Episodic** | Time-related event memories | ADD, UPDATE, MERGE | "2024-03-15: Got promoted" |
| **Semantic** | Knowledge about entities and concepts | ADD, UPDATE, SKIP | "Rust - User's favorite language" |
| **Procedural** | Step-by-step processes and workflows | ADD, UPDATE | "Morning routine: 1. Coffee 2. Email 3. Standup" |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install LLaMA-Factory (for SFT training)
pip install llamafactory

# Install veRL (for RL training)
pip install verl

# Set API key
export OPENAI_API_KEY="your-key"
```

---

## Evaluation (Try It First)

Before training your own model, we recommend first trying the memory construction framework with a strong API model (e.g., Claude 4.5 Sonnet) to understand how it works.

We evaluate on three benchmarks: **LoCoMo**, **LongMemEval**, and **PerLTQA**.

### Quick Test (Single Sample)

```bash
# Test single LoCoMo conversation
python -m eval.runner --dataset locomo --conv-id conv-26 \
    --model claude-4.5-sonnet --judge-model gpt-4.1

# Test single LongMemEval sample
python -m eval.runner --dataset longmemeval --sample-id e47becba \
    --model claude-4.5-sonnet --judge-model gpt-4.1

# Test single PerLTQA character
python -m eval.runner --dataset perltqa --character-id char_000 \
    --model claude-4.5-sonnet --judge-model gpt-4.1
```

### Full Benchmark Evaluation

```bash
# LoCoMo: All 10 conversations (1,986 questions)
python -m eval.runner --dataset locomo --model claude-4.5-sonnet --judge-model gpt-4.1

# LongMemEval: 400 held-out test samples (not used in training)
python -m eval.runner --dataset longmemeval \
    --split test \
    --model claude-4.5-sonnet --judge-model gpt-4.1

# PerLTQA: All 31 protagonists (8,316 questions)
python -m eval.runner --dataset perltqa --model claude-4.5-sonnet --judge-model gpt-4.1
```

**All options for `eval/runner.py`:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `locomo` | Dataset: `locomo`, `longmemeval`, or `perltqa` |
| `--conv-id` | | Conversation ID (locomo) |
| `--sample-id` | | Sample ID (longmemeval) |
| `--character-id` | | Character ID (perltqa) |
| `--split` | | Predefined split: `sft`, `rl`, or `test` (longmemeval) |
| `--subset-file` | | Path to JSON file with sample IDs |
| `--mode` | `full` | `build` (memory only), `answer` (QA only), or `full` |
| `--model` | from config | LLM model for memory agents |
| `--judge-model` | from config | LLM model for answer evaluation |
| `--provider` | `openai` | LLM provider (`openai`, `vllm`) |
| `--judge-provider` | same as `--provider` | Judge provider (auto-falls back to API when main is vllm) |
| `--base-url` | | API base URL (default: `http://localhost:8000/v1` for vllm) |
| `--api-key` | | API key (default: `EMPTY` for vllm; or set `OPENAI_API_KEY`) |
| `--sessions N` | all | Limit number of sessions to build |
| `--questions N` | all | Limit number of questions to test |
| `--top-k` | from config | Top-K memories for QA retrieval |
| `--db-path` | auto | Custom database path for memory persistence |
| `--vector-store` | `faiss` | Vector store backend |
| `--concurrency` | `1` | Concurrent workers for answering (locomo) |
| `--parallel` | off | Enable parallel sample processing (longmemeval) |
| `--workers` | `4` | Number of parallel workers |
| `--output-dir` | `logs` | Output directory for results |
| `--verbose` | off | Show detailed output |

**Using vLLM (self-hosted models):**
```bash
python -m eval.runner --dataset longmemeval --split test \
    --provider vllm --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B
```

---

## Training

If you want to train your own memory construction model, follow the steps below.

### Step 0: Generate Expert Trajectories

We use **LongMemEval** as the sole training source. The data splits are defined in `data/longmemeval/splits/longmemeval_splits.json`:
- **50 dialogues** (`sft` split) for SFT trajectory collection
- **50 dialogues** (`rl` split) for RL training (with synthetic QA pairs)
- **400 dialogues** (`test` split) for held-out evaluation

Use Claude 4.5 Sonnet to generate memory construction trajectories:

```bash
# Generate expert trajectories for SFT (50 dialogues, ~2,400 sessions)
python scripts/generate_expert_trajectories.py \
    --dataset longmemeval \
    --split sft \
    --output-dir expert_trajectories/longmemeval_sft \
    --expert-model claude-4.5-sonnet \
    --provider openai \
    --skip-existing

# Generate expert trajectories for RL (50 separate dialogues)
python scripts/generate_expert_trajectories.py \
    --dataset longmemeval \
    --split rl \
    --output-dir expert_trajectories/longmemeval_rl \
    --expert-model claude-4.5-sonnet \
    --provider openai

# Output structure:
# expert_trajectories/{dataset}/{sample_id}/
# ├── states/          # Memory state snapshots before each session
# ├── agent_calls.jsonl # Call records for 4 agents (Core, Episodic, Semantic, Procedural)
# └── metadata.json
```

**All options for `generate_expert_trajectories.py`:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | Dataset name |
| `--dataset-path` | | Path to custom dataset file (JSONL/JSON) |
| `--conv-id` | | Single conversation ID to process |
| `--subset-file` | | JSON file with conversation IDs |
| `--split` | | Predefined split: `sft`, `rl`, or `test` |
| `--expert-model` | from config | Expert model for generation |
| `--output-dir` | `./expert_trajectories` | Output directory |
| `--provider` | `openai` | LLM provider (`openai`, `vllm`) |
| `--parallel` | off | Enable parallel processing |
| `--workers` | `4` | Number of parallel workers |
| `--skip-existing` | off | Skip already-generated trajectories |

> **Note**: SFT and RL use **different** dialogue subsets to avoid data leakage. LoCoMo and PerLTQA serve as out-of-distribution test sets.

---

### Stage 1: SFT (Supervised Fine-Tuning)

**Goal**: Train the model to imitate expert memory construction behavior.

```bash
# 1. Convert SFT trajectories to LLaMA-Factory format
#    (~9,600 samples: 2,400 sessions × 4 memory types)
python scripts/convert_trajectories_to_sft.py \
    --trajectory-dir expert_trajectories/longmemeval_sft \
    --output-file /path/to/LLaMA-Factory/data/memory_building_sft.json \
    --max-length 20000

# 2. Register dataset in LLaMA-Factory/data/dataset_info.json
#    Add: "memory_building_sft": {"file_name": "memory_building_sft.json"}

# 3. Run SFT training
cd /path/to/LLaMA-Factory
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen3-4B --stage sft --do_train \
    --dataset memory_building_sft --template qwen3 \
    --cutoff_len 20000 --output_dir saves/membuilder-sft \
    --learning_rate 5e-7 --num_train_epochs 10 --bf16 \
    --deepspeed ds_z2_config.json
```

---

### Stage 2: ADRPO (Attributed Dense Reward Policy Optimization)

**Goal**: Further optimize memory construction using dense QA rewards with attribution-based gradient weighting.

```bash
# 1. Convert RL trajectories to veRL format (includes synthetic QA pairs)
#    For each session, 5 QA pairs are generated for dense reward computation
python scripts/prepare_rl_data.py \
    --trajectories-dir expert_trajectories/longmemeval_rl \
    --output-file data/memory_rl_train.parquet \
    --add-qa --qa-per-session 5

# 2. Start reward server
cd training/reward_server && ./start_server.sh
# Test: curl http://localhost:8765/health

# 3. Run veRL training
MODEL_PATH=/path/to/sft-model \
TRAIN_DATA=data/memory_rl_train.parquet \
bash scripts/run_memory_grpo_multinode.sh
```

---

## Configuration

### Reward Config (`training/reward_server/reward_config.json`)

```json
{
  "task_reward_mode": "api",           // "api" or "local"
  "memory_api_url": "http://localhost:8765",
  "enable_expert_length_penalty": true,
  "expert_length_penalty_weight": 0.8, // λ in paper
  "core_length_start_penalty": 150,    // θ_min
  "core_length_max_penalty": 400,      // θ_max
  "other_length_upper_tolerance": 1.3, // γ_u
  "other_length_lower_tolerance": 0.5, // γ_l
  "enable_attribution_weighting": true,
  "attribution_boost_factor": 4.0      // α in paper
}
```

### Model Config (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `SFT_EXPERT_MODEL` | claude-4.5-sonnet | Expert for trajectory generation |
| `QA_GENERATION_MODEL` | claude-4.5-opus | Expert for synthetic QA |
| `ANSWER_MODEL` | gpt-4.1-mini | Model for QA answering (during RL reward) |
| `JUDGE_MODEL` | gpt-4.1 | LLM Judge for evaluation |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |
| `CORE_MEMORY_CHAR_LIMIT` | 5000 | Max chars for core memory |

---

## Project Structure

```
MemBuilder/
├── config.py              # All configuration constants
├── llm_client.py          # OpenAI-compatible API client (exports AVAILABLE_PROVIDERS)
├── memory_system.py       # Multi-dimensional memory system
├── prompts.py             # Agent prompt templates
├── qa_generator.py        # Synthetic QA generation
│
├── scripts/
│   ├── generate_expert_trajectories.py  # Generate expert trajectories
│   ├── convert_trajectories_to_sft.py   # Convert to LLaMA-Factory format
│   ├── prepare_rl_data.py               # Prepare RL parquet data
│   ├── process_locomo.py                # Process LoCoMo dataset
│   ├── process_perltqa.py               # Process PerLTQA dataset
│   ├── run_memory_grpo_multinode.sh     # Launch veRL training
│   ├── convert_verl_to_hf.sh            # Convert veRL checkpoint to HF
│   └── launch_vllm_openai_server.sh     # Deploy with vLLM
│
├── data/
│   └── longmemeval/splits/              # Train/test splits
│
├── eval/
│   ├── runner.py          # Evaluation entry point
│   ├── datasets.py        # Dataset loaders
│   ├── llm_judge.py       # LLM-based answer evaluation
│   └── metrics.py         # Accuracy metrics computation
│
└── training/
    ├── sft/
    │   ├── train_example.sh             # SFT training script
    │   └── ds_z2_config.json            # DeepSpeed ZeRO-2 config
    ├── rl/
    │   └── adrpo.py                     # ADRPO algorithm implementation
    └── reward_server/
        ├── server.py                    # Reward API server
        ├── reward_function.py           # Reward computation
        ├── reward_config.json           # Reward hyperparameters
        └── start_server.sh              # Server startup script
```

---

## Post-Training

```bash
# Convert veRL checkpoint to HuggingFace (same GPU count as training)
bash scripts/convert_verl_to_hf.sh checkpoints/global_step_100 models/hf_model

# Deploy with vLLM
bash scripts/launch_vllm_openai_server.sh models/hf_model 8000 1

# Note: vLLM doesn't support embeddings, configure separately:
export OPENAI_EMBEDDINGS_BASE_URL="https://api.openai.com/v1"
```

### Two-Step Evaluation (Recommended)

Since the trained model specializes in **memory construction** but may not excel at **question answering**, we recommend a two-step approach: use your trained model for building memories, then a strong API model for answering.

```bash
# Step 1: Build memories with your vLLM-hosted model
python -m eval.runner \
    --provider vllm \
    --base-url http://localhost:8000/v1 \
    --model models/hf_model \
    --dataset locomo \
    --mode build \
    --db-path ./faiss_data/locomo/my_model

# Step 2: Answer questions with a strong API model (loads Step 1 memories)
python -m eval.runner \
    --provider openai \
    --model gpt-4.1-mini \
    --dataset locomo \
    --mode answer \
    --db-path ./faiss_data/locomo/my_model
```

This separates **memory construction** (your trained model's strength) from **QA answering** (where a general-purpose model performs better), giving the best overall evaluation results.

## Citation

If you find this work helpful, please cite:

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
