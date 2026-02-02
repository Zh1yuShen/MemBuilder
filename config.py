"""Configuration and constants for MemBuilder."""

import os

# =============================================================================
# API Configuration (from environment or defaults)
# =============================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

# =============================================================================
# Model Configuration (Paper Section 4.1 - Implementation Details)
# =============================================================================

# Base model for memory construction
BASE_MODEL = "Qwen3-4B-Instruct-2507"

# Expert model for SFT trajectory collection
SFT_EXPERT_MODEL = "claude-4.5-sonnet"

# Expert model for synthetic QA generation
QA_GENERATION_MODEL = "claude-4.5-opus"

# Answer model for RL reward (Paper F.2: GPT-4.1-mini for both answer generation and judging)
ANSWER_MODEL = "gpt-4.1-mini"

# LLM Judge model for evaluation
JUDGE_MODEL = "gpt-4.1"

# Embedding model for retrieval
EMBEDDING_MODEL = "text-embedding-3-small"

# =============================================================================
# Memory Architecture Configuration (Paper Section 3.2)
# =============================================================================

# Core Memory limits
CORE_MEMORY_CHAR_LIMIT = 5000

# Memory type prefixes for vector database
MEMORY_PREFIXES = {
    "episodic": "[EPISODIC]",
    "semantic": "[SEMANTIC]",
    "procedural": "[PROCEDURAL]"
}

# Agent action spaces (as described in paper Section 3.2)
AGENT_ACTIONS = {
    "core": ["APPEND", "REPLACE", "REWRITE"],
    "episodic": ["ADD", "UPDATE", "MERGE"],
    "semantic": ["ADD", "UPDATE", "SKIP"],
    "procedural": ["ADD", "UPDATE"]
}

# =============================================================================
# Retrieval Settings (Paper Appendix - Embedding and Retrieval)
# =============================================================================

# Memory construction: retrieve top-20 for agent context
MEMORY_CONSTRUCTION_TOP_K = 20

# QA answering: retrieve top-10 for answer generation
QA_ANSWERING_TOP_K = 10

# QA generation: retrieve top-20 for synthetic question context
QA_GENERATION_TOP_K = 20

# RAG baselines: retrieve top-5 chunks
RAG_BASELINE_TOP_K = 5

# Default (for backward compatibility)
DEFAULT_TOP_K = QA_ANSWERING_TOP_K
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL

# =============================================================================
# ADRPO hyperparameters (Paper Appendix - RL Training)
# =============================================================================
ADRPO_CONFIG = {
    "learning_rate": 1e-6,
    "batch_size": 128,
    "num_rollouts": 8,
    "clip_epsilon": 0.2,
    "alpha": 4,  # Contribution-aware gradient weight
    "lambda_length": 0.8,  # Length penalty coefficient
    "theta_min": 150,  # Core memory penalty-free threshold
    "theta_max": 400,  # Core memory full-penalty threshold
    "gamma_l": 0.5,  # Other memory tolerance range lower
    "gamma_u": 1.3,  # Other memory tolerance range upper
}
