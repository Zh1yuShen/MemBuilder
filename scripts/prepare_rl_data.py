#!/usr/bin/env python3
"""
Prepare RL Training Data for verl Framework.

Converts expert trajectories (agent_calls.jsonl) to verl-compatible parquet format.

Input: expert_trajectories/{dataset}/{sample_id}/agent_calls.jsonl
Output: data/memory_rl_train.parquet

The parquet file contains:
- prompt: Full agent prompt with state and session
- ground_truth: Expert output for reward computation
- ability: Task type marker
- reward_model: Configuration for reward computation

Usage:
    python scripts/prepare_rl_data.py \
        --trajectories-dir expert_trajectories/longmemeval_rl \
        --output-file data/memory_rl_train.parquet \
        --add-qa --qa-per-session 5
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import QA_GENERATION_MODEL
from qa_generator import QAGenerator
from llm_client import OpenAIClient
from memory_system import MemorySystem


class MemoryRLDataPreprocessor:
    """Prepare RL training data for verl."""
    
    def __init__(self):
        pass
    
    def process_trajectories(
        self,
        trajectories_dir: str,
        output_file: str,
        add_qa: bool = False,
        qa_per_session: int = 5
    ):
        """
        Process all trajectory files and create parquet dataset.
        
        Args:
            trajectories_dir: Directory containing trajectory folders
            output_file: Output parquet file path
            add_qa: Whether to add QA pairs for task reward
            qa_per_session: Number of QA pairs per session
        """
        trajectories_path = Path(trajectories_dir)
        
        print(f"Processing trajectories from: {trajectories_path}")
        
        # Load all agent calls and existing QA pairs
        agent_calls = []
        existing_qa_pairs = {}  # (conv_id, session_idx) -> qa_questions
        
        for conv_dir in trajectories_path.iterdir():
            if not conv_dir.is_dir():
                continue
            
            # Load agent calls
            agent_calls_file = conv_dir / "agent_calls.jsonl"
            if not agent_calls_file.exists():
                continue
            
            with open(agent_calls_file) as f:
                for line in f:
                    call = json.loads(line)
                    agent_calls.append(call)
            
            # Load existing QA pairs from rl_training_data.jsonl
            rl_data_file = conv_dir / "rl_training_data.jsonl"
            if rl_data_file.exists():
                with open(rl_data_file) as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            conv_id = data.get("conversation_id", conv_dir.name)
                            sess_idx = data.get("session_index", 0)
                            questions = data.get("questions", [])
                            if questions:
                                existing_qa_pairs[(conv_id, sess_idx)] = questions
                        except Exception:
                            pass
        
        print(f"Loaded {len(existing_qa_pairs)} sessions with existing QA pairs")
        
        print(f"Loaded {len(agent_calls)} agent calls")
        
        # Group by (conversation_id, session_index)
        grouped = defaultdict(list)
        for call in agent_calls:
            key = (call["conversation_id"], call["session_index"])
            grouped[key].append(call)
        
        print(f"Grouped into {len(grouped)} sessions")
        
        # Validate grouping
        valid_groups = []
        for key, calls in grouped.items():
            agent_types = {c["agent_type"] for c in calls}
            if agent_types == {"core", "episodic", "semantic", "procedural"}:
                valid_groups.append((key, calls))
            else:
                print(f"  Invalid group {key}: {agent_types}")
        
        print(f"Valid groups: {len(valid_groups)}")
        
        # Create training records
        records: List[Dict[str, Any]] = []

        qa_generator: Optional[QAGenerator] = None
        qa_llm_client: Optional[OpenAIClient] = None
        if add_qa:
            qa_llm_client = OpenAIClient(model=QA_GENERATION_MODEL)
            qa_generator = QAGenerator(llm_client=qa_llm_client, num_questions=qa_per_session)

        agent_order = ["core", "episodic", "semantic", "procedural"]
        agent_index_map = {a: i for i, a in enumerate(agent_order)}

        def get_full_prompt(call: Dict[str, Any]) -> str:
            inp = call.get("input", {})
            if isinstance(inp, dict) and inp.get("full_prompt"):
                return str(inp["full_prompt"])
            return ""

        def get_new_session_messages(call: Dict[str, Any]) -> List[Dict[str, Any]]:
            inp = call.get("input", {})
            if isinstance(inp, dict):
                new_sess = inp.get("new_session", {})
                if isinstance(new_sess, dict) and isinstance(new_sess.get("messages"), list):
                    return new_sess["messages"]
            if isinstance(call.get("new_session_messages"), list):
                return call.get("new_session_messages")
            return []

        def compute_state_before_path_str(p: str) -> str:
            """Normalize state path to relative form."""
            if not p:
                return ""
            # Store relative path in parquet for portability
            return str(p)

        for (conv_id, session_idx), calls in valid_groups:
            sorted_calls = sorted(calls, key=lambda c: agent_index_map.get(c.get("agent_type", ""), 999))
            session_group_id = f"{conv_id}_sess{session_idx}"

            core_call = next((c for c in sorted_calls if c.get("agent_type") == "core"), None)
            if core_call is None:
                continue

            state_before_path = compute_state_before_path_str(core_call.get("state_before_path", ""))
            new_session_messages = get_new_session_messages(core_call)
            session_timestamp = None
            try:
                inp = core_call.get("input", {})
                if isinstance(inp, dict):
                    ns = inp.get("new_session", {})
                    if isinstance(ns, dict):
                        session_timestamp = ns.get("timestamp")
            except Exception:
                session_timestamp = None

            qa_questions: List[Dict[str, Any]] = []
            
            # First try to use existing QA pairs from rl_training_data.jsonl
            if (conv_id, session_idx) in existing_qa_pairs:
                qa_questions = existing_qa_pairs[(conv_id, session_idx)]
                print(f"  Using existing {len(qa_questions)} QA pairs for {conv_id}_sess{session_idx}")
            elif add_qa and qa_generator is not None and qa_llm_client is not None:
                # Generate new QA pairs if not found
                try:
                    mem = MemorySystem(llm_client=qa_llm_client)
                    if state_before_path:
                        mem.load_state(state_before_path)

                    current_session_text = mem._format_messages(new_session_messages)
                    episodic = [m for m in mem.vector_store.memories if m.startswith("[EPISODIC]")]
                    semantic = [m for m in mem.vector_store.memories if m.startswith("[SEMANTIC]")]
                    procedural = [m for m in mem.vector_store.memories if m.startswith("[PROCEDURAL]")]

                    qa_questions = qa_generator.generate_qa(
                        current_session=current_session_text,
                        session_timestamp=str(session_timestamp or session_idx),
                        core_memory=mem.core_memory.human,
                        episodic_memories="\n".join(episodic),
                        semantic_memories="\n".join(semantic),
                        procedural_memories="\n".join(procedural),
                    )
                except Exception:
                    qa_questions = []

            for call in sorted_calls:
                agent_type = call.get("agent_type")
                if agent_type not in agent_order:
                    continue

                prompt_messages = [{"role": "user", "content": get_full_prompt(call)}]

                current_agent_gt = {
                    "agent_type": agent_type,
                    "expert_output": call.get("expert_output", {}),
                    "conversation_id": conv_id,
                    "session_index": session_idx,
                    "qa_questions": qa_questions,
                    "new_session_messages": new_session_messages,
                    "state_before_path": state_before_path,
                }

                other_agents_gt = [
                    {
                        "agent_type": c.get("agent_type"),
                        "expert_output": c.get("expert_output", {}),
                        "conversation_id": conv_id,
                        "session_index": session_idx,
                    }
                    for c in sorted_calls
                    if c.get("agent_type") != agent_type
                ]

                ground_truth_ordered = [current_agent_gt] + other_agents_gt

                agent_reward_config = {
                    "style": "custom",
                    "conversation_id": conv_id,
                    "session_index": session_idx,
                    "agent_type": agent_type,
                    "expert_output": call.get("expert_output", {}),
                    "ground_truth": ground_truth_ordered,
                    "qa_questions": qa_questions,  # 添加到顶层，方便 reward_function 访问
                    "state_before_path": state_before_path,
                }

                record = {
                    "prompt": prompt_messages,
                    "ability": "memory_management",
                    "reward_model": agent_reward_config,
                    "data_source": "memory_management_grouped",
                    "meta": {
                        "conversation_id": conv_id,
                        "session_index": session_idx,
                        "agent_type": agent_type,
                        "agent_index": agent_index_map.get(agent_type, -1),
                        "session_group_id": session_group_id,
                        "call_id": call.get("call_id", ""),
                        "is_grouped_training": True,
                    },
                }
                records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Save to parquet
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, engine="pyarrow", index=False)
        
        print(f"Saved {len(df)} records to {output_file}")
        
        # Print statistics
        print("\nStatistics:")
        print(f"  Sessions: {len(valid_groups)}")
        print(f"  Total records: {len(df)}")
        print(f"  By agent type:")
        for agent_type in ["core", "episodic", "semantic", "procedural"]:
            count = sum(
                1
                for m in df.get("meta", [])
                if isinstance(m, dict) and m.get("agent_type") == agent_type
            )
            print(f"    {agent_type}: {count}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Prepare RL training data")
    parser.add_argument("--trajectories-dir", type=str, required=True,
                       help="Directory containing expert trajectories")
    parser.add_argument("--output-file", type=str, required=True,
                       help="Output parquet file")
    parser.add_argument("--add-qa", action="store_true",
                       help="Add QA pairs for task reward")
    parser.add_argument("--qa-per-session", type=int, default=5,
                       help="Number of QA pairs per session")
    
    args = parser.parse_args()
    
    preprocessor = MemoryRLDataPreprocessor()
    preprocessor.process_trajectories(
        trajectories_dir=args.trajectories_dir,
        output_file=args.output_file,
        add_qa=args.add_qa,
        qa_per_session=args.qa_per_session
    )


if __name__ == "__main__":
    main()
