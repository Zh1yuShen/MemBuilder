#!/usr/bin/env python3
"""
Process LOCOMO dataset for MemBuilder evaluation and training.

Converts raw LOCOMO format to:
1. Evaluation format (for test_memory_system.py style testing)
2. Expert trajectories (for RL training data preparation)

Raw LOCOMO format:
- conversation: {session_1, session_1_date_time, session_2, ...}
- qa: [{question, answer, category, evidence}, ...]
- sample_id: conversation identifier

Output evaluation format:
- haystack_sessions: [session1_messages, session2_messages, ...]
- haystack_session_datetimes: [timestamp1, timestamp2, ...]
- question_id: unique question identifier
- question, answer, question_type

Usage:
    # Convert to evaluation format
    python process_locomo.py --mode eval --output data/locomo_eval.json
    
    # Generate expert trajectories for RL training
    python process_locomo.py --mode trajectories --output expert_trajectories/locomo
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


# Question category mapping (from LOCOMO paper)
CATEGORY_MAP = {
    1: "single-session",      # Single-hop factual
    2: "temporal-reasoning",  # Temporal reasoning
    3: "multi-session",       # Multi-hop inference
    4: "knowledge-update",    # Knowledge update (if exists)
}


def parse_session_messages(session_data: List[Dict]) -> List[Dict[str, str]]:
    """
    Convert LOCOMO session format to standard messages format.
    
    LOCOMO format:
        [{"speaker": "Caroline", "dia_id": "D1:1", "text": "..."}, ...]
    
    Output format:
        [{"role": "user/assistant", "content": "..."}, ...]
    """
    messages = []
    speakers = set()
    
    for turn in session_data:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        speakers.add(speaker)
        
        # Determine role based on speaker order (first speaker = user)
        role = "user" if len(speakers) == 1 or speaker == list(speakers)[0] else "assistant"
        
        # Include speaker name in content for clarity
        content = f"[{speaker}]: {text}"
        
        messages.append({
            "role": role,
            "content": content
        })
    
    return messages


def extract_sessions_and_timestamps(conversation: Dict) -> tuple:
    """
    Extract sessions and timestamps from LOCOMO conversation format.
    
    Returns:
        (sessions_list, timestamps_list, speaker_a, speaker_b)
    """
    sessions = []
    timestamps = []
    
    # Find all session keys (session_1, session_2, ...)
    session_keys = sorted([
        k for k in conversation.keys() 
        if k.startswith("session_") and not k.endswith("_date_time")
    ], key=lambda x: int(x.split("_")[1]))
    
    for session_key in session_keys:
        session_num = session_key.split("_")[1]
        timestamp_key = f"session_{session_num}_date_time"
        
        session_data = conversation.get(session_key, [])
        timestamp = conversation.get(timestamp_key, "")
        
        if session_data:
            messages = parse_session_messages(session_data)
            sessions.append({"messages": messages})
            timestamps.append(timestamp)
    
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    
    return sessions, timestamps, speaker_a, speaker_b


def convert_to_eval_format(raw_data: List[Dict]) -> List[Dict]:
    """
    Convert raw LOCOMO to evaluation format (one record per question).
    
    Each output record contains:
    - All sessions as haystack (for memory building)
    - Single question for evaluation
    """
    eval_records = []
    
    for sample in raw_data:
        sample_id = sample.get("sample_id", "unknown")
        conversation = sample.get("conversation", {})
        qa_list = sample.get("qa", [])
        
        # Extract sessions and timestamps
        sessions, timestamps, speaker_a, speaker_b = extract_sessions_and_timestamps(conversation)
        
        if not sessions:
            print(f"  Warning: No sessions found in {sample_id}")
            continue
        
        # Create one record per question
        for q_idx, qa in enumerate(qa_list):
            question_id = f"{sample_id}_q{q_idx + 1}"
            category = qa.get("category", 1)
            question_type = CATEGORY_MAP.get(category, "unknown")
            
            record = {
                "question_id": question_id,
                "sample_id": sample_id,
                "question": qa.get("question", ""),
                "answer": str(qa.get("answer", "")),  # Ensure string
                "question_type": question_type,
                "category": category,
                "evidence": qa.get("evidence", []),
                "haystack_sessions": sessions,
                "haystack_session_datetimes": timestamps,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "num_sessions": len(sessions),
            }
            
            eval_records.append(record)
    
    return eval_records


def convert_to_conversation_format(raw_data: List[Dict]) -> List[Dict]:
    """
    Convert raw LOCOMO to conversation-centric format (one record per conversation).
    
    Each output record contains all sessions and all questions for one conversation.
    Useful for building memory once and answering multiple questions.
    """
    conv_records = []
    
    for sample in raw_data:
        sample_id = sample.get("sample_id", "unknown")
        conversation = sample.get("conversation", {})
        qa_list = sample.get("qa", [])
        
        # Extract sessions and timestamps
        sessions, timestamps, speaker_a, speaker_b = extract_sessions_and_timestamps(conversation)
        
        if not sessions:
            continue
        
        # Convert QA list
        questions = []
        for q_idx, qa in enumerate(qa_list):
            category = qa.get("category", 1)
            questions.append({
                "question_id": f"{sample_id}_q{q_idx + 1}",
                "question": qa.get("question", ""),
                "answer": str(qa.get("answer", "")),
                "question_type": CATEGORY_MAP.get(category, "unknown"),
                "category": category,
                "evidence": qa.get("evidence", []),
            })
        
        record = {
            "conversation_id": sample_id,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "haystack_sessions": sessions,
            "haystack_session_datetimes": timestamps,
            "num_sessions": len(sessions),
            "questions": questions,
            "num_questions": len(questions),
        }
        
        conv_records.append(record)
    
    return conv_records


def generate_subset_config(
    eval_records: List[Dict],
    conversation_ids: Optional[List[str]] = None,
    num_questions: Optional[int] = None,
    output_path: str = None
) -> Dict:
    """
    Generate a subset configuration file for selective testing.
    
    Args:
        eval_records: Full evaluation records
        conversation_ids: List of conversation IDs to include (e.g., ["conv-26"])
        num_questions: Limit number of questions
        output_path: Optional path to save subset config
    """
    if conversation_ids:
        filtered = [r for r in eval_records if r["sample_id"] in conversation_ids]
    else:
        filtered = eval_records
    
    if num_questions:
        filtered = filtered[:num_questions]
    
    question_ids = [r["question_id"] for r in filtered]
    
    config = {
        "name": f"locomo_subset_{len(question_ids)}q",
        "question_ids": question_ids,
        "conversations": list(set(r["sample_id"] for r in filtered)),
        "total_questions": len(question_ids),
    }
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Saved subset config to {output_path}")
    
    return config


def print_statistics(raw_data: List[Dict], eval_records: List[Dict]):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("LOCOMO Dataset Statistics")
    print("=" * 60)
    
    print(f"\nConversations: {len(raw_data)}")
    print(f"Total Questions: {len(eval_records)}")
    
    # Questions per conversation
    by_conv = defaultdict(int)
    for r in eval_records:
        by_conv[r["sample_id"]] += 1
    
    print("\nQuestions per conversation:")
    for conv_id, count in sorted(by_conv.items()):
        print(f"  {conv_id}: {count}")
    
    # Questions by type
    by_type = defaultdict(int)
    for r in eval_records:
        by_type[r["question_type"]] += 1
    
    print("\nQuestions by type:")
    for q_type, count in sorted(by_type.items()):
        pct = count / len(eval_records) * 100
        print(f"  {q_type}: {count} ({pct:.1f}%)")
    
    # Sessions per conversation
    sessions_counts = []
    for sample in raw_data:
        conv = sample.get("conversation", {})
        session_keys = [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")]
        sessions_counts.append(len(session_keys))
    
    print(f"\nSessions per conversation: {min(sessions_counts)}-{max(sessions_counts)} (avg: {sum(sessions_counts)/len(sessions_counts):.1f})")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Process LOCOMO dataset for MemBuilder")
    parser.add_argument("--input", type=str, 
                       default="data/locomo_raw/data/locomo10.json",
                       help="Path to raw LOCOMO JSON file")
    parser.add_argument("--mode", type=str, choices=["eval", "conversation", "both"],
                       default="both",
                       help="Output mode: eval (per-question), conversation (per-conv), or both")
    parser.add_argument("--output-dir", type=str, default="data/locomo",
                       help="Output directory")
    parser.add_argument("--subset-conv", type=str, nargs="*",
                       help="Create subset for specific conversations (e.g., conv-26)")
    parser.add_argument("--subset-questions", type=int,
                       help="Limit number of questions in subset")
    
    args = parser.parse_args()
    
    # Load raw data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    print(f"Loading LOCOMO from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} conversations")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to evaluation format
    eval_records = convert_to_eval_format(raw_data)
    
    # Print statistics
    print_statistics(raw_data, eval_records)
    
    # Save outputs
    if args.mode in ["eval", "both"]:
        # Save as JSONL (one record per line, for streaming)
        eval_jsonl_path = output_dir / "locomo_eval.jsonl"
        with open(eval_jsonl_path, "w", encoding="utf-8") as f:
            for record in eval_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nSaved evaluation format (JSONL): {eval_jsonl_path}")
        print(f"  Records: {len(eval_records)}")
    
    if args.mode in ["conversation", "both"]:
        conv_records = convert_to_conversation_format(raw_data)
        conv_json_path = output_dir / "locomo_conversations.json"
        with open(conv_json_path, "w", encoding="utf-8") as f:
            json.dump(conv_records, f, indent=2, ensure_ascii=False)
        print(f"\nSaved conversation format (JSON): {conv_json_path}")
        print(f"  Conversations: {len(conv_records)}")
    
    # Generate subset config if requested
    if args.subset_conv or args.subset_questions:
        subset_path = output_dir / "subset_config.json"
        generate_subset_config(
            eval_records,
            conversation_ids=args.subset_conv,
            num_questions=args.subset_questions,
            output_path=str(subset_path)
        )
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
