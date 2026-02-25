#!/usr/bin/env python3
"""
Generate Expert Memory Building Trajectories for SFT Training.

This script generates expert memory construction trajectories using a strong
expert model (e.g., Claude 4.5 Sonnet) for supervised fine-tuning.

Output structure:
expert_trajectories/{dataset}/{sample_id}/
├── states/          # Memory state snapshots before each session
├── agent_calls.jsonl # Call records for 4 agents (Core, Episodic, Semantic, Procedural)
└── metadata.json

Usage:
    # Single conversation (LoCoMo)
    python generate_expert_trajectories.py \
        --dataset locomo \
        --conv-id conv-26 \
        --expert-model claude-sonnet-4-5

    # Using predefined split (LongMemEval)
    python generate_expert_trajectories.py \
        --dataset longmemeval \
        --split sft \
        --expert-model claude-sonnet-4-5 \
        --parallel --workers 4
"""

import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_system import MemorySystem
from config import SFT_EXPERT_MODEL

# Client module: internal version adds extra providers; public version has openai/vllm only.
try:
    from llm_client_internal import create_llm_client as _factory_create_client, AVAILABLE_PROVIDERS
except ImportError:
    from llm_client import create_llm_client as _factory_create_client, AVAILABLE_PROVIDERS

# Provider factory: creates the appropriate LLM client
_LLM_CLIENT_PROVIDER = "openai"  # default; overridden via --provider CLI arg

def _create_llm_client(model: str):
    """Create LLM client based on the global provider setting."""
    return _factory_create_client(provider=_LLM_CLIENT_PROVIDER, model=model)

class _RecordingClientWrapper:
    """Thin wrapper that records LLM chat_completion calls for trajectory logging,
    while delegating all execution to the underlying client."""

    def __init__(self, real_client):
        self._real = real_client
        self.calls = []  # list of {prompt, response, time_ms}

    def chat_completion(self, **kwargs):
        prompt = ""
        msgs = kwargs.get("messages", [])
        if msgs:
            prompt = msgs[0].get("content", "")
        start = time.time()
        result = self._real.chat_completion(**kwargs)
        elapsed_ms = (time.time() - start) * 1000
        self.calls.append({"prompt": prompt, "response": result, "time_ms": elapsed_ms})
        return result

    def get_embeddings(self, *args, **kwargs):
        return self._real.get_embeddings(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate unknown attributes to the real client."""
        return getattr(self._real, name)

    def clear(self):
        self.calls = []


class ExpertTrajectoryGenerator:
    """Generate expert memory building trajectories."""
    
    def __init__(
        self, 
        dataset: str, 
        expert_model: str, 
        output_dir: str = "./expert_trajectories"
    ):
        self.dataset = dataset
        self.expert_model = expert_model
        self.output_dir = Path(output_dir) / dataset
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized trajectory generator:")
        print(f"  Dataset: {dataset}")
        print(f"  Expert model: {expert_model}")
        print(f"  Output directory: {self.output_dir}")
    
    def generate_conversation_trajectory(
        self,
        conv_id: str,
        sessions: List[Dict],
        skip_existing: bool = False
    ) -> Dict:
        """Generate complete trajectory for a single conversation."""
        conv_dir = self.output_dir / conv_id
        
        # Check if already exists
        if skip_existing and (conv_dir / "agent_calls.jsonl").exists():
            print(f"Skipping existing: {conv_id}")
            return {"conv_id": conv_id, "status": "skipped"}
        
        conv_dir.mkdir(parents=True, exist_ok=True)
        states_dir = conv_dir / "states"
        states_dir.mkdir(exist_ok=True)
        
        # Initialize memory system with recording wrapper around the real client
        real_client = _create_llm_client(model=self.expert_model)
        recording_client = _RecordingClientWrapper(real_client)
        memory = MemorySystem(llm_client=recording_client)
        
        # Process each session (with per-session retry on transient errors)
        agent_calls = []
        max_session_retries = 3
        
        for session_idx, session in enumerate(sessions):
            messages = session.get("messages", session) if isinstance(session, dict) else session
            timestamp = session.get("timestamp") if isinstance(session, dict) else None

            print(f"  Processing session {session_idx + 1}/{len(sessions)}...")

            state_before_path = states_dir / f"state_{session_idx}"
            self._save_state(memory, state_before_path)

            last_error = None
            for attempt in range(1, max_session_retries + 1):
                try:
                    session_calls = self._process_session(
                        memory=memory,
                        messages=messages,
                        session_index=session_idx,
                        conv_id=conv_id,
                        user_id=conv_id,
                        state_before_path=str(state_before_path),
                        timestamp=timestamp,
                    )
                    agent_calls.extend(session_calls)
                    break  # success
                except Exception as e:
                    last_error = e
                    if attempt < max_session_retries:
                        wait_sec = attempt * 5
                        print(f"  ⚠️ Session {session_idx + 1} failed (attempt {attempt}/{max_session_retries}): {str(e)[:80]}")
                        print(f"     Restoring state and retrying in {wait_sec}s...")
                        time.sleep(wait_sec)
                        # Restore memory state from checkpoint before retry
                        memory.load_state(str(state_before_path))
                    else:
                        print(f"  ❌ Session {session_idx + 1} failed after {max_session_retries} attempts: {str(e)[:100]}")
                        raise last_error
        
        # Save final state
        final_state_path = states_dir / f"state_{len(sessions)}"
        self._save_state(memory, final_state_path)
        
        # Save agent calls
        with open(conv_dir / "agent_calls.jsonl", "w") as f:
            for call in agent_calls:
                f.write(json.dumps(call, ensure_ascii=False) + "\n")
        
        # Save metadata
        metadata = {
            "conv_id": conv_id,
            "num_sessions": len(sessions),
            "expert_model": self.expert_model,
            "total_agent_calls": len(agent_calls)
        }
        with open(conv_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "conv_id": conv_id,
            "status": "completed",
            "num_sessions": len(sessions),
            "agent_calls": len(agent_calls)
        }
    
    def _save_state(self, memory: MemorySystem, state_path: Path):
        """Save memory state to disk."""
        memory.save_state(str(state_path))

    def _process_session(
        self,
        memory: MemorySystem,
        messages: List[Dict[str, Any]],
        session_index: int,
        conv_id: str,
        user_id: str,
        state_before_path: str,
        timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        calls = []
        all_searchable_memories = []  # Collect memories; batch-insert after all agents

        core_call = self._call_core_agent(
            memory_system=memory,
            messages=messages,
            session_index=session_index,
            conv_id=conv_id,
            user_id=user_id,
            state_before_path=state_before_path,
            timestamp=timestamp,
        )
        calls.append(core_call)

        for agent_type in ["episodic", "semantic", "procedural"]:
            call_record, memories = self._call_memory_agent(
                memory_system=memory,
                agent_type=agent_type,
                messages=messages,
                session_index=session_index,
                conv_id=conv_id,
                user_id=user_id,
                state_before_path=state_before_path,
                timestamp=timestamp,
            )
            calls.append(call_record)
            all_searchable_memories.extend(memories)

        # Batch-insert all searchable memories AFTER all agents complete
        # (matches MemorySystem.add() behavior where agents run in parallel
        # and all see the same vector store state)
        if all_searchable_memories:
            memory.vector_store.add(
                all_searchable_memories,
                [{"user_id": user_id, "type": "memory"} for _ in all_searchable_memories],
            )

        return calls

    def _call_core_agent(
        self,
        memory_system: MemorySystem,
        messages: List[Dict[str, Any]],
        session_index: int,
        conv_id: str,
        user_id: str,
        state_before_path: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Capture state before for recording
        current_core = memory_system.core_memory.human or "[Empty]"
        core_usage = memory_system.core_memory.get_usage()

        # Clear recorder, then delegate ALL logic to MemorySystem._core_agent
        # (handles APPEND/REPLACE/REWRITE, content validation, compression)
        recording = memory_system.llm_client
        recording.clear()

        call_start = time.time()
        operation, new_core_memory = memory_system._core_agent(messages, user_id)
        call_time_ms = (time.time() - call_start) * 1000

        # Retrieve the prompt from the recorded wrapper (first call = agent decision)
        full_prompt = recording.calls[0]["prompt"] if recording.calls else ""

        return {
            "conversation_id": conv_id,
            "session_index": session_index,
            "agent_type": "core",
            "call_id": f"{conv_id}_session-{session_index}_core",
            "state_before_path": state_before_path,
            "input": {
                "system_prompt": memory_system.prompts["core"],
                "retrieved_memories": {
                    "core_memory": current_core,
                    "usage_percent": core_usage,
                },
                "new_session": {
                    "timestamp": timestamp,
                    "messages": messages,
                },
                "full_prompt": full_prompt,
            },
            "expert_output": {
                "operation": operation,
                "new_core_memory": new_core_memory,
            },
            "metadata": {
                "expert_model": self.expert_model,
                "temperature": 0.0,
                "call_time_ms": call_time_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }

    def _call_memory_agent(
        self,
        memory_system: MemorySystem,
        agent_type: str,
        messages: List[Dict[str, Any]],
        session_index: int,
        conv_id: str,
        user_id: str,
        state_before_path: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Capture existing memories for recording
        existing = memory_system._get_existing_memories(agent_type, user_id, messages)

        # Clear recorder, then delegate ALL logic to MemorySystem's agent method
        # (handles JSON retry, operation parsing, memory extraction with prefixes)
        recording = memory_system.llm_client
        recording.clear()

        call_start = time.time()
        metadata_dict = {"timestamp": timestamp} if timestamp else None

        if agent_type == "episodic":
            memories, stats = memory_system._episodic_agent(messages, user_id, metadata_dict)
        elif agent_type == "semantic":
            memories, stats = memory_system._semantic_agent(messages, user_id)
        else:  # procedural
            memories, stats = memory_system._procedural_agent(messages, user_id)

        call_time_ms = (time.time() - call_start) * 1000

        # Retrieve prompt and raw response from the recording wrapper
        full_prompt = recording.calls[0]["prompt"] if recording.calls else ""

        # Parse raw response for the operations list (for call record)
        operations = []
        if recording.calls:
            last_response = recording.calls[-1]["response"]
            try:
                clean = memory_system._clean_json_response(last_response)
                operations = json.loads(clean).get("operations", [])
            except Exception:
                pass

        # Return call record AND memories list (caller batches vector store insertion)
        call_record = {
            "conversation_id": conv_id,
            "session_index": session_index,
            "agent_type": agent_type,
            "call_id": f"{conv_id}_session-{session_index}_{agent_type}",
            "state_before_path": state_before_path,
            "input": {
                "system_prompt": memory_system.prompts[agent_type],
                "retrieved_memories": {
                    agent_type: [
                        {"id": f"{agent_type}_{i}", "content": mem, "score": 0.0}
                        for i, mem in enumerate(existing[:50])
                    ]
                },
                "new_session": {
                    "timestamp": timestamp,
                    "messages": messages,
                },
                "full_prompt": full_prompt,
            },
            "expert_output": {
                "operations": operations,
                "memories_added": memories,
            },
            "metadata": {
                "expert_model": self.expert_model,
                "temperature": 0.0,
                "call_time_ms": call_time_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        return call_record, memories
    
    def generate_batch(
        self,
        conversations: List[Dict],
        parallel: bool = False,
        workers: int = 4,
        skip_existing: bool = True
    ):
        """Generate trajectories for multiple conversations."""
        print(f"\nGenerating {len(conversations)} trajectories...")
        
        results = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self.generate_conversation_trajectory,
                        conv["id"],
                        conv["sessions"],
                        skip_existing
                    ): conv["id"]
                    for conv in conversations
                }
                
                for future in as_completed(futures):
                    conv_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        print(f"  Completed: {conv_id}")
                    except Exception as e:
                        print(f"  Failed: {conv_id} - {str(e)[:100]}")
                        results.append({"conv_id": conv_id, "status": "failed", "error": str(e)})
        else:
            for conv in conversations:
                try:
                    result = self.generate_conversation_trajectory(
                        conv["id"],
                        conv["sessions"],
                        skip_existing
                    )
                    results.append(result)
                except Exception as e:
                    print(f"  Failed: {conv['id']} - {str(e)[:100]}")
                    results.append({"conv_id": conv["id"], "status": "failed", "error": str(e)})
        
        # Summary
        completed = sum(1 for r in results if r.get("status") == "completed")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        failed = sum(1 for r in results if r.get("status") == "failed")
        
        print(f"\nGeneration complete:")
        print(f"  Completed: {completed}")
        print(f"  Skipped: {skipped}")
        print(f"  Failed: {failed}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate expert trajectories")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset file (JSONL/JSON)")
    parser.add_argument("--conv-id", type=str, help="Single conversation ID")
    parser.add_argument("--subset-file", type=str, help="JSON file with conversation IDs")
    parser.add_argument("--split", type=str, choices=["sft", "rl", "test"], 
                       help="Use predefined split from data/longmemeval/splits/longmemeval_splits.json")
    parser.add_argument("--expert-model", type=str, default=SFT_EXPERT_MODEL, 
                       help="Expert model for generation")
    parser.add_argument("--output-dir", type=str, default="./expert_trajectories",
                       help="Output directory")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing trajectories")
    parser.add_argument("--provider", type=str, default="openai", choices=AVAILABLE_PROVIDERS,
                       help="LLM provider (default: openai)")
    
    args = parser.parse_args()
    
    # Set global provider
    global _LLM_CLIENT_PROVIDER
    _LLM_CLIENT_PROVIDER = args.provider
    print(f"LLM provider: {args.provider}")
    
    # Handle --split parameter: convert to subset_file path
    if args.split and args.dataset == "longmemeval":
        splits_file = Path(__file__).parent.parent / "data" / "longmemeval" / "splits" / "longmemeval_splits.json"
        if splits_file.exists():
            with open(splits_file, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            if args.split in splits_data:
                # Create a temporary list of question IDs for this split
                args._split_question_ids = splits_data[args.split]
                print(f"Using {args.split} split: {len(args._split_question_ids)} dialogues")
            else:
                raise ValueError(f"Split '{args.split}' not found in {splits_file}")
        else:
            raise FileNotFoundError(f"Splits file not found: {splits_file}")
    else:
        args._split_question_ids = None
    
    generator = ExpertTrajectoryGenerator(
        dataset=args.dataset,
        expert_model=args.expert_model,
        output_dir=args.output_dir
    )
    
    def load_jsonl(path: Path) -> List[Dict[str, Any]]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def resolve_locomo_path(dataset_path: Optional[str]) -> Path:
        if dataset_path:
            return Path(dataset_path)

        candidates = [
            Path(__file__).parent.parent / "data" / "locomo_mc10.json",
            Path(__file__).parent.parent.parent / "locomo" / "locomo_mc10" / "data" / "locomo_mc10.json",
            Path(__file__).parent.parent.parent / "locomo" / "locomo_mc10" / "data" / "locomo_mc10.jsonl",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError("LoCoMo dataset file not found. Provide --dataset-path.")

    def locomo_conv_sessions(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        sessions = sample.get("haystack_sessions", [])
        dates = sample.get("haystack_session_datetimes") or sample.get("haystack_dates") or []
        out = []
        for idx, sess in enumerate(sessions):
            ts = dates[idx] if idx < len(dates) else None
            msgs = []
            for turn in sess:
                if isinstance(turn, dict):
                    role = turn.get("speaker") or turn.get("role") or "Unknown"
                    content = turn.get("message") or turn.get("text") or turn.get("content") or ""
                    if content:
                        msgs.append({"role": str(role), "content": str(content)})
            out.append({"messages": msgs, "timestamp": ts})
        return out

    def load_locomo_conversations(
        dataset_path: Optional[str],
        conv_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        path = resolve_locomo_path(dataset_path)
        data = load_jsonl(path)
        grouped = {}
        for item in data:
            qid = item.get("question_id")
            if not qid or "_q" not in qid:
                continue
            cid = qid.split("_q")[0]
            if conv_ids and cid not in set(conv_ids):
                continue
            if cid not in grouped:
                grouped[cid] = {
                    "id": cid,
                    "sessions": locomo_conv_sessions(item),
                }
        if conv_ids:
            ordered = []
            for cid in conv_ids:
                if cid in grouped:
                    ordered.append(grouped[cid])
            return ordered
        return list(grouped.values())

    def parse_subset_ids(obj: Any) -> List[str]:
        if isinstance(obj, list):
            if not obj:
                return []
            if isinstance(obj[0], str):
                return obj
            return []
        if isinstance(obj, dict):
            if "conv_ids" in obj and isinstance(obj["conv_ids"], list):
                return [str(x) for x in obj["conv_ids"]]
            if "question_ids" in obj and isinstance(obj["question_ids"], list):
                convs = sorted({str(q).split("_q")[0] for q in obj["question_ids"] if "_q" in str(q)})
                return convs
        return []

    if args.dataset == "locomo":
        if args.conv_id:
            conversations = load_locomo_conversations(args.dataset_path, conv_ids=[args.conv_id])
            if not conversations:
                raise ValueError(f"Conversation not found: {args.conv_id}")
            generator.generate_conversation_trajectory(
                conv_id=conversations[0]["id"],
                sessions=conversations[0]["sessions"],
                skip_existing=args.skip_existing,
            )
        elif args.subset_file:
            with open(args.subset_file, "r", encoding="utf-8") as f:
                subset_obj = json.load(f)
            conv_ids = parse_subset_ids(subset_obj)
            if not conv_ids and isinstance(subset_obj, list) and subset_obj and isinstance(subset_obj[0], dict):
                conversations = subset_obj
            else:
                conversations = load_locomo_conversations(args.dataset_path, conv_ids=conv_ids or None)
            generator.generate_batch(
                conversations,
                parallel=args.parallel,
                workers=args.workers,
                skip_existing=args.skip_existing,
            )
        else:
            parser.print_help()
    elif args.dataset == "longmemeval":
        # LongMemEval dataset support
        def resolve_longmemeval_path(dataset_path: Optional[str]) -> Path:
            if dataset_path:
                return Path(dataset_path)
            candidates = [
                Path(__file__).parent.parent / "data" / "longmemeval" / "longmemeval_s_cleaned.json",
                Path(__file__).parent.parent.parent / "longmemeval" / "longmemeval_s.json",
            ]
            for p in candidates:
                if p.exists():
                    return p
            raise FileNotFoundError("LongMemEval dataset not found. Provide --dataset-path.")

        def longmemeval_conv_sessions(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
            sessions = sample.get("haystack_sessions", [])
            dates = sample.get("haystack_dates", [])
            out = []
            for idx, sess in enumerate(sessions):
                ts = dates[idx] if idx < len(dates) else None
                msgs = []
                for turn in sess:
                    if isinstance(turn, dict):
                        role = turn.get("role", "user")
                        content = turn.get("content", "")
                        if content:
                            msgs.append({"role": str(role), "content": str(content)})
                out.append({"messages": msgs, "timestamp": ts})
            return out

        def load_longmemeval_conversations(
            dataset_path: Optional[str],
            question_ids: Optional[List[str]] = None,
        ) -> List[Dict[str, Any]]:
            path = resolve_longmemeval_path(dataset_path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Group by question_id (each sample is a unique conversation)
            conversations = []
            seen_ids = set()
            for item in data:
                qid = item.get("question_id", "")
                if not qid or qid in seen_ids:
                    continue
                if question_ids and qid not in set(question_ids):
                    continue
                seen_ids.add(qid)
                conversations.append({
                    "id": qid,
                    "sessions": longmemeval_conv_sessions(item),
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                })
            return conversations

        if args.conv_id:
            conversations = load_longmemeval_conversations(args.dataset_path, question_ids=[args.conv_id])
            if not conversations:
                raise ValueError(f"Conversation not found: {args.conv_id}")
            generator.generate_conversation_trajectory(
                conv_id=conversations[0]["id"],
                sessions=conversations[0]["sessions"],
                skip_existing=args.skip_existing,
            )
        elif args._split_question_ids:
            # Use --split parameter
            conversations = load_longmemeval_conversations(args.dataset_path, question_ids=args._split_question_ids)
            generator.generate_batch(
                conversations,
                parallel=args.parallel,
                workers=args.workers,
                skip_existing=args.skip_existing,
            )
        elif args.subset_file:
            with open(args.subset_file, "r", encoding="utf-8") as f:
                subset_obj = json.load(f)
            question_ids = subset_obj if isinstance(subset_obj, list) else subset_obj.get("question_ids", [])
            conversations = load_longmemeval_conversations(args.dataset_path, question_ids=question_ids or None)
            generator.generate_batch(
                conversations,
                parallel=args.parallel,
                workers=args.workers,
                skip_existing=args.skip_existing,
            )
        else:
            # Default: load first conversation
            conversations = load_longmemeval_conversations(args.dataset_path)
            if conversations:
                print(f"No --conv-id specified. Using first conversation: {conversations[0]['id']}")
                generator.generate_conversation_trajectory(
                    conv_id=conversations[0]["id"],
                    sessions=conversations[0]["sessions"],
                    skip_existing=args.skip_existing,
                )
            else:
                parser.print_help()
    else:
        if args.subset_file:
            with open(args.subset_file, "r", encoding="utf-8") as f:
                subset_data = json.load(f)

            generator.generate_batch(
                subset_data,
                parallel=args.parallel,
                workers=args.workers,
                skip_existing=args.skip_existing,
            )
        else:
            parser.print_help()



if __name__ == "__main__":
    main()
