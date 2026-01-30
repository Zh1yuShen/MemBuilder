"""
Synthetic QA Generation for Dense Session-Level Rewards.

This module generates question-answer pairs for each session to enable
dense reward computation during ADRPO training.

As described in the paper (Section 3.4.1), we use Claude 4.5 Opus for
synthetic question generation to provide dense intermediate rewards.
"""

import json
from typing import Dict, List, Optional

from config import QA_GENERATION_MODEL
from prompts import QA_GENERATION_PROMPT


class QAGenerator:
    """Generate synthetic QA pairs for session-level rewards."""
    
    def __init__(self, llm_client, num_questions: int = 5):
        """
        Initialize QA generator.
        
        Args:
            llm_client: LLM client for generation
            num_questions: Number of questions to generate per session
        """
        self.llm_client = llm_client
        self.num_questions = num_questions
    
    def generate_qa(
        self,
        current_session: str,
        session_timestamp: str,
        core_memory: str = "",
        episodic_memories: str = "",
        semantic_memories: str = "",
        procedural_memories: str = ""
    ) -> List[Dict]:
        """
        Generate QA pairs for a session.
        
        Args:
            current_session: Current session conversation text
            session_timestamp: Timestamp of the session
            core_memory: Current core memory content
            episodic_memories: Retrieved episodic memories
            semantic_memories: Retrieved semantic memories
            procedural_memories: Retrieved procedural memories
        
        Returns:
            List of QA dicts with 'question', 'answer', 'type', 'source'
        """
        prompt = QA_GENERATION_PROMPT.format(
            core_memory=core_memory or "[Empty]",
            episodic_memories=episodic_memories or "[None]",
            semantic_memories=semantic_memories or "[None]",
            procedural_memories=procedural_memories or "[None]",
            current_session=current_session,
            session_timestamp=session_timestamp,
            num_questions=self.num_questions
        )
        
        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response)
            questions = result.get("questions", [])
            
            # Validate and clean questions
            validated = []
            for q in questions:
                if all(k in q for k in ["question", "answer", "type"]):
                    validated.append({
                        "question": q["question"],
                        "answer": q["answer"],
                        "type": q.get("type", "single-session"),
                        "source": q.get("source", "current_session")
                    })
            
            return validated
            
        except Exception as e:
            print(f"QA generation error: {str(e)[:100]}")
            return []


def prepare_rl_dataset(
    conversations: List[Dict],
    qa_generator: QAGenerator,
    memory_system
) -> List[Dict]:
    """
    Prepare RL training dataset with synthetic QA pairs.
    
    Args:
        conversations: List of conversation dicts with 'sessions' and 'id'
        qa_generator: QA generator instance
        memory_system: Memory system instance
    
    Returns:
        List of training samples with state, session, and QA pairs
    """
    rl_data = []
    
    for conv in conversations:
        conv_id = conv.get("id", "unknown")
        sessions = conv.get("sessions", [])
        
        for idx, session in enumerate(sessions):
            # Get current memory state
            core_mem = memory_system.core_memory.human
            
            # Retrieve relevant memories for context
            session_text = session.get("text", "")
            search_result = memory_system.search(session_text, conv_id, top_k=20)
            retrieved = search_result.get('results', [])
            
            # Separate by type
            episodic = [r["memory"] for r in retrieved if "[EPISODIC]" in r.get("memory", "")]
            semantic = [r["memory"] for r in retrieved if "[SEMANTIC]" in r.get("memory", "")]
            procedural = [r["memory"] for r in retrieved if "[PROCEDURAL]" in r.get("memory", "")]
            
            # Generate QA pairs
            qa_pairs = qa_generator.generate_qa(
                current_session=session_text,
                session_timestamp=session.get("timestamp", ""),
                core_memory=core_mem,
                episodic_memories="\n".join(episodic),
                semantic_memories="\n".join(semantic),
                procedural_memories="\n".join(procedural)
            )
            
            # Create training sample
            rl_data.append({
                "conversation_id": conv_id,
                "session_index": idx,
                "session": session,
                "state_before": {
                    "core_memory": core_mem,
                    "episodic": episodic,
                    "semantic": semantic,
                    "procedural": procedural
                },
                "questions": qa_pairs
            })
            
            # Process session to update memory state
            messages = session.get("messages", [])
            if messages:
                memory_system.add(
                    messages, 
                    user_id=conv_id,
                    metadata={"timestamp": session.get("timestamp")}
                )
    
    return rl_data


def generate_qa_from_trajectories(
    trajectory_dir: str,
    llm_client,
    num_questions: int = 5,
    output_path: str = None
) -> List[Dict]:
    """
    Generate QA pairs from pre-computed expert trajectories.
    
    This function reads trajectory data (agent_calls.jsonl) and memory states,
    then generates QA pairs for each session using the QAGenerator.
    
    Args:
        trajectory_dir: Path to trajectory directory containing agent_calls.jsonl
        llm_client: LLM client for QA generation
        num_questions: Number of questions per session
        output_path: Optional output path for rl_training_data.jsonl
    
    Returns:
        List of RL training samples with QA pairs
    """
    import os
    from pathlib import Path
    
    trajectory_dir = Path(trajectory_dir)
    output_path = output_path or trajectory_dir / "rl_training_data.jsonl"
    
    # Load trajectory data
    agent_calls_path = trajectory_dir / "agent_calls.jsonl"
    if not agent_calls_path.exists():
        raise FileNotFoundError(f"agent_calls.jsonl not found in {trajectory_dir}")
    
    with open(agent_calls_path, 'r') as f:
        calls = [json.loads(line) for line in f if line.strip()]
    
    # Group by session
    sessions = {}
    for call in calls:
        idx = call.get("session_index", 0)
        if idx not in sessions:
            sessions[idx] = {"input": call.get("input", {}), "state_path": None}
        if call.get("agent_type") == "core" and call.get("state_after_path"):
            sessions[idx]["state_path"] = trajectory_dir / call["state_after_path"]
    
    # Load metadata
    metadata_path = trajectory_dir / "metadata.json"
    conv_id = "unknown"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            conv_id = json.load(f).get("conv_id", "unknown")
    
    # Initialize QA generator
    qa_gen = QAGenerator(llm_client, num_questions=num_questions)
    
    # Process sessions
    results = []
    for idx in sorted(sessions.keys()):
        sess = sessions[idx]
        inp = sess["input"]
        new_sess = inp.get("new_session", {})
        messages = new_sess.get("messages", [])
        timestamp = new_sess.get("timestamp", "")
        session_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        
        # Load state
        state = {"core_memory": "", "episodic": [], "semantic": [], "procedural": []}
        state_path = sess.get("state_path")
        if state_path and state_path.exists():
            core_path = state_path / "core_memory.json"
            if core_path.exists():
                with open(core_path) as f:
                    state["core_memory"] = json.load(f).get("human", "")
            
            payload_path = state_path / "vector_store" / "payload.json"
            if payload_path.exists():
                with open(payload_path) as f:
                    for mem in json.load(f).get("memories", []):
                        if "[EPISODIC]" in mem:
                            state["episodic"].append(mem)
                        elif "[SEMANTIC]" in mem:
                            state["semantic"].append(mem)
                        elif "[PROCEDURAL]" in mem:
                            state["procedural"].append(mem)
        
        print(f"  Session {idx+1}/{len(sessions)}...", end=" ", flush=True)
        
        # Generate QA pairs
        qa_pairs = qa_gen.generate_qa(
            current_session=session_text,
            session_timestamp=timestamp,
            core_memory=state.get("core_memory", ""),
            episodic_memories="\n".join(state.get("episodic", [])),
            semantic_memories="\n".join(state.get("semantic", [])),
            procedural_memories="\n".join(state.get("procedural", []))
        )
        print(f"got {len(qa_pairs)} questions")
        
        results.append({
            "conversation_id": conv_id,
            "session_index": idx,
            "session": {"messages": messages, "timestamp": timestamp},
            "state_after": state,
            "questions": qa_pairs
        })
    
    # Save results
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    total_q = sum(len(r["questions"]) for r in results)
    print(f"\nDone! {len(results)} sessions, {total_q} QA pairs -> {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from expert trajectories for RL training"
    )
    parser.add_argument(
        "--trajectory-dir", required=True,
        help="Path to trajectory directory containing agent_calls.jsonl"
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="Model for QA generation (default: gpt-4o)"
    )
    parser.add_argument(
        "--num-questions", type=int, default=5,
        help="Number of questions per session (default: 5)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: rl_training_data.jsonl in trajectory dir)"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for LLM service"
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Base URL for OpenAI-compatible API"
    )
    args = parser.parse_args()
    
    # Import LLM client
    from llm_client import OpenAIClient
    
    # Initialize client
    llm_client = OpenAIClient(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
        model=args.model
    )
    
    print("=" * 60)
    print("QA Generation from Expert Trajectories")
    print("=" * 60)
    print(f"Trajectory dir: {args.trajectory_dir}")
    print(f"Model: {args.model}")
    print(f"Questions per session: {args.num_questions}")
    print("=" * 60)
    
    generate_qa_from_trajectories(
        trajectory_dir=args.trajectory_dir,
        llm_client=llm_client,
        num_questions=args.num_questions,
        output_path=args.output
    )
