#!/usr/bin/env python3
"""
Convert expert trajectories to SFT training format.

This script converts the agent_calls.jsonl from expert trajectory generation
into the instruction-input-output format used by LLaMA-Factory for SFT training.

The expert trajectories already contain the full prompt in 'input.full_prompt',
so this script simply extracts and reformats the data.

Usage:
    python scripts/convert_trajectories_to_sft.py \
        --trajectory-dir expert_trajectories/longmemeval \
        --output-file /path/to/LLaMA-Factory/data/memory_building_sft.json \
        --max-length 20000

Output format (LLaMA-Factory compatible, alpaca style):
    [
        {"instruction": "...", "input": "", "output": "```json\n{...}\n```"},
        ...
    ]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_trajectory(trajectory_dir: Path) -> List[Dict]:
    """Load agent calls from trajectory directory."""
    agent_calls_path = trajectory_dir / "agent_calls.jsonl"
    if not agent_calls_path.exists():
        return []
    
    calls = []
    with open(agent_calls_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    calls.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {agent_calls_path}: {e}")
    return calls


def create_sft_sample(call: Dict) -> Optional[Dict]:
    """
    Create a single SFT sample from an agent call.
    
    Uses the pre-built full_prompt from the trajectory as the instruction,
    and formats the expert_output as a JSON code block.
    """
    # Get the full prompt (instruction) from input
    input_data = call.get("input") or {}
    full_prompt = input_data.get("full_prompt", "")
    
    if not full_prompt:
        # Fallback: try to build from system_prompt if full_prompt not available
        system_prompt = input_data.get("system_prompt", "")
        if not system_prompt:
            return None
        full_prompt = system_prompt
    
    # Get expert output
    expert_output = call.get("expert_output") or {}
    if not expert_output:
        return None
    
    # Skip error outputs
    if expert_output.get("operation") == "error":
        return None
    
    # Normalize output field names to match actual SFT training data format
    # Expert trajectories use "new_core_memory", but training data uses "content"
    normalized_output = dict(expert_output)
    if "new_core_memory" in normalized_output:
        normalized_output["content"] = normalized_output.pop("new_core_memory")
    
    # Format output as JSON code block (matching actual SFT data format)
    output_json = json.dumps(normalized_output, indent=2, ensure_ascii=False)
    output_text = f"```json\n{output_json}\n```"
    
    return {
        "instruction": full_prompt,
        "input": "",
        "output": output_text
    }


def estimate_token_length(text: str) -> int:
    """Rough estimate of token count (chars / 4)."""
    return len(text) // 4


def convert_trajectories(
    trajectory_dir: Path,
    output_file: Path,
    max_length: int = 20000
) -> int:
    """Convert all trajectories to SFT format."""
    all_samples = []
    skipped_too_long = 0
    skipped_invalid = 0
    
    # Find all conversation directories
    conv_dirs = []
    for item in trajectory_dir.iterdir():
        if item.is_dir() and (item / "agent_calls.jsonl").exists():
            conv_dirs.append(item)
    
    if not conv_dirs:
        # Check if trajectory_dir itself contains agent_calls.jsonl
        if (trajectory_dir / "agent_calls.jsonl").exists():
            conv_dirs = [trajectory_dir]
    
    print(f"Found {len(conv_dirs)} conversation(s) with trajectories")
    
    for conv_dir in conv_dirs:
        conv_id = conv_dir.name
        print(f"Processing: {conv_id}")
        
        calls = load_trajectory(conv_dir)
        conv_samples = 0
        
        for call in calls:
            # Create SFT sample directly from the call
            sample = create_sft_sample(call)
            
            if sample is None:
                skipped_invalid += 1
                continue
            
            # Check length
            total_len = estimate_token_length(
                sample["instruction"] + sample["output"]
            )
            
            if total_len > max_length:
                skipped_too_long += 1
                continue
            
            all_samples.append(sample)
            conv_samples += 1
        
        print(f"  -> {conv_samples} samples extracted")
    
    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Skipped (too long): {skipped_too_long}")
    print(f"  Skipped (invalid): {skipped_invalid}")
    print(f"  Output: {output_file}")
    print(f"{'='*50}")
    
    return len(all_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Convert expert trajectories to SFT training format"
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        required=True,
        help="Directory containing expert trajectories (with agent_calls.jsonl)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output JSON file for SFT data"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=20000,
        help="Maximum token length per sample (default: 20000)"
    )
    
    args = parser.parse_args()
    
    trajectory_dir = Path(args.trajectory_dir)
    output_file = Path(args.output_file)
    
    if not trajectory_dir.exists():
        print(f"Error: Trajectory directory not found: {trajectory_dir}")
        sys.exit(1)
    
    convert_trajectories(trajectory_dir, output_file, args.max_length)


if __name__ == "__main__":
    main()
