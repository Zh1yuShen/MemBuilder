"""
Attributed Dense Reward Policy Optimization (ADRPO) for MemBuilder.

This module implements the ADRPO algorithm as described in the paper:
1. Dense session-level rewards via synthetic QA
2. Contribution-aware gradient weighting for multi-dimensional memory

Based on GRPO with extensions for memory construction.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config import ADRPO_CONFIG


def main() -> None:
    raise SystemExit(
        "This repository launches verl training via scripts/run_memory_grpo_multinode.sh (sh-only entrypoint)."
    )


@dataclass
class RolloutResult:
    """Result from a single rollout."""
    memories: Dict[str, List[str]]  # Memories by type
    actions: Dict[str, List[Dict]]  # Actions by agent
    log_probs: Dict[str, float]     # Log probabilities by agent


def compute_reward(
    qa_pairs: List[Dict],
    memory_bank: Dict,
    answer_func,
    judge_func,
    expert_memory_lengths: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Compute session-level reward for a rollout.
    
    Reward = validity_gate * task_reward * (1 - lambda * length_penalty)
    
    Args:
        qa_pairs: List of QA dicts with 'question' and 'answer'
        memory_bank: Current memory state after rollout
        answer_func: Function to generate answers given question and memories
        judge_func: Function to evaluate answer correctness
        expert_memory_lengths: Optional expert memory lengths for penalty
    
    Returns:
        Tuple of (reward, retrieval_stats)
    """
    config = ADRPO_CONFIG
    
    # Check format validity
    if not _validate_memory_format(memory_bank):
        return 0.0, {"valid": False}
    
    # Compute task reward (QA accuracy)
    correct = 0
    retrieval_counts = {"episodic": 0, "semantic": 0, "procedural": 0}
    
    for qa in qa_pairs:
        question = qa["question"]
        gold_answer = qa["answer"]
        
        # Generate answer and track retrieved memories
        generated_answer, retrieved = answer_func(question, memory_bank)
        
        # Count retrievals by type
        for mem in retrieved:
            for mem_type in retrieval_counts:
                if f"[{mem_type.upper()}]" in mem:
                    retrieval_counts[mem_type] += 1
                    break
        
        # Judge correctness
        correct += judge_func(question, gold_answer, generated_answer)
    
    task_reward = correct / len(qa_pairs) if qa_pairs else 0.0
    
    # Compute length penalty
    length_penalty = _compute_length_penalty(memory_bank, expert_memory_lengths, config)
    
    # Final reward
    reward = task_reward * (1 - config["lambda_length"] * length_penalty)
    
    return reward, {
        "valid": True,
        "task_reward": task_reward,
        "length_penalty": length_penalty,
        "retrieval_counts": retrieval_counts,
        "correct": correct,
        "total": len(qa_pairs)
    }


def _validate_memory_format(memory_bank: Dict) -> bool:
    """Check if memory bank has valid format."""
    required_keys = ["core", "episodic", "semantic", "procedural"]
    return all(k in memory_bank for k in required_keys)


def _compute_length_penalty(
    memory_bank: Dict,
    expert_lengths: Optional[Dict],
    config: Dict
) -> float:
    """
    Compute length penalty for memory content.
    
    Core memory: Penalize based on character increase
    Other memories: Penalize deviation from expert length ratio
    """
    if expert_lengths is None:
        return 0.0
    
    penalties = []
    
    # Core memory penalty
    core_len = len(memory_bank.get("core", ""))
    if core_len > config["theta_max"]:
        penalties.append(1.0)
    elif core_len > config["theta_min"]:
        penalties.append(
            (core_len - config["theta_min"]) / 
            (config["theta_max"] - config["theta_min"])
        )
    else:
        penalties.append(0.0)
    
    # Other memories penalty
    for mem_type in ["episodic", "semantic", "procedural"]:
        current = sum(len(m) for m in memory_bank.get(mem_type, []))
        expert = expert_lengths.get(mem_type, current)
        
        if expert == 0:
            penalties.append(0.0)
            continue
        
        ratio = current / expert
        
        if config["gamma_l"] <= ratio <= config["gamma_u"]:
            penalties.append(0.0)
        elif ratio > config["gamma_u"]:
            pen = min(1.0, (ratio - config["gamma_u"]) / (2.0 - config["gamma_u"]))
            penalties.append(pen)
        else:
            pen = min(1.0, (config["gamma_l"] - ratio) / config["gamma_l"])
            penalties.append(pen)
    
    return sum(penalties) / len(penalties) if penalties else 0.0


def compute_advantage(rewards: List[float], epsilon: float = 1e-8) -> List[float]:
    """
    Compute advantages via within-group normalization.
    
    A_i = (r_i - mean) / (std + epsilon)
    """
    import numpy as np
    rewards = np.array(rewards)
    mean = np.mean(rewards)
    std = np.std(rewards)
    return ((rewards - mean) / (std + epsilon)).tolist()


def compute_gradient_weights(retrieval_counts: Dict[str, int], alpha: float) -> Dict[str, float]:
    """
    Compute contribution-aware gradient weights.
    
    The dominant contributing memory type gets weight alpha,
    others get weight 1.
    
    Args:
        retrieval_counts: Dict mapping memory type to retrieval count
        alpha: Weight amplification factor (>1)
    
    Returns:
        Dict mapping memory type to gradient weight
    """
    weights = {"core": 1.0}  # Core always weight 1 (not retrieved)
    
    # Find dominant type among searchable memories
    searchable_types = ["episodic", "semantic", "procedural"]
    counts = {t: retrieval_counts.get(t, 0) for t in searchable_types}
    
    if sum(counts.values()) == 0:
        # No retrievals, uniform weights
        for t in searchable_types:
            weights[t] = 1.0
    else:
        dominant = max(counts, key=counts.get)
        for t in searchable_types:
            weights[t] = alpha if t == dominant else 1.0
    
    return weights


class ADRPOTrainer:
    """
    ADRPO Trainer for memory construction.
    
    Extends GRPO with:
    - Dense session-level rewards
    - Contribution-aware gradient weighting
    """
    
    def __init__(
        self,
        model,
        ref_model,
        memory_system,
        qa_generator,
        answer_model,
        judge_model,
        config: Optional[Dict] = None
    ):
        """
        Initialize ADRPO trainer.
        
        Args:
            model: Policy model to train
            ref_model: Reference model for KL penalty
            memory_system: Memory system instance
            qa_generator: QA generator for dense rewards
            answer_model: Model for answering QA
            judge_model: Model for judging answers
            config: Optional config overrides
        """
        self.model = model
        self.ref_model = ref_model
        self.memory_system = memory_system
        self.qa_generator = qa_generator
        self.answer_model = answer_model
        self.judge_model = judge_model
        self.config = {**ADRPO_CONFIG, **(config or {})}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        """
        Perform single training step.
        
        Args:
            batch: List of training samples with session and QA data
        
        Returns:
            Dict with loss and metrics
        """
        all_losses = []
        all_rewards = []
        
        for sample in batch:
            session = sample["session"]
            qa_pairs = sample["questions"]
            state_before = sample["state_before"]
            
            # Sample N rollouts
            rollouts = []
            rewards = []
            retrieval_stats = []
            
            for _ in range(self.config["num_rollouts"]):
                # Generate rollout with current policy
                rollout = self._generate_rollout(session, state_before)
                rollouts.append(rollout)
                
                # Compute reward
                reward, stats = self._compute_rollout_reward(rollout, qa_pairs)
                rewards.append(reward)
                retrieval_stats.append(stats)
            
            # Compute advantages
            advantages = compute_advantage(rewards)
            
            # Compute policy gradient with contribution-aware weighting
            for i, (rollout, advantage, stats) in enumerate(
                zip(rollouts, advantages, retrieval_stats)
            ):
                if not stats.get("valid", False):
                    continue
                
                # Get gradient weights
                weights = compute_gradient_weights(
                    stats.get("retrieval_counts", {}),
                    self.config["alpha"]
                )
                
                # Compute loss for each agent
                loss = self._compute_loss(rollout, advantage, weights)
                all_losses.append(loss)
            
            all_rewards.extend(rewards)
        
        return {
            "loss": sum(all_losses) / len(all_losses) if all_losses else 0.0,
            "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "num_samples": len(batch)
        }
    
    def _generate_rollout(self, session: Dict, state_before: Dict) -> RolloutResult:
        """Generate single rollout with current policy."""
        # This would call the policy model to generate memory operations
        # Placeholder implementation
        messages = session.get("messages", [])
        
        memories = {
            "core": state_before.get("core_memory", ""),
            "episodic": [],
            "semantic": [],
            "procedural": []
        }
        
        actions = {
            "core": [],
            "episodic": [],
            "semantic": [],
            "procedural": []
        }
        
        log_probs = {
            "core": 0.0,
            "episodic": 0.0,
            "semantic": 0.0,
            "procedural": 0.0
        }
        
        return RolloutResult(
            memories=memories,
            actions=actions,
            log_probs=log_probs
        )
    
    def _compute_rollout_reward(
        self, 
        rollout: RolloutResult, 
        qa_pairs: List[Dict]
    ) -> Tuple[float, Dict]:
        """Compute reward for a rollout."""
        
        def answer_func(question, memory_bank):
            # Generate answer using answer model
            # Placeholder - would use actual retrieval and generation
            return "Generated answer", []
        
        def judge_func(question, gold, generated):
            # Judge answer correctness
            # Placeholder - would use actual LLM judge
            return 1 if gold.lower() in generated.lower() else 0
        
        return compute_reward(
            qa_pairs=qa_pairs,
            memory_bank=rollout.memories,
            answer_func=answer_func,
            judge_func=judge_func
        )
    
    def _compute_loss(
        self,
        rollout: RolloutResult,
        advantage: float,
        weights: Dict[str, float]
    ) -> float:
        """
        Compute ADRPO loss with contribution-aware weighting.
        
        Loss = -sum_m [ w_m * min(ratio * A, clip(ratio) * A) ] + beta * KL
        """
        config = self.config
        total_loss = 0.0
        
        for agent_type in ["core", "episodic", "semantic", "procedural"]:
            log_prob = rollout.log_probs[agent_type]
            weight = weights.get(agent_type, 1.0)
            
            # Compute importance ratio (placeholder)
            ratio = 1.0  # Would be exp(log_prob - ref_log_prob)
            
            # Clipped objective
            clipped_ratio = max(
                1 - config["clip_epsilon"],
                min(1 + config["clip_epsilon"], ratio)
            )
            
            # Contribution-aware: weight only applies to unclipped term
            unclipped_obj = weight * ratio * advantage
            clipped_obj = clipped_ratio * advantage
            
            agent_loss = -min(unclipped_obj, clipped_obj)
            total_loss += agent_loss
        
        return total_loss / 4  # Average over agents


if __name__ == "__main__":
    main()
