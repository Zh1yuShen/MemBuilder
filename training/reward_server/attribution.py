"""
Attribution Weight Manager for ADRPO.

Manages contribution-aware gradient weights that are passed between
the reward function and the verl training loop.

As described in the paper (Section 3.4.2), the dominant contributing
memory type receives an amplified gradient weight (alpha) while others
receive weight 1.
"""

import threading
from typing import Dict, Optional

_ATTRIBUTION_LOCK = threading.Lock()
_ATTRIBUTION_WEIGHTS: Dict[int, float] = {}
_PATCH_APPLIED = False


class AttributionWeightManager:
    """
    Manages attribution weights for contribution-aware gradient updates.
    
    Thread-safe singleton for passing weights between reward computation
    and the PPO/GRPO training loop.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._weights = {}
                    cls._instance._enabled = True
        return cls._instance
    
    def set_weights(self, batch_weights: Dict[int, Dict[str, float]]):
        """
        Set attribution weights for current batch.
        
        Args:
            batch_weights: {sample_idx: {agent_type: weight}}
                          e.g., {0: {'core': 1.0, 'episodic': 4.0, 'semantic': 1.0, 'procedural': 1.0}}
        """
        with self._lock:
            self._weights = batch_weights.copy()
    
    def get_weights(self, sample_idx: int) -> Dict[str, float]:
        """
        Get weights for a specific sample.
        
        Args:
            sample_idx: Index of sample in batch
        
        Returns:
            Dict mapping agent type to gradient weight
        """
        with self._lock:
            default = {'core': 1.0, 'episodic': 1.0, 'semantic': 1.0, 'procedural': 1.0}
            return self._weights.get(sample_idx, default)
    
    def clear(self):
        """Clear all weights after batch processing."""
        with self._lock:
            self._weights = {}
    
    def set_enabled(self, enabled: bool):
        """Enable or disable attribution weighting."""
        self._enabled = enabled
    
    def is_enabled(self) -> bool:
        """Check if attribution is enabled."""
        return self._enabled


def set_attribution_weights(weights: Dict[int, float]) -> None:
    global _ATTRIBUTION_WEIGHTS
    with _ATTRIBUTION_LOCK:
        _ATTRIBUTION_WEIGHTS = weights.copy()


def get_attribution_weights() -> Dict[int, float]:
    with _ATTRIBUTION_LOCK:
        return _ATTRIBUTION_WEIGHTS.copy()


def clear_attribution_weights() -> None:
    global _ATTRIBUTION_WEIGHTS
    with _ATTRIBUTION_LOCK:
        _ATTRIBUTION_WEIGHTS = {}


def apply_verl_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    try:
        import torch
        import verl.trainer.ppo.core_algos as core_algos

        original_fn = core_algos.compute_grpo_outcome_advantage

        def patched_compute_grpo_outcome_advantage(
            token_level_rewards,
            response_mask,
            index,
            norm_adv_by_std_in_grpo=True,
        ):
            advantages, returns = original_fn(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )

            attr_weights = get_attribution_weights()
            if attr_weights:
                batch_size = advantages.shape[0]
                max_idx = max(attr_weights.keys()) if attr_weights else -1

                if max_idx < batch_size and len(attr_weights) == batch_size:
                    weights = torch.ones(batch_size, device=advantages.device)
                    for idx, w in attr_weights.items():
                        weights[idx] = w
                    advantages = advantages * weights.unsqueeze(-1)

                clear_attribution_weights()

            return advantages, returns

        core_algos.compute_grpo_outcome_advantage = patched_compute_grpo_outcome_advantage
        _PATCH_APPLIED = True
    except Exception:
        return


def compute_attribution_weights(
    retrieval_counts: Dict[str, int],
    alpha: float = 4.0
) -> Dict[str, float]:
    """
    Compute contribution-aware gradient weights based on retrieval counts.
    
    The memory type that contributed most to answering QA pairs receives
    weight alpha, while others receive weight 1.
    
    Args:
        retrieval_counts: Dict mapping memory type to retrieval count
                         e.g., {'episodic': 5, 'semantic': 2, 'procedural': 0}
        alpha: Amplification factor for dominant type (default: 4.0)
    
    Returns:
        Dict mapping memory type to gradient weight
    """
    weights = {'core': 1.0}  # Core is always 1 (not retrieved)
    
    searchable_types = ['episodic', 'semantic', 'procedural']
    counts = {t: retrieval_counts.get(t, 0) for t in searchable_types}
    
    total = sum(counts.values())
    
    if total == 0:
        # No retrievals - uniform weights
        for t in searchable_types:
            weights[t] = 1.0
    else:
        # Find dominant type
        dominant = max(counts, key=counts.get)
        for t in searchable_types:
            weights[t] = alpha if t == dominant else 1.0
    
    return weights


apply_verl_patch()
