"""
Constraints Checker for Memory Agent Outputs.

Checks if policy model outputs satisfy length constraints:
1. Core memory length (not exceeding word limit)
2. New memories total length (not exceeding session length ratio)

Returns 0.0 - 0.3 score based on constraints satisfaction.
"""

from typing import Dict, List


class ConstraintsChecker:
    """Constraints reward checker for memory outputs."""
    
    # Core memory length limit (words)
    CORE_MEMORY_MAX_WORDS = 200
    
    # New memories length ratio threshold (relative to session length)
    NEW_MEMORIES_RATIO_THRESHOLD = 1.5
    
    # Reward weights
    CORE_LENGTH_REWARD = 0.1
    MEMORIES_LENGTH_REWARD = 0.2
    
    def __init__(self):
        pass
    
    def compute_constraints_reward(
        self,
        parsed_actions: Dict,
        session_messages: List[Dict],
        agent_type: str = None
    ) -> float:
        """
        Compute Constraints Reward.
        
        Args:
            parsed_actions: Parsed actions dict (single or multi-agent)
            session_messages: Current session messages
            agent_type: Optional, specify agent type for single-agent mode
            
        Returns:
            float: 0.0 - 0.3 score
        """
        reward = 0.0
        
        # Normalize parsed_actions structure
        normalized = parsed_actions
        if agent_type and agent_type not in parsed_actions:
            normalized = {agent_type: parsed_actions}
        
        # Check core memory length
        if self._check_core_length(normalized.get('core', {})):
            reward += self.CORE_LENGTH_REWARD
        
        # Check new memories total length
        if self._check_memories_length(normalized, session_messages):
            reward += self.MEMORIES_LENGTH_REWARD
        
        return reward
    
    def _check_core_length(self, core_data: Dict) -> bool:
        """Check if core memory length is acceptable."""
        if not isinstance(core_data, dict):
            return False
        
        op = core_data.get('operation', '')
        if op == 'REPLACE':
            core_text = core_data.get('new_text', '')
        else:
            core_text = core_data.get('content', '')
        
        word_count = len(core_text.split())
        return word_count <= self.CORE_MEMORY_MAX_WORDS
    
    def _check_memories_length(
        self,
        parsed_actions: Dict,
        session_messages: List[Dict]
    ) -> bool:
        """Check if new memories total length is acceptable."""
        # Calculate session total length (words)
        session_word_count = 0
        for msg in session_messages:
            content = msg.get('content', '')
            session_word_count += len(content.split())
        
        # Calculate new memories total length
        total_new_words = 0
        
        for memory_type in ['episodic', 'semantic', 'procedural']:
            if memory_type in parsed_actions:
                agent_data = parsed_actions[memory_type]
                
                # Training format: operations list
                if 'operations' in agent_data:
                    for op in agent_data['operations']:
                        if op.get('action') == 'ADD' and 'memory' in op:
                            total_new_words += len(op['memory'].split())
                        elif op.get('action') in ['UPDATE', 'MERGE'] and 'new_memory' in op:
                            total_new_words += len(op['new_memory'].split())
                
                # Backward compatible: memories_added
                elif 'memories_added' in agent_data:
                    for memory in agent_data['memories_added']:
                        if isinstance(memory, str):
                            total_new_words += len(memory.split())
                        elif isinstance(memory, dict):
                            text = memory.get('text', str(memory))
                            total_new_words += len(text.split())
        
        # Check if exceeds threshold
        if session_word_count == 0:
            threshold = 500  # Default threshold
        else:
            threshold = session_word_count * self.NEW_MEMORIES_RATIO_THRESHOLD
        
        return total_new_words <= threshold
    
    def get_detailed_feedback(
        self,
        parsed_actions: Dict,
        session_messages: List[Dict]
    ) -> Dict:
        """Get detailed constraint check feedback."""
        feedback = {
            'total_reward': 0.0,
            'core_length_check': {},
            'memories_length_check': {}
        }
        
        # Core memory length check
        core_data = parsed_actions.get('core', {})
        op = core_data.get('operation', '')
        if op == 'REPLACE':
            core_text = core_data.get('new_text', '')
        else:
            core_text = core_data.get('content', '')
        core_word_count = len(core_text.split())
        core_ok = core_word_count <= self.CORE_MEMORY_MAX_WORDS
        
        feedback['core_length_check'] = {
            'word_count': core_word_count,
            'limit': self.CORE_MEMORY_MAX_WORDS,
            'passed': core_ok,
            'ratio': core_word_count / self.CORE_MEMORY_MAX_WORDS if self.CORE_MEMORY_MAX_WORDS > 0 else 0
        }
        
        if core_ok:
            feedback['total_reward'] += self.CORE_LENGTH_REWARD
        
        # New memories length check
        session_word_count = sum(len(msg.get('content', '').split()) for msg in session_messages)
        
        total_new_words = 0
        memory_breakdown = {}
        
        for memory_type in ['episodic', 'semantic', 'procedural']:
            if memory_type in parsed_actions:
                agent_data = parsed_actions[memory_type]
                type_word_count = 0
                
                if 'operations' in agent_data:
                    for op in agent_data['operations']:
                        if op.get('action') == 'ADD' and 'memory' in op:
                            type_word_count += len(op['memory'].split())
                        elif op.get('action') in ['UPDATE', 'MERGE'] and 'new_memory' in op:
                            type_word_count += len(op['new_memory'].split())
                
                total_new_words += type_word_count
                memory_breakdown[memory_type] = type_word_count
        
        threshold = session_word_count * self.NEW_MEMORIES_RATIO_THRESHOLD if session_word_count > 0 else 500
        memories_ok = total_new_words <= threshold
        
        feedback['memories_length_check'] = {
            'session_word_count': session_word_count,
            'total_new_words': total_new_words,
            'threshold': threshold,
            'passed': memories_ok,
            'ratio': total_new_words / session_word_count if session_word_count > 0 else 0,
            'breakdown': memory_breakdown
        }
        
        if memories_ok:
            feedback['total_reward'] += self.MEMORIES_LENGTH_REWARD
        
        return feedback
