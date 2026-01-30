"""
Format Checker for Memory Agent Outputs.

Validates that policy model outputs follow the correct JSON format
for each memory agent type.
"""

import json
import re
from typing import Dict, Optional, Tuple


class FormatChecker:
    """Format validation for memory agent outputs."""
    
    REQUIRED_AGENTS = ['core', 'episodic', 'semantic', 'procedural']
    
    VALID_CORE_OPS = ['APPEND', 'REPLACE', 'REWRITE']
    
    VALID_ACTIONS_BY_AGENT = {
        'episodic': ['ADD', 'UPDATE', 'MERGE'],
        'semantic': ['ADD', 'UPDATE', 'SKIP'],
        'procedural': ['ADD', 'UPDATE']
    }
    
    def __init__(self):
        pass
    
    def _clean_think_tags(self, response: str) -> str:
        """Remove <think> tags from response."""
        if not response:
            return response
        
        clean = response.strip()
        
        if '<think>' in clean and '</think>' in clean:
            clean = clean.split('</think>')[-1].strip()
        
        clean = re.sub(r'<thinking>.*?</thinking>', '', clean, flags=re.DOTALL)
        
        return clean.strip()
    
    def check_format(self, response: str, agent_type: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if response has valid format for the given agent type.
        
        Args:
            response: Raw model output
            agent_type: One of 'core', 'episodic', 'semantic', 'procedural'
        
        Returns:
            Tuple of (is_valid, parsed_dict or None)
        """
        if not response:
            return False, None
        
        clean = self._clean_think_tags(response)
        
        # Remove markdown code blocks
        clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', clean).strip()
        
        # Extract JSON object
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not json_match:
            return False, None
        
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return False, None
        
        # Validate by agent type
        if agent_type == 'core':
            return self._validate_core(parsed)
        else:
            return self._validate_memory_agent(parsed, agent_type)
    
    def _validate_core(self, parsed: Dict) -> Tuple[bool, Optional[Dict]]:
        """Validate core memory agent output."""
        operation = parsed.get('operation', '').upper()
        
        if operation not in self.VALID_CORE_OPS:
            return False, None
        
        if operation == 'APPEND':
            if 'content' not in parsed:
                return False, None
        elif operation == 'REPLACE':
            if 'old_text' not in parsed or 'new_text' not in parsed:
                return False, None
        elif operation == 'REWRITE':
            if 'content' not in parsed:
                return False, None
        
        return True, parsed
    
    def _validate_memory_agent(self, parsed: Dict, agent_type: str) -> Tuple[bool, Optional[Dict]]:
        """Validate episodic/semantic/procedural agent output."""
        operations = parsed.get('operations', [])
        
        if not isinstance(operations, list):
            return False, None
        
        valid_actions = self.VALID_ACTIONS_BY_AGENT.get(agent_type, [])
        
        for op in operations:
            action = op.get('action', '').upper()
            if action not in valid_actions:
                return False, None
            
            # Validate required fields
            if action in ['ADD']:
                if 'memory' not in op:
                    return False, None
            elif action in ['UPDATE']:
                if 'old_memory' not in op or 'new_memory' not in op:
                    return False, None
            elif action == 'MERGE':
                if 'old_memories' not in op or 'new_memory' not in op:
                    return False, None
            elif action == 'SKIP':
                if 'reason' not in op:
                    return False, None
        
        return True, parsed
