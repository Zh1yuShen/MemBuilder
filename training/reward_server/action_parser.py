"""
Policy Model Output Parser.

Parses memory management actions from policy model outputs.
Supports multiple formats: JSON, structured text, markdown-wrapped JSON.

Output structure:
{
    "core": {
        "operation": "APPEND" | "REPLACE" | "REWRITE",
        "content": "..."  // APPEND/REWRITE
        "old_text": "...", "new_text": "..."  // REPLACE
    },
    "episodic": {
        "operations": [
            {"action": "ADD", "memory": "..."},
            {"action": "UPDATE", "old_memory": "...", "new_memory": "..."},
            {"action": "MERGE", "old_memories": [...], "new_memory": "..."},
            {"action": "SKIP", "reason": "..."}
        ]
    },
    "semantic": { ... },
    "procedural": { ... }
}
"""

import json
import re
from typing import Dict, List, Optional


class ActionParser:
    """Parse policy model outputs into structured actions."""
    
    ALL_AGENT_TYPES = ['core', 'episodic', 'semantic', 'procedural']
    VALID_CORE_OPS = ['APPEND', 'REPLACE', 'REWRITE']
    VALID_MEMORY_ACTIONS = ['ADD', 'UPDATE', 'MERGE', 'SKIP']
    
    def __init__(self):
        pass
    
    def parse(self, response: str) -> Dict:
        """
        Parse response into actions dict.
        
        Args:
            response: Policy model output text
            
        Returns:
            Dict: Parsed actions
            
        Raises:
            ValueError: If parsing fails
        """
        # Clean response
        response = self._clean_response(response)
        
        # Try JSON parsing
        try:
            actions = json.loads(response)
            if self._validate_actions(actions):
                return actions
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON block (may be markdown-wrapped)
        try:
            actions = self._extract_json_block(response)
            if actions and self._validate_actions(actions):
                return actions
        except:
            pass
        
        # Try structured text parsing
        try:
            actions = self._parse_structured_text(response)
            if self._validate_actions(actions):
                return actions
        except:
            pass
        
        raise ValueError("Cannot parse policy model output. Ensure valid JSON or structured format.")
    
    def _clean_response(self, response: str) -> str:
        """Clean response of model meta-tags."""
        if not response:
            return ""
        
        clean = response.strip()
        
        # Remove <think> tags
        if '<think>' in clean and '</think>' in clean:
            clean = clean.split('</think>')[-1].strip()
        
        # Remove <thinking> tags
        clean = re.sub(r'<thinking>.*?</thinking>', '', clean, flags=re.DOTALL)
        
        # Remove markdown code block markers
        clean = re.sub(r'^```(?:json)?\s*', '', clean)
        clean = re.sub(r'\s*```$', '', clean)
        
        return clean.strip()
    
    def _extract_json_block(self, response: str) -> Optional[Dict]:
        """Extract JSON from response."""
        # Find JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        
        # Try finding larger nested JSON
        brace_count = 0
        start_idx = None
        
        for i, char in enumerate(response):
            if char == '{':
                if start_idx is None:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    try:
                        return json.loads(response[start_idx:i+1])
                    except:
                        pass
                    start_idx = None
        
        return None
    
    def _parse_structured_text(self, response: str) -> Dict:
        """Parse structured text format."""
        actions = {}
        
        # Look for agent sections
        for agent_type in self.ALL_AGENT_TYPES:
            pattern = rf'{agent_type}[:\s]*(.+?)(?={"|".join(self.ALL_AGENT_TYPES)}[:\s]|$)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                section = match.group(1).strip()
                
                if agent_type == 'core':
                    actions[agent_type] = self._parse_core_section(section)
                else:
                    actions[agent_type] = self._parse_memory_section(section)
        
        return actions
    
    def _parse_core_section(self, section: str) -> Dict:
        """Parse core memory section."""
        # Look for operation type
        for op in self.VALID_CORE_OPS:
            if op in section.upper():
                if op == 'APPEND':
                    content_match = re.search(r'content[:\s]*["\']?(.+?)["\']?\s*(?:$|\n)', section, re.IGNORECASE)
                    return {
                        'operation': 'APPEND',
                        'content': content_match.group(1) if content_match else ''
                    }
                elif op == 'REPLACE':
                    old_match = re.search(r'old[_\s]?text[:\s]*["\']?(.+?)["\']?(?:,|\n|new)', section, re.IGNORECASE)
                    new_match = re.search(r'new[_\s]?text[:\s]*["\']?(.+?)["\']?\s*(?:$|\n|,|})', section, re.IGNORECASE)
                    return {
                        'operation': 'REPLACE',
                        'old_text': old_match.group(1) if old_match else '',
                        'new_text': new_match.group(1) if new_match else ''
                    }
                elif op == 'REWRITE':
                    content_match = re.search(r'content[:\s]*["\']?(.+?)["\']?\s*(?:$|\n)', section, re.IGNORECASE)
                    return {
                        'operation': 'REWRITE',
                        'content': content_match.group(1) if content_match else ''
                    }
        
        # Default to APPEND with section as content
        return {'operation': 'APPEND', 'content': section}
    
    def _parse_memory_section(self, section: str) -> Dict:
        """Parse episodic/semantic/procedural section."""
        operations = []
        
        for action in self.VALID_MEMORY_ACTIONS:
            pattern = rf'{action}[:\s]*(.+?)(?={"|".join(self.VALID_MEMORY_ACTIONS)}[:\s]|$)'
            matches = re.finditer(pattern, section, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                content = match.group(1).strip()
                if action == 'ADD':
                    operations.append({'action': 'ADD', 'memory': content})
                elif action == 'UPDATE':
                    # Try to extract old and new
                    operations.append({
                        'action': 'UPDATE',
                        'old_memory': '',
                        'new_memory': content
                    })
                elif action == 'MERGE':
                    operations.append({
                        'action': 'MERGE',
                        'old_memories': [],
                        'new_memory': content
                    })
                elif action == 'SKIP':
                    operations.append({'action': 'SKIP', 'reason': content})
        
        return {'operations': operations}
    
    def _validate_actions(self, actions: Dict) -> bool:
        """Validate actions structure."""
        if not isinstance(actions, dict):
            return False
        
        # Must have at least one agent type
        has_any = False
        
        for agent_type in self.ALL_AGENT_TYPES:
            if agent_type in actions:
                has_any = True
                agent_data = actions[agent_type]
                
                if agent_type == 'core':
                    # Core must have operation
                    if not isinstance(agent_data, dict):
                        return False
                    op = agent_data.get('operation', '').upper()
                    if op and op not in self.VALID_CORE_OPS:
                        return False
                else:
                    # Memory agents must have operations list
                    if not isinstance(agent_data, dict):
                        return False
                    ops = agent_data.get('operations', [])
                    if not isinstance(ops, list):
                        return False
        
        return has_any
