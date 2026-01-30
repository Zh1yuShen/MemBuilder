#!/usr/bin/env python3
"""
Process raw PerLTQA dataset into standard evaluation format.

Raw data structure:
- perltmem_en_v2.json: Contains character memories (profile, dialogues, events, social_relationships)
- perltqa_en_v2.json: Contains QA pairs for each character

Output format (per character):
{
    "character_id": "char_001",
    "character_name": "Wang Xiaoming",
    "character_profile": {...},
    "haystack_sessions": [...],  # Converted from dialogues
    "haystack_session_datetimes": [...],
    "num_sessions": N,
    "questions": [
        {"question_id": "...", "question": "...", "answer": "...", "question_type": "..."},
        ...
    ],
    "num_questions": M
}
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def parse_dialogue_content(dialogue_data: dict) -> list:
    """Convert PerLTQA dialogue format to standard session format."""
    sessions = []
    
    contents = dialogue_data.get('contents', {})
    
    for timestamp, messages in sorted(contents.items()):
        session = {
            'messages': [],
            'timestamp': timestamp
        }
        
        for msg in messages:
            # Parse "Speaker: Content" format
            if ':' in msg:
                parts = msg.split(':', 1)
                speaker = parts[0].strip()
                content = parts[1].strip() if len(parts) > 1 else ""
                
                # Determine role (user/assistant based on position)
                role = 'user' if len(session['messages']) % 2 == 0 else 'assistant'
                session['messages'].append({
                    'role': role,
                    'content': f"{speaker}: {content}"
                })
            else:
                role = 'user' if len(session['messages']) % 2 == 0 else 'assistant'
                session['messages'].append({
                    'role': role,
                    'content': msg
                })
        
        if session['messages']:
            sessions.append(session)
    
    return sessions


def extract_character_profile(char_mem: dict) -> dict:
    """Extract character profile information."""
    profile = {}
    
    if 'profile' in char_mem and isinstance(char_mem['profile'], dict):
        for key, value in char_mem['profile'].items():
            if isinstance(value, str):
                profile[key.lower().replace(' ', '_')] = value
    
    if 'profile_description' in char_mem:
        profile['description'] = char_mem['profile_description']
    
    return profile


def process_perltqa(
    mem_path: str,
    qa_path: str,
    output_path: str,
    max_characters: int = None
):
    """Process PerLTQA dataset."""
    
    print(f"📂 Loading PerLTMem from: {mem_path}")
    with open(mem_path, 'r', encoding='utf-8') as f:
        mem_data = json.load(f)
    
    print(f"📂 Loading PerLTQA from: {qa_path}")
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    print(f"   Found {len(mem_data)} characters in memory data")
    print(f"   Found {len(qa_data)} character entries in QA data")
    
    processed = []
    char_id = 0
    
    for char_name in mem_data.keys():
        if max_characters and char_id >= max_characters:
            break
        
        char_mem = mem_data[char_name]
        
        # Find corresponding QA data
        char_qa = None
        for qa_entry in qa_data:
            if char_name in qa_entry:
                char_qa = qa_entry[char_name]
                break
        
        if not char_qa:
            print(f"   ⚠️ No QA data for {char_name}, skipping")
            continue
        
        # Convert dialogues to sessions
        sessions = []
        timestamps = []
        
        dialogues = char_mem.get('dialogues', {})
        for dial_key, dial_data in dialogues.items():
            dial_sessions = parse_dialogue_content(dial_data)
            for sess in dial_sessions:
                sessions.append(sess['messages'])
                timestamps.append(sess.get('timestamp', ''))
        
        # Also add events as context if available
        events = char_mem.get('events', {})
        for event_key, event_data in events.items():
            if isinstance(event_data, dict) and 'content' in event_data:
                event_session = [{
                    'role': 'user',
                    'content': f"[Event] {event_data.get('summary', '')}"
                }, {
                    'role': 'assistant', 
                    'content': event_data['content']
                }]
                sessions.append(event_session)
                timestamps.append(event_data.get('creation_time', ''))
        
        # Process questions
        questions = []
        q_id = 0
        
        for category in ['profile', 'social_relationship', 'events', 'dialogues']:
            if category in char_qa:
                category_data = char_qa[category]
                
                # profile 是直接的问题列表
                # 其他类型是嵌套结构: [{'key': [{Question: ...}, ...]}, ...]
                if category == 'profile':
                    # 直接是问题列表
                    for qa_item in category_data:
                        if isinstance(qa_item, dict) and 'Question' in qa_item:
                            questions.append({
                                'question_id': f"{char_name.replace(' ', '_')}_{category}_q{q_id}",
                                'question': qa_item.get('Question', ''),
                                'answer': qa_item.get('Answer', ''),
                                'question_type': category,
                                'reference_memory': qa_item.get('Reference Memory', '')
                            })
                            q_id += 1
                else:
                    # 嵌套结构: [{'memory_key': [qa_items...]}, ...]
                    for nested_item in category_data:
                        if isinstance(nested_item, dict):
                            for mem_key, qa_items in nested_item.items():
                                if isinstance(qa_items, list):
                                    for qa_item in qa_items:
                                        if isinstance(qa_item, dict) and 'Question' in qa_item:
                                            questions.append({
                                                'question_id': f"{char_name.replace(' ', '_')}_{category}_q{q_id}",
                                                'question': qa_item.get('Question', ''),
                                                'answer': qa_item.get('Answer', ''),
                                                'question_type': category,
                                                'reference_memory': qa_item.get('Reference Memory', ''),
                                                'memory_key': mem_key
                                            })
                                            q_id += 1
        
        if not questions:
            print(f"   ⚠️ No questions for {char_name}, skipping")
            continue
        
        # Build character entry
        char_entry = {
            'character_id': f'char_{char_id:03d}',
            'character_name': char_name,
            'character_profile': extract_character_profile(char_mem),
            'haystack_sessions': sessions,
            'haystack_session_datetimes': timestamps,
            'num_sessions': len(sessions),
            'questions': questions,
            'num_questions': len(questions)
        }
        
        processed.append(char_entry)
        print(f"   ✅ {char_name}: {len(sessions)} sessions, {len(questions)} questions")
        char_id += 1
    
    # Save processed data
    print(f"\n💾 Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_sessions = sum(c['num_sessions'] for c in processed)
    total_questions = sum(c['num_questions'] for c in processed)
    
    print(f"\n📊 Statistics:")
    print(f"   Characters: {len(processed)}")
    print(f"   Total sessions: {total_sessions}")
    print(f"   Total questions: {total_questions}")
    print(f"   Avg sessions/character: {total_sessions/len(processed):.1f}")
    print(f"   Avg questions/character: {total_questions/len(processed):.1f}")
    
    return processed


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process PerLTQA dataset')
    parser.add_argument('--mem-path', default=None, help='Path to perltmem file')
    parser.add_argument('--qa-path', default=None, help='Path to perltqa file')
    parser.add_argument('--output', default=None, help='Output path')
    parser.add_argument('--max-characters', type=int, default=None, help='Max characters to process')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    
    mem_path = args.mem_path or base_dir / 'data' / 'perltqa_raw' / 'Dataset' / 'en_v2' / 'perltmem_en_v2.json'
    qa_path = args.qa_path or base_dir / 'data' / 'perltqa_raw' / 'Dataset' / 'en_v2' / 'perltqa_en_v2.json'
    output_path = args.output or base_dir / 'data' / 'perltqa' / 'perltqa_v2_standard.json'
    
    process_perltqa(str(mem_path), str(qa_path), str(output_path), args.max_characters)


if __name__ == '__main__':
    main()
