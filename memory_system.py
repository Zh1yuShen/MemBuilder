"""
Multi-dimensional Memory System for MemBuilder.

This module implements the core memory architecture with four specialized agents:
- Core Memory: Persistent user profile (always included in context)
- Episodic Memory: Time-stamped event records
- Semantic Memory: Factual knowledge about entities
- Procedural Memory: Step-by-step processes and workflows

Note: This implementation uses FAISS for vector storage instead of mem0.
The original codebase used mem0 (https://github.com/mem0ai/mem0) for memory storage,
but we provide a standalone FAISS-based VectorStore for simplicity and to avoid
external dependencies. The memory construction logic (prompts, agents, operations)
remains the same as described in the paper.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from jinja2 import Template
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from config import (
    CORE_MEMORY_CHAR_LIMIT, MEMORY_PREFIXES, 
    MEMORY_CONSTRUCTION_TOP_K, QA_ANSWERING_TOP_K, DEFAULT_TOP_K
)
from prompts import (
    CORE_MEMORY_PROMPT, EPISODIC_MEMORY_PROMPT, 
    SEMANTIC_MEMORY_PROMPT, PROCEDURAL_MEMORY_PROMPT,
    CORE_MEMORY_COMPRESS_PROMPT, ANSWER_GENERATION_PROMPT
)


@dataclass
class CoreMemoryBlock:
    """Core Memory Block - persistent user profile."""
    human: str = ""
    
    def get_usage(self) -> float:
        """Get percentage of block usage."""
        return len(self.human) / CORE_MEMORY_CHAR_LIMIT * 100
    
    def needs_rewrite(self) -> bool:
        """Check if block needs rewriting (>90% full)."""
        return self.get_usage() > 90


class VectorStore:
    """Simple vector store using FAISS for memory retrieval."""
    
    def __init__(self, embedding_func, dimension: int = 1536):
        """
        Initialize vector store.
        
        Args:
            embedding_func: Function to generate embeddings
            dimension: Embedding dimension
        """
        import faiss
        self.embedding_func = embedding_func
        self._faiss = faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.memories = []  # Store memory texts
        self.metadata = []  # Store metadata

    def save(self, output_dir: str) -> None:
        """Persist FAISS index + memory payload to disk."""
        os.makedirs(output_dir, exist_ok=True)

        index_path = os.path.join(output_dir, "index.faiss")
        payload_path = os.path.join(output_dir, "payload.json")

        self._faiss.write_index(self.index, index_path)

        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": self.dimension,
                    "memories": self.memories,
                    "metadata": self.metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, input_dir: str) -> None:
        """Load FAISS index + memory payload from disk (in-place)."""
        index_path = os.path.join(input_dir, "index.faiss")
        payload_path = os.path.join(input_dir, "payload.json")

        if not os.path.exists(index_path) or not os.path.exists(payload_path):
            raise FileNotFoundError(f"VectorStore files not found in: {input_dir}")

        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.dimension = int(payload.get("dimension", self.dimension))
        self.index = self._faiss.read_index(index_path)
        self.memories = list(payload.get("memories", []))
        self.metadata = list(payload.get("metadata", []))
    
    def add(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add memories to the vector store."""
        import numpy as np
        embeddings = self.embedding_func(texts)
        vectors = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity
        self._faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.memories.extend(texts)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in texts])
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for similar memories."""
        if not self.memories:
            return []
        import numpy as np
        query_embedding = self.embedding_func([query])[0]
        query_vector = np.array([query_embedding], dtype=np.float32)
        self._faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, min(top_k, len(self.memories)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "memory": self.memories[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx]
                })
        return results


class MemorySystem:
    """
    Multi-dimensional Memory System aligned with paper architecture.
    
    Features:
    - Four specialized agents (Core, Episodic, Semantic, Procedural)
    - Parallel agent processing
    - Vector-based memory retrieval
    """
    
    def __init__(self, llm_client, embedding_func=None):
        """
        Initialize the memory system.
        
        Args:
            llm_client: LLM client for agent operations
            embedding_func: Optional custom embedding function
        """
        self.llm_client = llm_client
        
        # Set up embedding function
        if embedding_func:
            self.embedding_func = embedding_func
        else:
            self.embedding_func = lambda texts: llm_client.get_embeddings(texts)
        
        # Initialize vector store for searchable memories
        self.vector_store = VectorStore(self.embedding_func)
        
        # Initialize Core Memory Block (persistent, not searchable)
        self.core_memory = CoreMemoryBlock()
        
        # Initialize prompt templates
        self.prompts = {
            "core": CORE_MEMORY_PROMPT,
            "episodic": EPISODIC_MEMORY_PROMPT,
            "semantic": SEMANTIC_MEMORY_PROMPT,
            "procedural": PROCEDURAL_MEMORY_PROMPT
        }
    
    def save(self, db_path: str) -> None:
        """
        Save memory system to disk.
        
        Persists:
        - FAISS vector index
        - Memory texts and metadata
        - Core memory block
        
        Args:
            db_path: Directory path to save the memory database
        """
        os.makedirs(db_path, exist_ok=True)
        
        # Save vector store (FAISS index + payload)
        self.vector_store.save(db_path)
        
        # Save core memory block separately
        core_path = os.path.join(db_path, "core_memory.json")
        with open(core_path, "w", encoding="utf-8") as f:
            json.dump({"human": self.core_memory.human}, f, ensure_ascii=False, indent=2)
        
        print(f"💾 记忆已保存到: {db_path}")
        print(f"   向量索引: {len(self.vector_store.memories)} 条记忆")
        print(f"   核心记忆: {len(self.core_memory.human)} 字符")
    
    def load(self, db_path: str) -> bool:
        """
        Load memory system from disk.
        
        Args:
            db_path: Directory path containing saved memory database
            
        Returns:
            True if loaded successfully, False if files not found
        """
        # Check for vector_store subdirectory (new format) or direct files (old format)
        vector_store_dir = os.path.join(db_path, "vector_store")
        if os.path.isdir(vector_store_dir):
            # New format: files in vector_store/ subdirectory
            index_path = os.path.join(vector_store_dir, "index.faiss")
            payload_path = os.path.join(vector_store_dir, "payload.json")
            load_dir = vector_store_dir
        else:
            # Old format: files directly in db_path
            index_path = os.path.join(db_path, "index.faiss")
            payload_path = os.path.join(db_path, "payload.json")
            load_dir = db_path
        
        if not os.path.exists(index_path) or not os.path.exists(payload_path):
            print(f"⚠️  记忆数据库不存在: {db_path}")
            return False
        
        # Load vector store
        self.vector_store.load(load_dir)
        
        # Load core memory block if exists
        core_path = os.path.join(db_path, "core_memory.json")
        if os.path.exists(core_path):
            with open(core_path, "r", encoding="utf-8") as f:
                core_data = json.load(f)
                self.core_memory.human = core_data.get("human", "")
        
        print(f"📂 记忆已加载: {db_path}")
        print(f"   向量索引: {len(self.vector_store.memories)} 条记忆")
        print(f"   核心记忆: {len(self.core_memory.human)} 字符")
        return True

    def _clean_json_response(self, content: str) -> str:
        """Clean JSON response, remove <think> tags and markdown markers."""
        import re
        if not content:
            return content
        
        clean = content.strip()

        # Remove any think blocks and markdown fences.
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL | re.IGNORECASE).strip()
        clean = re.sub(r"```(?:json)?|```", "", clean, flags=re.IGNORECASE).strip()

        # Robustly extract the first complete JSON object by brace matching.
        start = clean.find("{")
        if start == -1:
            return clean

        depth = 0
        in_string = False
        escape = False
        end = -1
        for i, ch in enumerate(clean[start:], start=start):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end != -1:
            return clean[start:end + 1].strip()
        return clean[start:].strip()

    def _debug_enabled(self) -> bool:
        """Whether debug logs are enabled for memory construction."""
        return os.environ.get("MEMBUILDER_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    def _debug_log(self, tag: str, message: str) -> None:
        """Print guarded debug log."""
        if self._debug_enabled():
            print(f"  [DEBUG:{tag}] {message}", flush=True)

    def _parse_json_with_repair(self, text: str, debug_tag: str = "agent") -> Tuple[Any, str]:
        """Parse JSON with lightweight repairs for near-valid model outputs."""
        import re

        source = (text or "").strip()
        if not source:
            raise ValueError("Empty JSON text after cleaning.")

        candidates = [source]

        # Common repair #1: remove trailing commas before } or ]
        c1 = re.sub(r",(\s*[}\]])", r"\1", source)
        if c1 not in candidates:
            candidates.append(c1)

        # Common repair #2: missing comma between adjacent objects: } { -> }, {
        c2 = re.sub(r"(\})(\s*)(\{)", r"\1,\2\3", c1)
        if c2 not in candidates:
            candidates.append(c2)

        # Common repair #3: missing comma between array close and next object: ] { -> ], {
        c3 = re.sub(r"(\])(\s*)(\{)", r"\1,\2\3", c2)
        if c3 not in candidates:
            candidates.append(c3)

        last_err = None
        for idx, candidate in enumerate(candidates, 1):
            try:
                parsed = json.loads(candidate)
                if idx > 1:
                    self._debug_log(debug_tag, f"json_repair_applied=variant_{idx}")
                return parsed, candidate
            except Exception as e:
                last_err = e
                # Heuristic repair using json error position for missing comma cases.
                if isinstance(e, json.JSONDecodeError) and "Expecting ',' delimiter" in str(e):
                    pos = e.pos
                    left = pos - 1
                    while left >= 0 and candidate[left].isspace():
                        left -= 1
                    right = pos
                    while right < len(candidate) and candidate[right].isspace():
                        right += 1
                    left_ch = candidate[left] if left >= 0 else ""
                    right_ch = candidate[right] if right < len(candidate) else ""
                    self._debug_log(
                        debug_tag,
                        f"json_error_pos={pos}, left='{left_ch}', right='{right_ch}', line={e.lineno}, col={e.colno}"
                    )
                    if left_ch in {'"', '}', ']'} and right_ch in {'"', '{', '['}:
                        candidate2 = candidate[:right] + "," + candidate[right:]
                        try:
                            parsed = json.loads(candidate2)
                            self._debug_log(debug_tag, "json_repair_applied=insert_comma_at_error_pos")
                            return parsed, candidate2
                        except Exception as e2:
                            last_err = e2

        raise last_err

    def _core_secondary_local_repair(self, text: str) -> Optional[str]:
        """Best-effort local repair for malformed Core agent JSON."""
        import re

        source = (text or "").strip()
        if not source:
            return None

        start = source.find("{")
        end = source.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        blob = source[start:end + 1]

        op_match = re.search(r'"operation"\s*:\s*"([^"]+)"', blob, flags=re.IGNORECASE)
        if not op_match:
            return None
        operation = op_match.group(1).strip().upper()
        if operation not in {"APPEND", "REPLACE", "REWRITE", "SKIP"}:
            return None

        known_keys = ["operation", "content", "old_text", "new_text", "reason"]

        def extract_string_field(key: str) -> Optional[str]:
            key_match = re.search(rf'"{re.escape(key)}"\s*:', blob)
            if not key_match:
                return None
            colon_pos = blob.find(":", key_match.start())
            if colon_pos == -1:
                return None
            quote_start = blob.find('"', colon_pos + 1)
            if quote_start == -1:
                return None

            next_positions = []
            scan_start = quote_start + 1
            for next_key in known_keys:
                if next_key == key:
                    continue
                m = re.search(rf',\s*"{re.escape(next_key)}"\s*:', blob[scan_start:])
                if m:
                    next_positions.append(scan_start + m.start())

            boundary = min(next_positions) if next_positions else blob.rfind("}")
            if boundary == -1 or boundary <= quote_start:
                boundary = len(blob)

            quote_end = blob.rfind('"', quote_start + 1, boundary)
            if quote_end == -1 or quote_end <= quote_start:
                return None
            return blob[quote_start + 1:quote_end]

        payload: Dict[str, Any] = {"operation": operation}
        if operation in {"APPEND", "REWRITE"}:
            content = extract_string_field("content")
            if content is None:
                return None
            payload["content"] = content
        elif operation == "REPLACE":
            old_text = extract_string_field("old_text")
            new_text = extract_string_field("new_text")
            if old_text is None or new_text is None:
                return None
            payload["old_text"] = old_text
            payload["new_text"] = new_text
        elif operation == "SKIP":
            reason = extract_string_field("reason")
            if reason is not None:
                payload["reason"] = reason

        return json.dumps(payload, ensure_ascii=False)

    def _core_secondary_regenerate_json(self, broken_json: str, debug_tag: str = "core_agent") -> Optional[str]:
        """One-shot Core-specific JSON regeneration from malformed output."""
        prompt = f"""You are a strict JSON repair tool.
Given a malformed JSON produced by a memory agent, return ONE valid JSON object only.

Allowed schemas:
1) APPEND/REWRITE: {{"operation":"APPEND|REWRITE","content":"..."}}
2) REPLACE: {{"operation":"REPLACE","old_text":"...","new_text":"..."}}
3) SKIP: {{"operation":"SKIP","reason":"..."}}

Rules:
- Output JSON only, no markdown, no explanation.
- Escape all internal double quotes inside string values.
- Keep original meaning; do not invent new facts.

Malformed JSON:
{broken_json[:12000]}
"""
        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            clean = self._clean_json_response(response)
            parsed, valid_json = self._parse_json_with_repair(clean, debug_tag=debug_tag)
            if not isinstance(parsed, dict):
                return None
            operation = str(parsed.get("operation", "")).upper().strip()
            if operation not in {"APPEND", "REPLACE", "REWRITE", "SKIP"}:
                return None
            self._debug_log(debug_tag, "core_secondary_repair_applied=regenerate_json")
            return valid_json
        except Exception as e:
            self._debug_log(debug_tag, f"core_secondary_regen_failed={type(e).__name__}: {str(e)[:120]}")
            return None

    def _retry_with_json_validation(self, prompt: str, max_retries: int = 10, debug_tag: str = "agent") -> str:
        """Unified retry mechanism with JSON validation."""
        import time
        prompt_preview = (prompt or "")[:200].replace("\n", " ")
        self._debug_log(debug_tag, f"prompt_len={len(prompt or '')}, prompt_preview={prompt_preview}")
        
        for attempt in range(max_retries):
            try:
                temperature = 0.0 if attempt == 0 else 0.1
                response = self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
                
                if not response:
                    raise ValueError("Empty response from LLM")

                raw_preview = response[:220].replace("\n", " ")
                self._debug_log(debug_tag, f"attempt={attempt + 1}, raw_len={len(response)}, raw_preview={raw_preview}")
                
                # Clean the response
                clean_response = self._clean_json_response(response)
                clean_preview = clean_response[:220].replace("\n", " ")
                self._debug_log(debug_tag, f"attempt={attempt + 1}, clean_len={len(clean_response)}, clean_preview={clean_preview}")
                
                # Validate JSON (with Core-specific secondary fallback).
                try:
                    _, valid_json = self._parse_json_with_repair(clean_response, debug_tag=debug_tag)
                    return valid_json
                except Exception as parse_err:
                    if debug_tag == "core_agent" and isinstance(parse_err, json.JSONDecodeError):
                        repaired = self._core_secondary_local_repair(clean_response)
                        if repaired:
                            _, valid_json = self._parse_json_with_repair(repaired, debug_tag=debug_tag)
                            self._debug_log(debug_tag, "core_secondary_repair_applied=local_salvage")
                            return valid_json

                        regenerated = self._core_secondary_regenerate_json(clean_response, debug_tag=debug_tag)
                        if regenerated:
                            return regenerated
                    raise parse_err
                
            except Exception as e:
                self._debug_log(debug_tag, f"attempt={attempt + 1}, error={type(e).__name__}: {str(e)[:300]}")
                if attempt < max_retries - 1:
                    wait_time = min(0.5 * (2 ** attempt), 16.0)  # 最高16s
                    print(f"  ⚠️ LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:80]}, {wait_time:.1f}s后重试...", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ LLM调用失败，已重试 {max_retries} 次: {str(e)[:100]}", flush=True)
                    raise

    def save_state(self, state_dir: str) -> None:
        os.makedirs(state_dir, exist_ok=True)

        with open(os.path.join(state_dir, "core_memory.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"human": self.core_memory.human, "usage": self.core_memory.get_usage()},
                f,
                ensure_ascii=False,
                indent=2,
            )

        self.vector_store.save(os.path.join(state_dir, "vector_store"))

    def load_state(self, state_dir: str) -> None:
        core_path = os.path.join(state_dir, "core_memory.json")
        if not os.path.exists(core_path):
            raise FileNotFoundError(f"Missing core_memory.json in: {state_dir}")

        with open(core_path, "r", encoding="utf-8") as f:
            core_state = json.load(f)

        self.core_memory.human = core_state.get("human", "")
        self.vector_store.load(os.path.join(state_dir, "vector_store"))
    
    def add(self, messages: List[Dict], user_id: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Process messages with all four agents in parallel.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            user_id: User identifier
            metadata: Optional metadata (e.g., timestamp)
        
        Returns:
            Dict with operation statistics
        """
        operation_stats = {
            'core': {},
            'episodic': {'ADD': 0, 'UPDATE': 0, 'MERGE': 0},
            'semantic': {'ADD': 0, 'UPDATE': 0, 'SKIP': 0},
            'procedural': {'ADD': 0, 'UPDATE': 0}
        }
        
        searchable_memories = []
        
        # Process all 4 agents in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._core_agent, messages, user_id): "core",
                executor.submit(self._episodic_agent, messages, user_id, metadata): "episodic",
                executor.submit(self._semantic_agent, messages, user_id): "semantic",
                executor.submit(self._procedural_agent, messages, user_id): "procedural"
            }
            try:
                for future in as_completed(futures, timeout=300):
                    agent_type = futures[future]
                    try:
                        if agent_type == "core":
                            op_type, _ = future.result()
                            operation_stats['core'] = {op_type: 1} if op_type else {}
                        else:
                            memories, stats = future.result()
                            searchable_memories.extend(memories)
                            for op, count in stats.items():
                                if op in operation_stats[agent_type]:
                                    operation_stats[agent_type][op] += count
                    except Exception as e:
                        print(f"  Error in {agent_type} agent: {str(e)[:100]}")
            except FuturesTimeoutError:
                unfinished = [name for f, name in futures.items() if not f.done()]
                self._debug_log("agents", f"timeout=300s, unfinished={unfinished}")
                raise
        
        # Store searchable memories in vector database
        results = []
        if searchable_memories:
            metadatas = [{"user_id": user_id, "type": "memory"} for _ in searchable_memories]
            self.vector_store.add(searchable_memories, metadatas)
            # Build results list matching old code format
            results = [{"memory": mem, "user_id": user_id} for mem in searchable_memories]
        
        return {
            "results": results,  # Match runner.py expectation
            "memories_added": len(searchable_memories),
            "operation_stats": operation_stats
        }
    
    def search(self, query: str, user_id: str, limit: int = None, top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Search for relevant memories.
        
        Results are filtered by user_id metadata to prevent cross-user leakage
        when a single MemorySystem instance is shared across users.
        
        Args:
            query: Search query
            user_id: User identifier for metadata filtering
            limit: Number of results (alias for top_k, for compatibility)
            top_k: Number of results to return
        
        Returns:
            Dict with 'results' key containing list of memory dicts
        """
        # Support both 'limit' and 'top_k' parameter names for compatibility
        k = limit if limit is not None else top_k
        # Fetch extra candidates so we still have enough after user_id filtering
        raw_results = self.vector_store.search(query, k * 2)
        filtered = [
            r for r in raw_results
            if not r.get("metadata", {}).get("user_id") or r["metadata"]["user_id"] == user_id
        ]
        return {'results': filtered[:k]}
    
    def generate_answer(self, question: str, memories: List[Dict] = None, 
                       user_id: str = None, current_time: Optional[str] = None,
                       additional_context: Optional[str] = None) -> str:
        """
        Generate answer with Core Memory always included (MIRIX-style).
        
        Args:
            question: Question to answer
            memories: Pre-retrieved memories list (if None, will search internally)
            user_id: User identifier
            current_time: Current time in dataset timeline (for relative time questions)
            additional_context: Additional context (e.g., character profile)
        
        Returns:
            Generated answer string
        """
        # If memories not provided, retrieve them
        if memories is None:
            search_result = self.search(question, user_id, top_k=QA_ANSWERING_TOP_K)
            memories = search_result.get('results', [])
        
        # Build context - matching old code format exactly
        context = ""
        
        # Add additional context first (e.g., character profile)
        if additional_context:
            context += f"{additional_context}\n\n"
        
        # Add Core Memory
        context += f"""Core Memory (User Profile):
{self.core_memory.human if self.core_memory.human else "[No core memory available]"}

Retrieved Memories:
"""
        # Add retrieved memories
        for i, mem in enumerate(memories, 1):
            if isinstance(mem, dict):
                memory_text = mem.get('memory', mem.get('text', str(mem)))
            else:
                memory_text = str(mem)
            context += f"{i}. {memory_text}\n"
        
        # Add current time context if available
        time_context = ""
        if current_time:
            time_context = f"\n\n**The current date/time is {current_time}. Use this as the reference point when answering questions about relative time (e.g., 'last year', 'yesterday', 'recently').**"
        
        # Generate answer with MIRIX-style flexible understanding
        prompt = f"""{context}{time_context}

Question: {question}

Instructions:
1. Carefully analyze the retrieved memories to find relevant information
2. Consider synonyms and related concepts (e.g., "support group", "activist group" may refer to similar things)
3. If memories mention specific dates/times, use those to answer time-related questions
4. If memories contain contradictory information, prioritize the most recent memory
5. Focus on the content of the memories, not just exact word matches

**For factual questions (What/When/Where/Who):**
- Answer based on direct information in the memories
- If the specific fact is not mentioned, respond: "Not answerable"

**For inference/reasoning questions (Would/Could/Likely):**
- You CAN make reasonable inferences based on related information in the memories

**When to say "Not answerable":**
- If the question asks about a specific person but the memories are about a DIFFERENT person
- If the question asks about an event/action that is NOT mentioned in ANY memories
- If you find information about a similar but DIFFERENT event

Provide a concise, direct answer based on the available information, or state "Not answerable" if the specific information is not present."""
        
        try:
            return self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)[:100]}")
            return f"Error: {str(e)[:100]}"
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for prompt consumption."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"[REAL_USER]: {content}")
            elif role == "assistant":
                formatted.append(f"[AI_ASSISTANT]: {content}")
            elif role == "system":
                formatted.append(f"[SYSTEM_INSTRUCTION]: {content}")
            else:
                formatted.append(f"[OTHER_ROLE:{role}]: {content}")
        return "\n".join(formatted)
    
    def _get_existing_memories(self, memory_type: str, user_id: str, 
                               messages: Optional[List[Dict]] = None) -> List[str]:
        """Get existing memories of a specific type for context."""
        prefix = MEMORY_PREFIXES.get(memory_type, "")
        
        # Build search query from messages
        if messages:
            query_text = " ".join([m.get("content", "") for m in messages])
            query = f"{prefix} {query_text}"
        else:
            query = prefix
        
        # Use MEMORY_CONSTRUCTION_TOP_K (20) as per paper Appendix
        search_result = self.search(query, user_id, top_k=MEMORY_CONSTRUCTION_TOP_K)
        results = search_result.get('results', []) if isinstance(search_result, dict) else []
        
        # Filter by type and limit to top-20
        return [r["memory"] for r in results if isinstance(r, dict) and r.get("memory", "").startswith(prefix)][:MEMORY_CONSTRUCTION_TOP_K]
    
    def _core_agent(self, messages: List[Dict], user_id: str) -> Tuple[str, str]:
        """Core Memory Agent: Manage persistent user profile."""
        formatted_messages = self._format_messages(messages)
        core_snapshot = self.core_memory.human
        
        template = Template(self.prompts["core"])
        prompt = template.render(
            current_core_memory=self.core_memory.human or "[Empty]",
            core_usage=self.core_memory.get_usage(),
            messages=formatted_messages
        )
        
        try:
            # Enforce strict JSON output and retry/repair logic for core agent as well.
            response = self._retry_with_json_validation(prompt, debug_tag="core_agent")
            result = json.loads(response)
            operation = str(result.get("operation", "APPEND")).upper().strip()
            if operation not in {"APPEND", "REPLACE", "REWRITE", "SKIP"}:
                raise ValueError(f"Invalid core operation: {operation}")
            
            # Apply operation
            if operation == "APPEND":
                content = result.get("content", "")
                # Ensure content is a string
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False) if content else ""
                if content:
                    self.core_memory.human = (
                        self.core_memory.human + f"\n{content}" 
                        if self.core_memory.human else content
                    )
            elif operation == "REPLACE":
                old_text = result.get("old_text", "")
                new_text = result.get("new_text", "")
                if old_text and new_text:
                    self.core_memory.human = self.core_memory.human.replace(old_text, new_text)
            elif operation == "REWRITE":
                content = result.get("content", "")
                # Ensure content is a string
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False) if content else ""
                if content:
                    self.core_memory.human = content
            
            # Compress if needed
            if len(self.core_memory.human) > CORE_MEMORY_CHAR_LIMIT:
                self._compress_core_memory()
            
            return (operation, self.core_memory.human)
            
        except json.JSONDecodeError as e:
            self.core_memory.human = core_snapshot
            print(f"  Core Agent JSON error: {str(e)[:100]}")
            return ("ERROR", "")
        except Exception as e:
            self.core_memory.human = core_snapshot
            print(f"  Core Agent error: {str(e)[:100]}")
            return ("ERROR", "")
    
    def _compress_core_memory(self):
        """Compress core memory if it exceeds the limit."""
        prompt = CORE_MEMORY_COMPRESS_PROMPT.format(
            length=len(self.core_memory.human),
            limit=CORE_MEMORY_CHAR_LIMIT,
            content=self.core_memory.human
        )
        
        response = self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response)
        self.core_memory.human = result.get("content", "")[:CORE_MEMORY_CHAR_LIMIT]
    
    def _episodic_agent(self, messages: List[Dict], user_id: str, 
                        metadata: Optional[Dict] = None) -> Tuple[List[str], Dict]:
        """Episodic Memory Agent: Extract time-ordered events."""
        existing = self._get_existing_memories("episodic", user_id, messages)
        formatted_messages = self._format_messages(messages)
        timestamp = metadata.get("timestamp", "Not provided") if metadata else "Not provided"
        
        template = Template(self.prompts["episodic"])
        prompt = template.render(
            existing_episodic="\n".join(existing) or "[No existing memories]",
            messages=formatted_messages,
            conversation_timestamp=timestamp
        )
        
        return self._process_memory_agent(prompt, "episodic")
    
    def _semantic_agent(self, messages: List[Dict], user_id: str) -> Tuple[List[str], Dict]:
        """Semantic Memory Agent: Extract conceptual knowledge."""
        existing = self._get_existing_memories("semantic", user_id, messages)
        formatted_messages = self._format_messages(messages)
        
        template = Template(self.prompts["semantic"])
        prompt = template.render(
            existing_semantic="\n".join(existing) or "[No existing memories]",
            messages=formatted_messages
        )
        
        return self._process_memory_agent(prompt, "semantic")
    
    def _procedural_agent(self, messages: List[Dict], user_id: str) -> Tuple[List[str], Dict]:
        """Procedural Memory Agent: Extract step-by-step processes."""
        existing = self._get_existing_memories("procedural", user_id, messages)
        formatted_messages = self._format_messages(messages)
        
        template = Template(self.prompts["procedural"])
        prompt = template.render(
            existing_procedural="\n".join(existing) or "[No existing memories]",
            messages=formatted_messages
        )
        
        return self._process_memory_agent(prompt, "procedural")
    
    def _process_memory_agent(self, prompt: str, memory_type: str) -> Tuple[List[str], Dict]:
        """Process agent response and extract memories with retry and JSON validation."""
        prefix = MEMORY_PREFIXES[memory_type]
        stats = {'ADD': 0, 'UPDATE': 0, 'MERGE': 0, 'SKIP': 0}
        memories = []
        
        try:
            # Use unified retry mechanism with JSON validation
            response = self._retry_with_json_validation(prompt, debug_tag=f"{memory_type}_agent")
            
            # Debug: check response type
            if not response:
                print(f"  ⚠️ {memory_type.capitalize()} Agent: 空响应")
                return [], stats
            
            result = json.loads(response)

            if isinstance(result, dict):
                operations = result.get("operations", [])
            elif isinstance(result, list):
                operations = result
            else:
                print(f"  ⚠️ {memory_type.capitalize()} Agent: 无法识别的响应类型 (type={type(result).__name__})，跳过")
                return [], stats

            if not isinstance(operations, list):
                print(f"  ⚠️ {memory_type.capitalize()} Agent: operations 不是列表，跳过")
                return [], stats
            
            for op in operations:
                # Support both dict format and compact list format:
                #   {"action":"ADD","memory":"..."}
                #   ["ADD", "memory text"]
                #   ["UPDATE", "old memory", "new memory"] (optional old_memory)
                if isinstance(op, dict):
                    action = str(op.get("action", "")).upper().strip()
                    payload = op
                elif isinstance(op, (list, tuple)) and len(op) >= 1:
                    action = str(op[0]).upper().strip()
                    payload = {}
                    if action == "ADD":
                        payload["memory"] = op[1] if len(op) >= 2 else ""
                    elif action in ["UPDATE", "MERGE"]:
                        if len(op) >= 3:
                            payload["old_memory"] = op[1]
                            payload["new_memory"] = op[2]
                        elif len(op) >= 2:
                            payload["new_memory"] = op[1]
                    elif action == "SKIP":
                        pass
                else:
                    continue

                if action in stats:
                    stats[action] += 1
                
                if action == "ADD":
                    memory = payload.get("memory", "")
                    # Ensure memory is string (some models return dict)
                    if isinstance(memory, dict):
                        memory = json.dumps(memory, ensure_ascii=False)
                    if memory:
                        if not memory.startswith(prefix):
                            memory = f"{prefix} {memory}"
                        memories.append(memory)
                        
                elif action in ["UPDATE", "MERGE"]:
                    new_memory = payload.get("new_memory", "")
                    # Ensure new_memory is string
                    if isinstance(new_memory, dict):
                        new_memory = json.dumps(new_memory, ensure_ascii=False)
                    if new_memory:
                        if not new_memory.startswith(prefix):
                            new_memory = f"{prefix} {new_memory}"
                        memories.append(new_memory)
            
            print(f"  📝 {memory_type.capitalize()} Agent: {len(operations)} 操作 (ADD: {stats['ADD']}, UPDATE: {stats['UPDATE']}, SKIP: {stats['SKIP']})")
            return memories, stats
            
        except Exception as e:
            print(f"  ❌ {memory_type.capitalize()} Agent error: {str(e)[:100]}")
            return [], stats
