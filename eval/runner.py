#!/usr/bin/env python3
"""
Unified evaluation runner for MemBuilder.

Usage:
    # LOCOMO - 单个对话
    python -m eval.runner --dataset locomo --conv-id conv-26 --questions 10
    
    # LOCOMO - 全部对话
    python -m eval.runner --dataset locomo --mode full
    
    # 使用 vLLM
    python -m eval.runner --dataset locomo --provider vllm --vllm-url http://localhost:8000/v1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_system import MemorySystem

# Internal client provides MetaAI support and separate embedding endpoints.
# Falls back to the open-source client if internal module is not available.
try:
    from llm_client_internal import OpenAIClient, create_llm_client
    _USING_INTERNAL_CLIENT = True
except ImportError:
    from llm_client import OpenAIClient, create_llm_client
    _USING_INTERNAL_CLIENT = False

from config import (
    ANSWER_MODEL, JUDGE_MODEL, EMBEDDING_MODEL,
    QA_ANSWERING_TOP_K, OPENAI_API_KEY, OPENAI_BASE_URL,
    OPENAI_EMBEDDINGS_BASE_URL, OPENAI_EMBEDDINGS_API_KEY
)

from eval.llm_judge import evaluate_answer, LLMJudge
from eval.metrics import compute_accuracy, print_accuracy_stats
from eval.datasets import load_dataset, get_current_time


def _make_embedding_func():
    """Create a separate embedding function if a dedicated embedding endpoint is configured.
    
    Only activates when OPENAI_EMBEDDINGS_BASE_URL is explicitly set in env/config.
    Otherwise returns None — the llm_client handles embedding internally
    (internal client auto-falls back to yunwu.ai for vLLM; open-source uses same endpoint).
    
    Returns:
        Embedding function, or None.
    """
    if not OPENAI_EMBEDDINGS_BASE_URL:
        return None
    
    from openai import OpenAI
    emb_client = OpenAI(
        api_key=OPENAI_EMBEDDINGS_API_KEY or "EMPTY",
        base_url=OPENAI_EMBEDDINGS_BASE_URL
    )
    
    def _embedding_func(texts):
        response = emb_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return [item.embedding for item in response.data]
    
    return _embedding_func


def _create_memory_system(llm_client):
    """Create MemorySystem with proper embedding configuration.
    
    If a separate embedding endpoint is configured (via config.py),
    it will be used instead of the llm_client's default embeddings.
    This is essential when the chat provider (e.g., vLLM) cannot serve embeddings.
    """
    embedding_func = _make_embedding_func()
    return MemorySystem(llm_client=llm_client, embedding_func=embedding_func)


def process_longmemeval_sample(sample_args):
    """Process a single LongMemEval sample (for parallel processing)."""
    sample, idx, total, provider, model_name, judge_model_name, judge_provider, kwargs, judge_kwargs, max_sessions, top_k, mode, verbose = sample_args
    
    sample_id = sample['question_id']
    user_id = f'test_{sample_id}'
    
    # Create new LLM client for this process
    sample_llm_client = create_llm_client(provider=provider, model=model_name, **kwargs)
    sample_judge_client = create_llm_client(provider=judge_provider, model=judge_model_name, **judge_kwargs)
    sample_judge = LLMJudge(sample_judge_client)
    
    # Limit sessions if requested
    if max_sessions:
        sample = sample.copy()
        sample['haystack_sessions'] = sample['haystack_sessions'][:max_sessions]
        if 'haystack_session_datetimes' in sample:
            sample['haystack_session_datetimes'] = sample['haystack_session_datetimes'][:max_sessions]
    
    # Initialize MemorySystem
    memory = _create_memory_system(sample_llm_client)
    
    result = {
        'question_id': sample_id,
        'question': sample['question'],
        'expected_answer': sample['answer'],
        'question_type': sample.get('question_type', 'unknown'),
        'index': idx
    }
    
    try:
        # Build memories
        if mode in ['full', 'build']:
            build_result = build_memories(
                memory, sample, user_id,
                max_sessions=max_sessions,
                verbose=False  # Quiet for parallel
            )
            result['memories_built'] = build_result['total_memories']
        
        # Answer question
        if mode in ['full', 'answer']:
            current_time = get_current_time(sample)
            
            search_start = time.time()
            search_result = memory.search(sample['question'], user_id=user_id, limit=top_k)
            search_time = time.time() - search_start
            memories = search_result.get('results', []) if isinstance(search_result, dict) else []
            
            gen_start = time.time()
            answer = memory.generate_answer(sample['question'], memories, user_id, current_time=current_time)
            gen_time = time.time() - gen_start
            
            judge_start = time.time()
            llm_score = sample_judge.evaluate(sample['question'], sample['answer'], answer)
            judge_time = time.time() - judge_start
            
            is_correct = (llm_score == 1)
            result.update({
                'generated_answer': answer,
                'is_correct': is_correct,
                'llm_score': llm_score,
                'memories_found': len(memories),
                'search_time': search_time,
                'gen_time': gen_time,
                'judge_time': judge_time,
            })
            
            status = '✅' if is_correct else '❌'
            print(f"[{idx}/{total}] {status} {sample_id}: {result.get('memories_built', 0)}条记忆, {search_time+gen_time:.1f}s")
        
    except Exception as e:
        result['error'] = str(e)
        result['llm_score'] = 0
        result['is_correct'] = False
        print(f"[{idx}/{total}] ❌ ERROR {sample_id}: {str(e)[:50]}")
    
    return result


def get_db_path(dataset_name: str, model_name: str, vector_store: str = "faiss", 
                conversation_id: str = None, sample_id: str = None, 
                character_id: str = None) -> str:
    """
    生成统一的数据库路径（参考老代码 test_memory_system.py）
    
    结构: {vector_store}_data/{dataset}/{model}/{id}/
    例如: faiss_data/locomo/gpt-4o-mini/conv-26/
    
    Args:
        dataset_name: 数据集名称 (locomo, longmemeval, perltqa)
        model_name: 模型名称 (gpt-4o-mini, claude-sonnet-4-5-20250929等)
        vector_store: 向量数据库类型 (faiss, qdrant)
        conversation_id: 对话ID (用于locomo，如conv-26)
        sample_id: 样本ID (用于longmemeval)
        character_id: 人物ID (用于perltqa)
    
    Returns:
        数据库路径字符串
    """
    # 清理模型名称中的特殊字符
    clean_model = model_name.replace('/', '_').replace('.', '_')
    
    # 基础路径
    base_dir = f"{vector_store}_data"
    
    # 构建路径: base_dir/dataset/model/id
    if conversation_id:
        # LOCOMO: faiss_data/locomo/gpt-4o-mini/conv-26
        db_path = f"./{base_dir}/{dataset_name}/{clean_model}/{conversation_id}"
    elif sample_id:
        # LongMemEval: faiss_data/longmemeval/gpt-4o-mini/sample_xxx
        db_path = f"./{base_dir}/{dataset_name}/{clean_model}/{sample_id}"
    elif character_id:
        # PerLTQA: faiss_data/perltqa/gpt-4o-mini/char_xxx
        db_path = f"./{base_dir}/{dataset_name}/{clean_model}/{character_id}"
    else:
        # 默认（单对话测试）
        db_path = f"./{base_dir}/{dataset_name}/{clean_model}/default"
    
    return db_path


def build_memories(
    memory: MemorySystem, 
    sample: Dict, 
    user_id: str,
    max_sessions: int = None, 
    verbose: bool = True
) -> Dict[str, Any]:
    """Build memories from conversation sessions."""
    sessions = sample['haystack_sessions']
    timestamps = sample.get('haystack_session_datetimes', [])
    
    if max_sessions:
        sessions = sessions[:max_sessions]
        timestamps = timestamps[:max_sessions]
    
    print(f"\n📦 构建记忆 ({len(sessions)} 个会话)")
    print("=" * 60)
    
    build_start = time.time()
    total_memories = 0
    session_results = []
    
    for i, session in enumerate(sessions):
        if verbose:
            print(f"处理会话 {i+1}/{len(sessions)}...", end=" ", flush=True)
        
        timestamp = timestamps[i] if i < len(timestamps) else None
        metadata = {'timestamp': timestamp} if timestamp else None
        
        try:
            session_start = time.time()
            messages = session.get('messages', session) if isinstance(session, dict) else session
            result = memory.add(messages, user_id=user_id, metadata=metadata)
            session_time = time.time() - session_start
            
            if result and 'results' in result:
                count = len(result['results'])
                total_memories += count
                session_results.append({
                    'session_id': i,
                    'memories_count': count,
                    'timestamp': timestamp,
                    'session_time': session_time
                })
                if verbose:
                    print(f"✅ {count} 条记忆 ({session_time:.1f}s)")
            else:
                if verbose:
                    print(f"⚠️ 无结果")
        except Exception as e:
            print(f"❌ 错误: {str(e)[:50]}")
            session_results.append({'session_id': i, 'error': str(e)})
    
    build_time = time.time() - build_start
    
    print(f"\n✅ 记忆构建完成:")
    print(f"   总记忆数: {total_memories}")
    print(f"   处理会话: {len(sessions)}")
    print(f"   总耗时: {build_time:.1f}秒")
    
    return {
        'total_memories': total_memories,
        'sessions_processed': len(sessions),
        'build_time': build_time,
        'session_results': session_results,
        'user_id': user_id
    }


def _answer_single_question(
    memory: MemorySystem,
    q: Dict,
    user_id: str,
    judge: LLMJudge,
    index: int,
    total: int,
    top_k: int = 10,
    current_time: str = None
) -> Dict:
    """Answer a single question (for parallel processing)."""
    result = {
        'question_id': q['question_id'],
        'question': q['question'],
        'expected_answer': q['answer'],
        'question_type': q.get('question_type', 'unknown'),
        'index': index
    }
    
    try:
        # Search memories
        search_start = time.time()
        search_result = memory.search(q['question'], user_id=user_id, limit=top_k)
        search_time = time.time() - search_start
        
        memories = search_result.get('results', []) if isinstance(search_result, dict) else []
        
        # Generate answer
        gen_start = time.time()
        answer = memory.generate_answer(q['question'], memories, user_id, current_time=current_time)
        gen_time = time.time() - gen_start
        
        # Judge
        judge_start = time.time()
        llm_score = judge.evaluate(q['question'], q['answer'], answer)
        judge_time = time.time() - judge_start
        
        is_correct = (llm_score == 1)
        
        result.update({
            'generated_answer': answer,
            'is_correct': is_correct,
            'llm_score': llm_score,
            'memories_found': len(memories),
            'search_time': search_time,
            'gen_time': gen_time,
            'judge_time': judge_time,
        })
        
        status = '✅' if is_correct else '❌'
        print(f"[{index}/{total}] {status} {q['question_id']}: {search_time+gen_time:.1f}s")
        
    except Exception as e:
        result['error'] = str(e)
        result['llm_score'] = 0
        result['is_correct'] = False
        print(f"[{index}/{total}] ❌ ERROR {q['question_id']}: {str(e)[:50]}")
    
    return result


def answer_questions(
    memory: MemorySystem,
    questions: List[Dict],
    user_id: str,
    judge: LLMJudge,
    limit: int = None,
    top_k: int = 10,
    current_time: str = None,
    verbose: bool = True,
    concurrency: int = 1
) -> List[Dict]:
    """Answer questions using the memory system with optional concurrency.
    
    Args:
        concurrency: Number of concurrent workers (1 = sequential)
    """
    if limit:
        questions = questions[:limit]
    
    total_questions = len(questions)
    
    if concurrency > 1 and total_questions > 1:
        # Parallel processing
        print(f"\n📝 回答问题 ({total_questions} 个问题, {concurrency} 并发)")
        print("=" * 60)
        
        total_start = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _answer_single_question, memory, q, user_id, judge,
                    i, total_questions, top_k, current_time
                ): i for i, q in enumerate(questions, 1)
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    print(f"[{idx}/{total_questions}] ❌ 并发错误: {str(e)[:50]}")
        
        # Sort by original index
        results.sort(key=lambda x: x.get('index', 0))
        
        total_time = time.time() - total_start
        print_accuracy_stats(results, "📊 答题总结")
        print(f"   总耗时: {total_time:.1f}秒")
        
        return results
    
    # Sequential processing
    print(f"\n📝 回答问题 ({total_questions} 个问题)")
    print("=" * 60)
    
    results = []
    total_start = time.time()
    
    for i, q in enumerate(questions, 1):
        if verbose:
            print(f"\n[{i}/{total_questions}] {q['question_id']}")
            print(f"  问题: {q['question']}")
            print(f"  标准答案: {q['answer']}")
            print(f"  类型: {q.get('question_type', 'unknown')}")
        
        result = {
            'question_id': q['question_id'],
            'question': q['question'],
            'expected_answer': q['answer'],
            'question_type': q.get('question_type', 'unknown')
        }
        
        try:
            # Search memories first (matching old code pattern)
            search_start = time.time()
            search_result = memory.search(q['question'], user_id=user_id, limit=top_k)
            search_time = time.time() - search_start
            
            # Extract memories list from result
            memories = search_result.get('results', []) if isinstance(search_result, dict) else []
            
            if verbose:
                print(f"  检索到 {len(memories)} 条相关记忆 ({search_time:.1f}s)")
            
            # Generate answer - pass memories to match old code signature
            gen_start = time.time()
            answer = memory.generate_answer(q['question'], memories, user_id, current_time=current_time)
            gen_time = time.time() - gen_start
            
            if verbose:
                print(f"  💡 生成答案: {answer[:200]}..." if len(answer) > 200 else f"  💡 生成答案: {answer}")
            
            # Judge
            judge_start = time.time()
            llm_score = judge.evaluate(q['question'], q['answer'], answer)
            judge_time = time.time() - judge_start
            
            is_correct = (llm_score == 1)
            
            result.update({
                'generated_answer': answer,
                'is_correct': is_correct,
                'llm_score': llm_score,
                'memories_found': len(memories),
                'search_time': search_time,
                'gen_time': gen_time,
                'judge_time': judge_time,
            })
            
            status = '✅' if is_correct else '❌'
            if verbose:
                print(f"  ⏱️  耗时: 搜索{search_time:.1f}秒 + 生成{gen_time:.1f}秒")
                print(f"  🎯 LLM Judge: {status} {'CORRECT' if is_correct else 'WRONG'} ({judge_time:.1f}秒)")
            else:
                print(f"[{i}/{total_questions}] {status} {q['question_id']}: {search_time+gen_time:.1f}s")
            
        except Exception as e:
            result['error'] = str(e)
            result['llm_score'] = 0
            result['is_correct'] = False
            print(f"[{i}/{total_questions}] ❌ ERROR {q['question_id']}: {str(e)[:50]}")
        
        results.append(result)
    
    total_time = time.time() - total_start
    
    print_accuracy_stats(results, "📊 答题总结")
    print(f"   总耗时: {total_time:.1f}秒")
    
    return results


def run_evaluation(args) -> int:
    """Main evaluation loop."""
    
    # ========== API 配置 ==========
    provider = args.provider
    model = args.model or ANSWER_MODEL
    judge_model = args.judge_model or JUDGE_MODEL
    
    # Resolve API config from args / env / config.py (used for openai provider and judge)
    _orig_api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    _orig_base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
    
    # Provider-specific setup
    if provider == 'vllm':
        pass  # vLLM config passed via explicit kwargs, NOT env vars
    elif provider == 'metaai':
        pass  # MetaAI uses its own authentication (signature-based), no OpenAI env vars needed
    else:  # openai
        if not _orig_api_key:
            print("Error: OPENAI_API_KEY not configured.")
            print("Set via: --api-key, OPENAI_API_KEY env var, or config.py")
            return 1
        
        os.environ["OPENAI_API_KEY"] = _orig_api_key
        if _orig_base_url:
            os.environ["OPENAI_BASE_URL"] = _orig_base_url
    
    print("=" * 60)
    print(f"🧪 MemBuilder Evaluation")
    print(f"   数据集: {args.dataset}")
    print(f"   模式: {args.mode}")
    print(f"   模型: {model}")
    print(f"   Judge: {judge_model}")
    print(f"   Provider: {provider}")
    print(f"   Client: {'internal' if _USING_INTERNAL_CLIENT else 'open-source'}")
    if OPENAI_EMBEDDINGS_BASE_URL:
        print(f"   Embedding: {EMBEDDING_MODEL} @ {OPENAI_EMBEDDINGS_BASE_URL}")
    elif _USING_INTERNAL_CLIENT:
        print(f"   Embedding: {EMBEDDING_MODEL} (internal auto-fallback)")
    else:
        if provider == 'vllm':
            print(f"   Embedding: {EMBEDDING_MODEL} ⚠️  WARNING: vLLM may not support embeddings!")
            print(f"              Set OPENAI_EMBEDDINGS_BASE_URL env var for a separate embedding endpoint.")
        else:
            print(f"   Embedding: {EMBEDDING_MODEL} (same as chat endpoint)")
    print("=" * 60)
    
    # ========== 处理 --split 参数 ==========
    subset_file = getattr(args, 'subset_file', None)
    split_sample_ids = None
    if args.split and args.dataset == 'longmemeval':
        splits_file = Path(__file__).parent.parent / "data" / "longmemeval" / "splits" / "longmemeval_splits.json"
        if splits_file.exists():
            with open(splits_file, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            if args.split in splits_data:
                split_sample_ids = splits_data[args.split]
                print(f"   Using {args.split} split: {len(split_sample_ids)} samples")
            else:
                print(f"Error: Split '{args.split}' not found in {splits_file}")
                return 1
        else:
            print(f"Error: Splits file not found: {splits_file}")
            return 1
    
    # ========== 加载数据 ==========
    print("\n📚 加载数据...")
    data = load_dataset(
        dataset_name=args.dataset,
        conv_id=args.conv_id,
        sample_id=args.sample_id,
        character_id=getattr(args, 'character_id', None),
        num_samples=args.questions,
        subset_file=subset_file,
        sample_ids=split_sample_ids
    )
    print(f"   加载 {len(data)} 条数据")
    
    # ========== 初始化客户端 ==========
    # 使用 create_llm_client 工厂函数 (internal or open-source, auto-detected)
    kwargs = {}
    if provider == 'vllm':
        kwargs['base_url'] = args.vllm_url
        kwargs['api_key'] = 'EMPTY'
    elif provider == 'openai':
        if args.base_url:
            kwargs['base_url'] = args.base_url
        if args.api_key:
            kwargs['api_key'] = args.api_key
    elif provider == 'metaai':
        if not _USING_INTERNAL_CLIENT:
            print("Error: --provider metaai requires llm_client_internal.py (internal only).")
            return 1
    
    llm_client = create_llm_client(provider=provider, model=model, **kwargs)
    
    # Judge uses a separate provider when the main provider can't run judge models
    judge_provider = args.judge_provider
    if judge_provider is None:
        if provider == 'vllm':
            # vLLM only serves the loaded model; judge needs a full API
            judge_provider = 'metaai' if _USING_INTERNAL_CLIENT else 'openai'
        else:
            judge_provider = provider
    
    if judge_provider == 'metaai' and not _USING_INTERNAL_CLIENT:
        print("Error: judge-provider metaai requires llm_client_internal.py (internal only).")
        return 1
    
    # Build judge kwargs: when judge uses a different provider than main,
    # pass the original (pre-vLLM-override) API config so the judge doesn't
    # accidentally point at the vLLM server.
    if judge_provider == provider:
        judge_kwargs = kwargs
    else:
        judge_kwargs = {}
        if judge_provider == 'openai':
            if _orig_api_key:
                judge_kwargs['api_key'] = _orig_api_key
            if _orig_base_url:
                judge_kwargs['base_url'] = _orig_base_url
            if not _orig_api_key:
                print("⚠️  Warning: No OpenAI API key found for judge.")
                print("   Set OPENAI_API_KEY env var or --api-key before running.")
    
    judge_client = create_llm_client(provider=judge_provider, model=judge_model, **judge_kwargs)
    judge = LLMJudge(judge_client)
    
    print(f"   Judge Provider: {judge_provider}")
    
    # ========== 按对话分组处理 (LOCOMO) ==========
    if args.dataset == 'locomo':
        all_conv_ids = sorted(set(d['conversation_id'] for d in data))
        
        if len(all_conv_ids) > 1:
            print(f"\n🔄 检测到 {len(all_conv_ids)} 个对话，将逐个处理")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(args.output_dir) / f"locomo_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for i, conv_id in enumerate(all_conv_ids, 1):
            print(f"\n{'='*60}")
            print(f"🔹 [{i}/{len(all_conv_ids)}] 处理对话: {conv_id}")
            print(f"{'='*60}")
            
            conv_questions = [d for d in data if d['conversation_id'] == conv_id]
            conv_sample = conv_questions[0]
            user_id = f'test_{conv_id}'
            
            print(f"   问题数: {len(conv_questions)}")
            print(f"   会话数: {conv_sample.get('num_sessions', 'unknown')}")
            
            # 生成数据库路径（如果用户指定了db_path，使用它；否则自动生成）
            if args.db_path:
                conv_db_path = os.path.join(args.db_path, conv_id)
            else:
                conv_db_path = get_db_path(
                    dataset_name='locomo',
                    model_name=model,
                    vector_store=args.vector_store,
                    conversation_id=conv_id
                )
            print(f"   📁 记忆路径: {conv_db_path}")
            
            # Limit sessions if requested
            if args.sessions:
                conv_sample = conv_sample.copy()
                conv_sample['haystack_sessions'] = conv_sample['haystack_sessions'][:args.sessions]
                conv_sample['haystack_session_datetimes'] = conv_sample['haystack_session_datetimes'][:args.sessions]
            
            # 初始化 MemorySystem
            memory = _create_memory_system(llm_client)
            
            # 构建记忆
            if args.mode in ['full', 'build']:
                build_result = build_memories(
                    memory, conv_sample, user_id,
                    max_sessions=args.sessions,
                    verbose=args.verbose
                )
                print(f"   ✅ 构建完成: {build_result['total_memories']} 条记忆")
                
                # 保存记忆到磁盘
                memory.save(conv_db_path)
            
            # 仅回答模式：从磁盘加载记忆
            elif args.mode == 'answer':
                if not memory.load(conv_db_path):
                    print(f"   ⚠️  跳过: 记忆数据库不存在，请先运行 --mode build")
                    continue
            
            # 回答问题
            if args.mode in ['full', 'answer']:
                current_time = get_current_time(conv_sample)
                if current_time:
                    print(f"📅 数据集时间线: {current_time}")
                
                results = answer_questions(
                    memory, conv_questions, user_id, judge,
                    limit=args.questions,
                    top_k=args.top_k,
                    current_time=current_time,
                    verbose=args.verbose,
                    concurrency=args.concurrency
                )
                
                # 保存单个对话结果
                valid_results = [r for r in results if 'llm_score' in r]
                conv_result = {
                    'conversation_id': conv_id,
                    'questions': len(valid_results),
                    'correct': sum(r['llm_score'] for r in valid_results),
                    'accuracy': round(sum(r['llm_score'] for r in valid_results) / len(valid_results) * 100, 2) if valid_results else 0,
                    'results': results
                }
                
                with open(result_dir / f'{conv_id}_results.json', 'w', encoding='utf-8') as f:
                    json.dump(conv_result, f, indent=2, ensure_ascii=False)
                
                all_results.append(conv_result)
        
        # 汇总报告
        if all_results:
            total_q = sum(r['questions'] for r in all_results)
            total_c = sum(r['correct'] for r in all_results)
            acc = total_c / total_q * 100 if total_q > 0 else 0
            
            summary = {
                'timestamp': timestamp,
                'dataset': args.dataset,
                'model': model,
                'judge_model': judge_model,
                'total_conversations': len(all_results),
                'total_questions': total_q,
                'total_correct': total_c,
                'overall_accuracy': f"{acc:.2f}%",
                'by_conversation': {r['conversation_id']: f"{r['accuracy']:.2f}%" for r in all_results}
            }
            
            with open(result_dir / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print(f"📊 总体结果")
            print(f"{'='*60}")
            print(f"   准确率: {acc:.2f}% ({total_c}/{total_q})")
            for r in all_results:
                print(f"   {r['conversation_id']}: {r['accuracy']:.2f}%")
            print(f"\n💾 结果保存到: {result_dir}/")
    
    # ========== LongMemEval: 每个样本独立处理 ==========
    elif args.dataset == 'longmemeval':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(args.output_dir) / f"longmemeval_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        # Prepare arguments for each sample
        sample_args_list = [
            (sample, i, len(data), provider, model, judge_model, judge_provider, kwargs, judge_kwargs,
             args.sessions, args.top_k, args.mode, args.verbose)
            for i, sample in enumerate(data, 1)
        ]
        
        if args.parallel and len(data) > 1:
            # Parallel processing
            print(f"\n🚀 LongMemEval 并行处理 ({args.workers} 进程, {len(data)} 样本)")
            print("=" * 60)
            
            # Use multiprocessing for true parallelism
            from multiprocessing import Pool
            
            with Pool(processes=args.workers) as pool:
                all_results = pool.map(process_longmemeval_sample, sample_args_list)
            
            # Sort by original index
            all_results.sort(key=lambda x: x.get('index', 0))
        else:
            # Sequential processing
            for i, sample in enumerate(data, 1):
                sample_id = sample['question_id']
                user_id = f'test_{sample_id}'
                
                print(f"\n{'='*60}")
                print(f"🔹 [{i}/{len(data)}] 样本: {sample_id}")
                print(f"{'='*60}")
                print(f"   会话数: {len(sample.get('haystack_sessions', []))}")
                
                # 生成数据库路径
                if args.db_path:
                    sample_db_path = os.path.join(args.db_path, sample_id)
                else:
                    sample_db_path = get_db_path(
                        dataset_name='longmemeval',
                        model_name=model,
                        vector_store=args.vector_store,
                        sample_id=sample_id
                    )
                print(f"   📁 记忆路径: {sample_db_path}")
                
                # Limit sessions if requested
                if args.sessions:
                    sample = sample.copy()
                    sample['haystack_sessions'] = sample['haystack_sessions'][:args.sessions]
                    if 'haystack_session_datetimes' in sample:
                        sample['haystack_session_datetimes'] = sample['haystack_session_datetimes'][:args.sessions]
                
                # 初始化 MemorySystem
                memory = _create_memory_system(llm_client)
                
                # 构建记忆
                if args.mode in ['full', 'build']:
                    build_result = build_memories(
                        memory, sample, user_id,
                        max_sessions=args.sessions,
                        verbose=args.verbose
                    )
                    print(f"   ✅ 构建完成: {build_result['total_memories']} 条记忆")
                    
                    # 保存记忆到磁盘
                    memory.save(sample_db_path)
                
                # 仅回答模式：从磁盘加载记忆
                elif args.mode == 'answer':
                    if not memory.load(sample_db_path):
                        print(f"   ⚠️  跳过: 记忆数据库不存在，请先运行 --mode build")
                        continue
                
                # 回答问题 (LongMemEval 每个样本只有一个问题)
                if args.mode in ['full', 'answer']:
                    current_time = get_current_time(sample)
                    
                    # 构造问题格式
                    question = {
                        'question_id': sample['question_id'],
                        'question': sample['question'],
                        'answer': sample['answer'],
                        'question_type': sample.get('question_type', 'unknown')
                    }
                    
                    results = answer_questions(
                        memory, [question], user_id, judge,
                        top_k=args.top_k,
                        current_time=current_time,
                        verbose=args.verbose
                    )
                    
                    if results:
                        all_results.extend(results)
        
        # 汇总报告
        if all_results:
            total_q = len(all_results)
            total_c = sum(r.get('llm_score', 0) for r in all_results)
            acc = total_c / total_q * 100 if total_q > 0 else 0
            
            summary = {
                'timestamp': timestamp,
                'dataset': args.dataset,
                'model': model,
                'judge_model': judge_model,
                'total_samples': total_q,
                'total_correct': total_c,
                'overall_accuracy': f"{acc:.2f}%",
            }
            
            with open(result_dir / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            with open(result_dir / 'results.json', 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print_accuracy_stats(all_results, "📊 LongMemEval 总体结果")
            print(f"\n💾 结果保存到: {result_dir}/")
    
    # ========== PerLTQA: 按人物处理 ==========
    elif args.dataset == 'perltqa':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(args.output_dir) / f"perltqa_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for i, character in enumerate(data, 1):
            char_id = character['character_id']
            char_name = character.get('character_name', char_id)
            user_id = f'test_{char_id}'
            questions = character.get('questions', [])
            
            print(f"\n{'='*60}")
            print(f"🔹 [{i}/{len(data)}] {char_name} ({char_id})")
            print(f"{'='*60}")
            print(f"   问题数: {len(questions)}")
            print(f"   会话数: {character.get('num_sessions', 'unknown')}")
            
            # 生成数据库路径
            if args.db_path:
                char_db_path = os.path.join(args.db_path, char_id)
            else:
                char_db_path = get_db_path(
                    dataset_name='perltqa',
                    model_name=model,
                    vector_store=args.vector_store,
                    character_id=char_id
                )
            print(f"   📁 记忆路径: {char_db_path}")
            
            # Limit sessions if requested
            if args.sessions:
                character = character.copy()
                character['haystack_sessions'] = character['haystack_sessions'][:args.sessions]
                if 'haystack_session_datetimes' in character:
                    character['haystack_session_datetimes'] = character['haystack_session_datetimes'][:args.sessions]
            
            # 初始化 MemorySystem
            memory = _create_memory_system(llm_client)
            
            # 构建记忆
            if args.mode in ['full', 'build']:
                build_result = build_memories(
                    memory, character, user_id,
                    max_sessions=args.sessions,
                    verbose=args.verbose
                )
                print(f"   ✅ 构建完成: {build_result['total_memories']} 条记忆")
                
                # 保存记忆到磁盘
                memory.save(char_db_path)
            
            # 仅回答模式：从磁盘加载记忆
            elif args.mode == 'answer':
                if not memory.load(char_db_path):
                    print(f"   ⚠️  跳过: 记忆数据库不存在，请先运行 --mode build")
                    continue
            
            # 回答问题
            if args.mode in ['full', 'answer']:
                current_time = get_current_time(character)
                
                results = answer_questions(
                    memory, questions, user_id, judge,
                    limit=args.questions,
                    top_k=args.top_k,
                    current_time=current_time,
                    verbose=args.verbose,
                    concurrency=args.concurrency
                )
                
                # 保存单个人物结果
                valid_results = [r for r in results if 'llm_score' in r]
                char_result = {
                    'character_id': char_id,
                    'character_name': char_name,
                    'questions': len(valid_results),
                    'correct': sum(r['llm_score'] for r in valid_results),
                    'accuracy': round(sum(r['llm_score'] for r in valid_results) / len(valid_results) * 100, 2) if valid_results else 0,
                    'results': results
                }
                
                with open(result_dir / f'{char_id}_results.json', 'w', encoding='utf-8') as f:
                    json.dump(char_result, f, indent=2, ensure_ascii=False)
                
                all_results.append(char_result)
        
        # 汇总报告
        if all_results:
            total_q = sum(r['questions'] for r in all_results)
            total_c = sum(r['correct'] for r in all_results)
            acc = total_c / total_q * 100 if total_q > 0 else 0
            
            summary = {
                'timestamp': timestamp,
                'dataset': args.dataset,
                'model': model,
                'judge_model': judge_model,
                'total_characters': len(all_results),
                'total_questions': total_q,
                'total_correct': total_c,
                'overall_accuracy': f"{acc:.2f}%",
                'by_character': {r['character_id']: f"{r['accuracy']:.2f}%" for r in all_results}
            }
            
            with open(result_dir / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print(f"📊 PerLTQA 总体结果")
            print(f"{'='*60}")
            print(f"   准确率: {acc:.2f}% ({total_c}/{total_q})")
            for r in all_results:
                print(f"   {r['character_name']}: {r['accuracy']:.2f}%")
            print(f"\n💾 结果保存到: {result_dir}/")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='MemBuilder Unified Evaluation')
    
    # Dataset options
    parser.add_argument('--dataset', choices=['locomo', 'longmemeval', 'perltqa'], 
                       default='locomo', help='Dataset to use')
    parser.add_argument('--conv-id', default=None, help='Conversation ID (for locomo)')
    parser.add_argument('--sample-id', default=None, help='Sample ID (for longmemeval)')
    parser.add_argument('--character-id', default=None, help='Character ID (for perltqa)')
    parser.add_argument('--subset-file', default=None, help='Path to subset config file')
    parser.add_argument('--split', choices=['sft', 'rl', 'test'], default=None,
                       help='Use predefined split from data/longmemeval/splits/longmemeval_splits.json')
    
    # Mode options
    parser.add_argument('--mode', choices=['build', 'answer', 'full'], 
                       default='full', help='Test mode')
    
    # Limit options
    parser.add_argument('--sessions', type=int, default=None, help='Limit sessions to build')
    parser.add_argument('--questions', type=int, default=None, help='Limit questions to test')
    
    # Model configuration
    parser.add_argument('--model', default=None, help='LLM model for memory agents')
    parser.add_argument('--judge-model', default=None, help='LLM model for answer judge')
    
    # Provider options
    parser.add_argument('--provider', default='openai', 
                       choices=['openai', 'vllm', 'metaai'],
                       help='LLM provider (default: openai). Use --base-url to specify API endpoint.')
    parser.add_argument('--judge-provider', default=None,
                       choices=['openai', 'vllm', 'metaai'],
                       help='Judge LLM provider (default: same as --provider, or metaai when --provider=vllm)')
    parser.add_argument('--base-url', default=None, help='OpenAI API base URL')
    parser.add_argument('--api-key', default=None, help='OpenAI API key')
    parser.add_argument('--vllm-url', default='http://localhost:8000/v1', help='vLLM server URL')
    
    # Retrieval options
    parser.add_argument('--top-k', type=int, default=QA_ANSWERING_TOP_K, help='Top-K memories')
    
    # Persistence options
    parser.add_argument('--db-path', default=None,
                       help='Custom database path for memory persistence (auto-generated if not specified)')
    parser.add_argument('--vector-store', default='faiss', choices=['faiss'],
                       help='Vector store type (default: faiss)')
    
    # Concurrency options
    parser.add_argument('--concurrency', type=int, default=1,
                       help='Concurrent workers for answering questions (LOCOMO)')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel sample processing (LongMemEval)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for LongMemEval sample processing')
    
    # Output options
    parser.add_argument('--output-dir', default='logs', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    return run_evaluation(args)


if __name__ == "__main__":
    exit(main())
