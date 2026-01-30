"""
Evaluation metrics computation.
"""

from typing import Dict, List, Any
from collections import defaultdict


def compute_accuracy(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute accuracy metrics from evaluation results.
    
    Args:
        results: List of dicts with 'correct' or 'llm_score' (0 or 1) 
                 and optional 'type' or 'question_type' fields
    
    Returns:
        Dict with overall and per-type accuracy
    """
    if not results:
        return {"overall": 0.0, "total": 0, "correct": 0}
    
    # 支持两种字段名
    def get_score(r):
        return r.get("correct", r.get("llm_score", r.get("is_correct", 0)))
    
    def get_type(r):
        return r.get("type", r.get("question_type", "unknown"))
    
    # Overall accuracy
    total_correct = sum(get_score(r) for r in results)
    overall = total_correct / len(results)
    
    # Per-type accuracy
    type_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        q_type = get_type(r)
        type_results[q_type]["total"] += 1
        type_results[q_type]["correct"] += get_score(r)
    
    per_type = {
        t: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for t, stats in type_results.items()
    }
    
    return {
        "overall": overall,
        "per_type": per_type,
        "total": len(results),
        "correct": total_correct,
        "by_type_counts": dict(type_results)
    }


def print_accuracy_stats(results: List[Dict], title: str = "📊 评估结果") -> tuple:
    """
    统计并打印准确率（含按类型分组）
    
    Args:
        results: 评估结果列表
        title: 标题
    
    Returns:
        (correct_count, accuracy_percentage)
    """
    stats = compute_accuracy(results)
    
    print(f"\n{title}:")
    print(f"   准确率: {stats['correct']}/{stats['total']} ({stats['overall']*100:.2f}%)")
    
    if stats.get('per_type'):
        print(f"\n📊 按问题类型:")
        for q_type, acc in sorted(stats['per_type'].items()):
            counts = stats['by_type_counts'].get(q_type, {})
            total = counts.get('total', 0)
            correct = counts.get('correct', 0)
            print(f"   {q_type}: {correct}/{total} ({acc*100:.2f}%)")
    
    return stats['correct'], stats['overall'] * 100
