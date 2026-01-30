"""
LLM Judge for answer correctness evaluation.

Based on the LOCOMO evaluation methodology, adapted for MemBuilder.
"""

import json
from typing import Optional

# 详细的评估 Prompt（参考老代码 llm_judge.py）
ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Handling "Not answerable" cases:
1. If the GOLD answer is "Not answerable" (meaning the information truly doesn't exist in the conversation history):
   - The generated answer should be CORRECT if it clearly indicates unavailability
   - Accept equivalent expressions: "Not answerable", "There is no information", "There is no direct record", "does not appear to be", "no explicit mention", "cannot be determined", "no specific details available"
   - As long as the generated answer conveys that the information is unavailable, count it as CORRECT

2. If the GOLD answer is a SPECIFIC answer (e.g., "7 May 2023", "John", "Paris"):
   - The generated answer saying "Not answerable" should be counted as WRONG
   - This means the system failed to retrieve information that actually exists in the conversation history
   - Even if phrased as "no information available" or similar, it's still WRONG when the gold answer is specific
   - IMPORTANT: Even if the generated answer mentions the correct information but attributes it to a DIFFERENT person/entity than asked in the question, it should be counted as WRONG. For example, if the question asks about "Alice's opinion" but the answer says "Bob thinks X" (even if X matches the gold answer), this is WRONG because it answers about the wrong person.

3. CRITICAL RULE for "Not answerable" responses:
   - When the generated answer indicates "Not answerable" or similar (cannot find, no information, etc.), the ONLY way it can be CORRECT is if the GOLD answer is ALSO "Not answerable"
   - If the gold answer contains ANY specific information (names, dates, facts, opinions, etc.), then a "Not answerable" response is ALWAYS WRONG, regardless of any explanation or reasoning provided in the generated answer
   - Do NOT be misled by keywords in the explanation - focus on whether the answer actually provides the requested information

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


class LLMJudge:
    """LLM-based answer evaluation judge."""
    
    def __init__(self, llm_client, model: str = None):
        """
        Initialize LLM Judge.
        
        Args:
            llm_client: LLM client for API calls
            model: Model to use for judging (default: from config)
        """
        self.llm_client = llm_client
        self.model = model
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks and extra text."""
        import re
        if not text:
            return text
        
        # Remove markdown code blocks
        clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*\}', clean, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return clean
    
    def evaluate(self, question: str, gold_answer: str, generated_answer: str) -> int:
        """
        Evaluate the generated answer against the gold answer.
        
        Args:
            question: The question being asked
            gold_answer: The ground truth answer
            generated_answer: The generated answer to evaluate
        
        Returns:
            1 if CORRECT, 0 if WRONG
        """
        prompt = ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer
        )
        
        response_text = ""
        try:
            response_text = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # 尝试解析 JSON（使用 extract_json 清理）
            try:
                clean_json = self._extract_json(response_text)
                result = json.loads(clean_json)
                label = result.get("label", "WRONG")
                return 1 if label == "CORRECT" else 0
            except json.JSONDecodeError:
                # 回退：从文本中提取
                upper_text = response_text.upper()
                if "CORRECT" in upper_text and "WRONG" not in upper_text:
                    return 1
                elif "WRONG" in upper_text:
                    return 0
                # 如果两者都有或都没有，默认 WRONG
                return 0
                
        except Exception as e:
            # 最终回退：检查是否有 CORRECT
            upper_text = response_text.upper() if response_text else ""
            if "CORRECT" in upper_text and "WRONG" not in upper_text:
                return 1
            return 0


def evaluate_answer(
    question: str,
    gold_answer: str,
    generated_answer: str,
    llm_client,
    judge_model: str = None
) -> int:
    """
    Evaluate the generated answer against the gold answer using an LLM judge.
    
    Convenience function that wraps LLMJudge.
    
    Args:
        question: The question being asked
        gold_answer: The ground truth answer
        generated_answer: The generated answer to evaluate
        llm_client: LLM client for judge calls
        judge_model: Model to use for judging (optional)
    
    Returns:
        1 if CORRECT, 0 if WRONG
    """
    judge = LLMJudge(llm_client, model=judge_model)
    return judge.evaluate(question, gold_answer, generated_answer)
