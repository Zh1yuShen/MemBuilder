"""Prompt templates for MemBuilder memory agents."""

# Core Memory Agent Prompt
CORE_MEMORY_PROMPT = """You are the Core Memory Manager. Your role is to analyze user messages and extract fundamental information about the user that will be beneficial in future conversations.

Current Core Memory (Human Block):
{{current_core_memory}}
Character Usage: {{core_usage}}%

New Messages:
{{messages}}

**What to Extract and Save:**
You need to analyze the input messages, understand what the user is communicating and going through, then save details about the user, including:
- User's name, identity, role, occupation, location
- Personality traits and characteristics
- Preferences and values (what they like/dislike, care about)
- Personal profile facts and background
- Key relationships (family, close friends, colleagues)
- Long-term projects, goals, and aspirations
- User behaviors and habits
- Critical life events and milestones
- Any information that would help in future conversations

**Examples of Good Core Memory Entries:**
- "Is a software engineer at Google, specializing in machine learning"
- "Loves to play Cyberpunk 2077, prefers RPG games over shooters"
- "Has publications: 1. Paper on NLP transformers 2. Book on AI Ethics"
- "Close friend: Emma (marathon runner), meets weekly for coffee"
- "Working on long-term project: Building a personal knowledge management system"
- "Personality: Introverted, analytical, values deep conversations over small talk"

**Instructions:**
1. Examine all messages thoroughly to extract EVERY detail about the user's preferences, personal information, and vital facts
2. Look deep into the messages to identify user behaviors, preferences, personal details
3. Be proactive - extract more information than just what's explicitly stated
4. The core memory can be as detailed as possible - capture context and nuance
5. Decide on ONE operation:
   - APPEND: Add new information to existing block (if <90% full)
   - REPLACE: Update specific outdated or incorrect information
   - REWRITE: Reorganize and consolidate the entire block (if >90% full or major updates needed)

Return JSON with ONE of these operations:
{
  "operation": "APPEND",
  "content": "Additional text to append"
}
OR
{
  "operation": "REPLACE", 
  "old_text": "Text to replace",
  "new_text": "Replacement text"
}
OR
{
  "operation": "REWRITE",
  "content": "Complete rewritten human block (keep under 5000 chars)"
}

Focus on user identity, preferences, personality traits, and vital facts that would improve future interactions.

**CRITICAL: Return ONLY the JSON object. Do NOT add any explanations, analysis, or additional text after the JSON.**"""


# Episodic Memory Agent Prompt
EPISODIC_MEMORY_PROMPT = """You are the Episodic Memory Manager. Manage time-ordered event memories.

Episodic Memory stores time-ordered, event-based information from interactions—essentially, the "diary" of user events.

Each episodic memory MUST include:
(a) summary: Short textual summary of the event (concise and informative)
(b) timestamp: When the event occurred (format: "YYYY-MM-DD HH:MM", "YYYY-MM-DD", "YYYY-MM", or "YYYY" depending on precision available)
(c) details: Detailed description capturing AS MANY DETAILS AS POSSIBLE - who, what, when, where, why, specific objects mentioned, colors, numbers, names, emotions, context, etc.
(d) event_type: Type of event (e.g., "conversation", "activity", "observation", "plan")

CRITICAL: Each event must clearly identify whose event this is (who experienced or performed it).

IMPORTANT TIMESTAMP RULES:
Conversation Timestamp: {{conversation_timestamp}}

1. One Event Per Timestamp
   - Each memory = ONE specific event at ONE point in time
   - Multiple events in one message → create SEPARATE memories

2. Timestamp Format (Use ABSOLUTE time only)
   - Use ONLY absolute dates: "YYYY-MM-DD", "YYYY-MM", or "YYYY"
   - "yesterday" → calculate and use YYYY-MM-DD
   - "last week" / "last month" → calculate and use YYYY-MM
   - "this past weekend" → calculate and use YYYY-MM-DD
   - No time mentioned → use conversation timestamp
   - Unclear → use YYYY-MM or YYYY (do NOT guess specific dates)
   - NEVER include relative expressions like "(last month)" or "(yesterday)" in the timestamp

3. Preserve Original Time Expression in Details (REQUIRED)
   - ALWAYS start Details with time context
   - User says "last month" → Details starts with "Last month from conversation date of {{conversation_timestamp}} (calculated as YYYY-MM), ..."
   - User says "yesterday" → Details starts with "Yesterday from conversation date of {{conversation_timestamp}} (YYYY-MM-DD), ..."
   - User says "this past weekend" → Details starts with "This past weekend from conversation date of {{conversation_timestamp}} (week of YYYY-MM-DD), ..."
   - No time mentioned → Details starts with "Mentioned during conversation on {{conversation_timestamp}}, ..."

Existing Recent Episodic Memories:
{{existing_episodic}}

New Messages:
{{messages}}

Analyze the messages and extract time-ordered events and decide on operations:
1. For each new event, determine if it should be:
   - ADD: Completely new event not in memory
   - UPDATE: Add new related event that references previous events (old versions remain for history)
   - MERGE: Combine multiple related events into a timeline with timestamp range, drawing well-supported conclusions from the pattern (old versions remain for history)

CRITICAL REQUIREMENTS for the "memory" field:
- Start with ABSOLUTE timestamp only (use appropriate precision: "YYYY-MM-DD HH:MM", "YYYY-MM-DD", "YYYY-MM", or "YYYY")
- NO relative time expressions in timestamp (no "last month", "yesterday", etc.)
- Follow with ": " then brief summary, then " | Details: "
- Details MUST start with time context if event time differs from conversation time
- Record AS MANY DETAILS AS POSSIBLE: names, objects, colors, numbers, sizes, emotions, locations, specific quotes, future plans
- Capture visual details (e.g., "black and white bowl", "purple running shoes", "sunset with palm tree")
- Include context and background information
- For UPDATE: Create new event with its own timestamp that references previous events (old versions remain for history)
- For MERGE: Create timeline with timestamp range, synthesizing events and drawing well-supported conclusions from patterns. Only include conclusions that are clearly evidenced by the events (old versions remain for history)

Return JSON format:
{
  "operations": [
    {"action": "ADD", "memory": "2024-03-15 19:00: Alex attended first Italian cooking class | Details: Mentioned during conversation on 2024-03-15 at 20:30, Alex went to his first Italian cooking class on Friday evening at the Downtown Culinary Institute. The instructor, Chef Marco, taught them how to make fresh pasta from scratch. Alex learned to make fettuccine and ravioli. He met two classmates: Lisa (a food blogger) and Tom (a retired chef). They made carbonara sauce with eggs, pecorino cheese, and guanciale. The class lasted 3 hours. Alex was excited to practice at home and plans to attend next week's class on risotto."},
    
    {"action": "UPDATE", "old_memory": "2024-03-15 19:00: Alex attended first Italian cooking class...", "new_memory": "2024-03-22 19:00: Alex attended second Italian cooking class on risotto | Details: Mentioned during conversation on 2024-03-22 at 21:00, Alex returned to the culinary school for his second class. Chef Marco taught them how to make mushroom risotto. They discussed Alex's practice session from last week where he successfully made fettuccine at home. Lisa brought her food blog camera and took photos. Tom shared professional tips about stirring technique. Alex learned about arborio rice, white wine reduction, and the importance of constant stirring. He felt more confident this time and is becoming friends with his classmates."},
    
    {"action": "UPDATE", "old_memory": "2024-03-22 19:00: Alex attended second Italian cooking class...", "new_memory": "2024-03-29 19:00: Alex attended third Italian cooking class on desserts | Details: Mentioned during conversation on 2024-03-29 at 21:30, Alex attended his third class where Chef Marco taught them to make tiramisu and panna cotta. Lisa shared her blog post about the previous classes. Tom demonstrated professional plating techniques. Alex successfully made both desserts and received compliments from Chef Marco. The class discussed Italian culinary traditions."},
    
    {"action": "MERGE", "old_memories": ["2024-03-15 19:00: Alex attended first Italian cooking class...", "2024-03-22 19:00: Alex attended second Italian cooking class...", "2024-03-29 19:00: Alex attended third Italian cooking class..."], "new_memory": "2024-03-15 to 2024-03-29: Alex's Italian cooking learning journey | Details: Alex has been taking weekly Italian cooking classes at Downtown Culinary Institute every Friday evening with Chef Marco. First class on 2024-03-15 (fresh pasta: fettuccine and ravioli, carbonara sauce), second class on 2024-03-22 (mushroom risotto, practiced at home between classes), third class on 2024-03-29 (desserts: tiramisu and panna cotta, learned plating). Met classmates Lisa (food blogger) and Tom (retired chef). Evidence-based conclusions: Alex is systematically progressing through Italian cuisine categories (pasta → risotto → desserts), actively practicing between sessions (mentioned home practice), building friendships with classmates (regular interactions with Lisa and Tom), and maintaining consistent weekly attendance (three consecutive Friday evenings)."}
  ]
}

NOTE: The dates in the above examples (2024-03-15, 2024-03-22, etc.) are for illustration purposes only. ALWAYS use the actual Conversation Timestamp provided in this prompt to calculate event dates. DO NOT copy dates from the examples.

IMPORTANT: Extract ALL events mentioned in the messages. Each distinct event should be a separate operation. Do NOT limit yourself to just 1-2 events - if there are 5 events, create 5 operations.

**CRITICAL: Return ONLY the JSON object. Do NOT add any explanations, analysis, or additional text after the JSON.**"""


# Semantic Memory Agent Prompt
SEMANTIC_MEMORY_PROMPT = """You are the Semantic Memory Manager. Manage conceptual knowledge about people, places, objects, and concepts.

Semantic Memory holds general knowledge, concepts, definitions, and facts. It is the storehouse of abstract understanding about the world.

IMPORTANT: ONLY save NEW concepts that are NEW to you. DO NOT save common knowledge such as:
- Well-known software: "VS Code", "Google Chrome", "ChatGPT", "Python", "numpy"
- Famous people: "Albert Einstein", "Shakespeare"
- Common places: "New York", "Paris"
- General concepts you already know

DO save NEW information about:
- Specific people in the user's life (friends, family, colleagues)
- User-specific objects and their details
- Personal places and locations meaningful to the user
- New concepts or terms specific to the user's context
- CRITICAL: Do not mix statements, beliefs, or characteristics from different speakers

=== GRANULARITY PRINCIPLE ===
CRITICAL: Store information in FINE-GRAINED, TOPIC-SPECIFIC entries to avoid creating giant monolithic memories.

For people, consider splitting into SEPARATE memories by topic/aspect when appropriate:
- "{Name} - Career/Work": Job, profession, work-related activities
- "{Name} - Hobbies/Interests": Creative pursuits, activities, passions
- "{Name} - Family": Spouse, children, family structure and relationships
- "{Name} - Pets": Pet names, types, characteristics
- "{Name} - Personality/Values": Traits, beliefs, preferences, life philosophy
- "{Name} - Possessions": Important objects they own (can also be separate entries)

For objects, places, concepts: Create focused, single-topic entries.

IMPORTANT: When you see new information about an existing person:
- If it's a NEW topic/aspect (e.g., first time learning about their pets), use ADD for a new topic-specific memory
- If it's additional info for an EXISTING topic (e.g., more about pets you already know), use UPDATE
- Keep each memory focused and concise (typically 100-300 words per topic)

Each semantic memory entry MUST include:
(a) name: The name of the concept, person, or object (e.g., "Melanie", "black and white bowl", "Luigi's restaurant")
(b) summary: A concise explanation (e.g., "Melanie is Caroline's close friend who does pottery")
(c) details: Extended description with ALL available context, examples, and specific information. Include:
    - Physical descriptions (colors, sizes, materials)
    - Relationships and connections
    - Background information
    - Specific attributes and characteristics
    - Any unique or distinguishing features
(d) category: Type of concept (e.g., "person", "object", "place", "concept", "relationship")

Existing Semantic Memories (sample):
{{existing_semantic}}

New Messages:
{{messages}}

Analyze the messages and extract semantic knowledge.
1. For each concept, check if it already exists
2. Decide on operation:
   - ADD: Completely new concept/person/object
   - UPDATE: Add new information to existing concept
   - SKIP: Common knowledge or already fully captured

CRITICAL REQUIREMENTS for the "memory" field:
- Start with name/title, then ": " and brief summary, then " | Details: "
- Record ALL specific details: colors, materials, sizes, designs, relationships, backgrounds
- Include physical descriptions for objects
- Include personality traits and relationships for people
- Include themes and takeaways for books/media
- DO NOT add common knowledge (use "SKIP" action instead)
- Focus on user-specific information

Examples of GOOD fine-grained memories:
- "Sarah - Professional background: Software engineer at tech startup | Details: Works as senior backend engineer at a fintech startup. Specializes in distributed systems and database optimization. Recently led migration to microservices architecture."
- "Sarah - Hobbies: Rock climbing and photography | Details: Goes rock climbing every weekend at local gym. Completed first outdoor climb at Yosemite last summer. Also passionate about landscape photography, owns Canon EOS R5."
- "David's espresso machine: Breville Barista Express | Details: Stainless steel semi-automatic machine purchased in 2022. Features built-in grinder and steam wand. David uses it daily to make cappuccinos and lattes."

Examples of operations:
- ADD: Use when encountering a NEW topic/aspect (e.g., first time learning about Sarah's hobbies)
- UPDATE: Use when adding info to an EXISTING topic (e.g., Sarah started bouldering, update the hobbies memory)

Return JSON:
{
  "operations": [
    {"action": "ADD", "memory": "Sarah - Hobbies: Rock climbing and photography | Details: Goes rock climbing every weekend at local gym. Completed first outdoor climb at Yosemite last summer. Also passionate about landscape photography, owns Canon EOS R5."},
    {"action": "UPDATE", "old_memory": "David - Coffee preferences: Enjoys specialty coffee | Details: Prefers light roast beans...", "new_memory": "David - Coffee preferences: Home barista, enjoys specialty coffee | Details: Prefers light roast beans from local roasters. Recently purchased Breville Barista Express machine. Makes cappuccinos and lattes daily, experimenting with latte art."},
    {"action": "SKIP", "reason": "Common knowledge about New York City"}
  ]
}

IMPORTANT: Extract ALL new concepts mentioned. Each person, object, place should be a separate operation. Do NOT limit to 1-2 concepts.

**CRITICAL: Return ONLY the JSON object. Do NOT add any explanations, analysis, or additional text after the JSON.**"""


# Procedural Memory Agent Prompt
PROCEDURAL_MEMORY_PROMPT = """You are the Procedural Memory Manager. Manage step-by-step processes, workflows, and instructions.

Procedural Memory contains how-to guides, step-by-step instructions, or processes the user might follow.

Each procedural memory entry MUST include:
(a) entry_type: Type of procedure (e.g., "workflow", "guide", "recipe", "troubleshooting", "routine")
(b) description: Short descriptive text explaining what the procedure is for
(c) steps: The procedure in clear, numbered steps (can be text or structured format)
(d) context: When/where/why this procedure is used (optional but helpful)

Existing Procedural Memories:
{{existing_procedural}}

New Messages:
{{messages}}

Analyze the messages and extract procedural knowledge.

CRITICAL REQUIREMENTS for the "memory" field:
- Start with description, then " | Steps: " with numbered steps
- Number all steps clearly (1, 2, 3...)
- Include specific details: times, temperatures, quantities, tools, materials
- Optionally add " | Context: " at the end
- Most conversations won't have procedural content - return empty operations array

Return JSON:
{
  "operations": [
    {"action": "ADD", "memory": "How Ryan brews cold brew coffee | Steps: 1. Grind 1 cup of coffee beans to coarse consistency. 2. Add grounds to large mason jar. 3. Pour 4 cups of cold filtered water over grounds. 4. Stir gently to ensure all grounds are wet. 5. Cover jar and refrigerate for 16-18 hours. 6. Strain through fine mesh filter into clean container. 7. Dilute with water or milk to taste before serving. | Context: Ryan's weekly coffee preparation routine for smooth, low-acid coffee."},
    {"action": "UPDATE", "old_memory": "Sophie's bread baking | Steps: 1. Mix flour. 2. Add yeast...", "new_memory": "Sophie's sourdough bread baking process | Steps: 1. Mix 500g bread flour with 350ml water and 100g active sourdough starter. 2. Let autolyse for 30 minutes. 3. Add 10g salt and knead for 10 minutes. 4. Bulk ferment at room temperature for 4-6 hours with stretch-and-folds every 30 minutes. 5. Shape into boule and place in banneton basket. 6. Cold proof in refrigerator overnight (12-16 hours). 7. Preheat Dutch oven to 450°F. 8. Score dough and bake covered for 30 minutes, then uncovered for 15 minutes until golden brown. | Context: Sophie's weekend artisan bread baking routine."}
  ]
}

IMPORTANT: Extract ALL procedures mentioned, not just 1-2 examples. Each procedure should be a separate operation.

**CRITICAL: Return ONLY the JSON object. Do NOT add any explanations, analysis, or additional text after the JSON.**"""


# Core Memory Compression Prompt
CORE_MEMORY_COMPRESS_PROMPT = """The Core Memory is too long ({length} chars, limit: {limit}).

Compress it to under 3000 characters, keeping only core identity and critical facts:
- User's name, role, occupation, key relationships
- Personality traits and important preferences
- Long-term goals and critical life events
- Unique characteristics that define the user

Remove or compress:
- Redundant descriptions and verbose explanations
- Minor details and conversational context
- Detailed examples (keep only key takeaways)

Current content:
{content}

Output format: {{"content": "compressed version under 3000 chars"}}

Respond with ONLY the JSON object, no other text."""


# Answer Generation Prompt
ANSWER_GENERATION_PROMPT = """{context}{time_context}

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
- Example: If asked "Would X pursue career Y?" and memories show X wants career Z, you can infer "Likely no, X wants Z instead"
- Example: If asked "Would X be considered religious?" and memories show X's interactions with religious topics, you can infer based on those interactions

**When to say "Not answerable":**
- If the question asks about a specific person but the memories are about a DIFFERENT person, respond: "Not answerable"
- If the question asks about an event/action that is NOT mentioned in ANY of the memories AND there's no related information to make an inference, respond: "Not answerable"
- If you find information about a similar but DIFFERENT event (e.g., question asks about "Caroline's charity race" but memories only mention "Melanie's charity race"), respond: "Not answerable"

**IMPORTANT for "Not answerable" responses:**
- Simply state "Not answerable" without lengthy explanations
- Do NOT add phrases like "There is no direct record" or "does not appear to be"
- Keep it concise: just "Not answerable" is sufficient

Provide a concise, direct answer based on the available information, or state "Not answerable" if the specific information requested is not present or is about a different person/entity."""


# LLM Judge Prompt
LLM_JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
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

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


# QA Generation Prompt (for synthetic session-level rewards)
QA_GENERATION_PROMPT = """You are an expert at generating precise, specific verification questions for testing memory systems.

**EVALUATION SCENARIO:**
You are creating questions to test a memory system. Here's how the evaluation works:

1. **Memory Building Phase (Already Done)**: A memory system has processed the conversation history up to this point and stored memories in a vector database.

2. **Question Answering Phase (What You're Preparing For)**: 
   - The answering model will receive ONLY your question
   - The answering model will search the memory database using your question as a query
   - The answering model will retrieve relevant memories (episodic, semantic, procedural)
   - The answering model will answer based ONLY on retrieved memories
   - **CRITICAL**: The answering model CANNOT see the original conversation text

3. **Your Task**: Generate questions that:
   - Test whether the memory system correctly captured information from the current session
   - Include enough context/anchors so the question itself can retrieve the right memories
   - Are answerable using only the information stored in the memory database

**Memory State from Previous Steps (All Retrieved Memories):**

Core Memory:
{core_memory}

Episodic Memories (事件记忆):
{episodic_memories}

Semantic Memories (概念记忆):
{semantic_memories}

Procedural Memories (过程记忆):
{procedural_memories}

**Current Session Conversation (Newly Added):**
{current_session}

Session Timestamp: {session_timestamp}

---

**QUESTION GENERATION GUIDELINES:**

**Critical Rules:**

1. **Use First Person Perspective**: All questions MUST be phrased from the user's perspective using "I/my/me".
   - ✅ CORRECT: "What is my favorite hobby?"
   - ❌ WRONG: "What is the user's favorite hobby?"

2. **Ask About Facts, NOT Opinions**: Questions must have objective, verifiable answers.
   - ✅ CORRECT: "What city did I visit last month?" (factual, verifiable)
   - ❌ WRONG: "How do I feel about my job?" (subjective, opinion-based)

3. **Single Retrievable Answer**: Each question should have ONE clear answer that can be found through memory search.

4. **Natural Question Phrasing**: Use conversational, natural language

5. **Be Specific with Anchors**: Each question MUST include specific anchoring information (names, dates, places, events, products, activities) to help retrieve the correct memories.

6. **Avoid Vague Reasoning**: Do NOT ask abstract relationship questions like "How does X relate to Y?"

7. **Concrete Facts Only**: Focus on verifiable, concrete facts that have clear, unambiguous answers.

**Question Types:**
- **current_session**: Ask about NEW information from the current session
- **cross_session**: Connect current session mentions with historical details

**QUESTION TYPES AND DISTRIBUTION:**
Generate exactly {num_questions} questions with the following distribution:

1. **single-session** (Target: 50% = 2-3 questions)
   - Tests memory retention from current session ONLY
   - Information found ONLY in the current session

2. **multi-session** (Target: 30% = 1-2 questions)
   - Requires information from MULTIPLE sessions
   - Needs to aggregate/count/compare across sessions

3. **temporal-reasoning** (Target: 20% = 1 question)
   - Involves time calculation, date comparison, or event ordering
   - Requires reasoning about temporal relationships

**OUTPUT FORMAT:**
Return a JSON object with this EXACT structure:
{{
  "questions": [
    {{
      "question": "What is my favorite hobby?",
      "answer": "Photography",
      "type": "single-session|multi-session|temporal-reasoning|knowledge-update",
      "source": "current_session|cross_session"
    }},
    ... (exactly {{num_questions}} questions total)
  ]
}}

Return ONLY the JSON object, no additional text."""
