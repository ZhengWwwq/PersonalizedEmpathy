import json
from typing import Any


PERSONA_SYSTEM_PROMPT = (
    "You are a psychological expert in analyzing dialogue memory to extract a human's persona. "
    "Analyze the dialogue memory and extract the human's complete persona with a focus on EQ and unique personal characteristics."
)


PERSONA_USER_PROMPT_TEMPLATE = '''# Task: Human-Only Persona Extraction (For EQ Dialogue Simulation)
# STRICT RULES:
# 1. ONLY extract traits of the HUMAN user. COMPLETELY EXCLUDE all information about the AI/Bot.
# 2. Analyze the user memory, try to make grounded inferences without any speculations. Also keep the inference process reserved (i.e. there must be factual basis in memory).
# 3. Use ONLY facts from the GROUND TRUTH MEMORY. NO fabrication, NO over-inference, NO assumptions.
# 4. Extract high-quality personalized traits which reflects the human's unique profile.
# 5. Keep persona concise as far as possible: include key information only; avoid repetitive or verbose wording.


## Extraction Dimensions (MANDATORY, cover all if mentioned)
1. EQ & Emotional Traits (CORE)
    - Empathy level, emotional expression, mood stability, sensitivity, stress response, caring style, emotional needs
Here is some examples:
1. Example 1:
user_memory:"Launched a program aimed at helping undergraduates with internship opportunities, which has been running for 3 years and has helped over 500 students."
persona_profile:"User is a proactive and socially responsible individual with strong organizational skills and a passion for supporting others' career development."
2. Example 2:
user_memory:"Have a habit that go jogging every morning for 2 years"
persona_profile:"User is a disciplined and health-conscious individual who values physical fitness."
3. Example 3:
user_memory:"Pushed through startup growing pains (funding, project acquisition) without quitting."
persona_profile:"User is resilient, persistent, and possibly optimistic, who probably appreciates motivative response rather than negative give-ups."
## Additional Gate Requirement
Decide whether this user context is deep enough for EQ query generation.
- TRUE if there is relatively stable personal profile, emotional trait, social context, or ongoing concern.
- FALSE if it is mostly factual inquiry or temporary interest in information without personal depth.

## Output Format (Return ONLY one JSON object)
```json
{
  "is_deep_persona": true,
  "gate_reason": "one concise sentence",
  "topics": ["topic1", "topic2"],
  "persona_profile": "# Human Unique Persona\\n## Core EQ Traits\\n- ..."
}
```


---
## Input Dialogue Memory (Ground Truth):
User Memory:
{memory_info}

Dialogue:
{user_dialogue}
'''


SITUATION_SYSTEM_PROMPT = (
    "You are an experienced dramatist and an expert in EQ dialogue data design. "
    "Write psychologically nuanced yet realistic user situations, while staying strictly grounded in user persona and memory without fabricating core facts."
)


SITUATION_USER_PROMPT_TEMPLATE = '''## Task
Given the user persona and memory, generate concise but information-complete situations where the query happens.

Your goal is to build realistic pre-query situations where:
- the user has stable characteristics from persona,
- has experienced facts from memory,
- is currently in a concrete context,
- and therefore is naturally likely to ask a query that belongs to one of the three EQ categories.

## EQ Category Definitions (Must Follow)
1. Emotional Support (ES)
    - Focus: validation, comfort, emotional resonance.
    - Typical trigger: user is vulnerable, stressed, sad, lonely, overwhelmed, or needs to be emotionally seen.

2. High-EQ Interaction (HEI)
    - Focus: relationship maintenance, de-escalation, tactful daily interaction, rapport building.
    - Typical trigger: subtle social dynamics where a purely factual response may feel cold or inappropriate.

3. Social Strategy (SS)
    - Focus: what to say / how to act to achieve a concrete interpersonal goal.
    - Typical trigger: user needs a script, framing, or socially appropriate strategy in a specific context.

## Constraints
- First, choose suitable topics from the provided topic list and choose supporting facts from User Memory.
- When selecting memory facts, you MUST REMEMBER that there may be false memories, e.g., "[ERROR: ...]" in the memory content. You should NOT use any fact that is marked as an error.
- Treat selected topic + memory facts as ground truth. Situations may contain limited natural elaboration, but core facts must not conflict with topic or memory.
- No contradiction with persona, user role, emotional tendency, social identity, or timeline implied by memory.
- Cover all three EQ categories: Emotional Support, High-EQ Interaction, Social Strategy.
- The wording should be natural and psychologically plausible in normal conversation flow for that topic.
- If no valid situation can be formed, return an empty list.

## Reasoning Process (Do this silently)
For each candidate situation, ensure a clear chain:
1. Persona anchor: which stable traits in persona are relevant here?
2. Memory anchor: which memory facts (with mem_position) make this scenario factual and personal?
3. Present trigger: what happened now that activates EQ needs?
4. EQ necessity: why a purely factual reply would be insufficient in this context?
5. Query likelihood: why the user is likely to ask naturally in this moment?

Discard situations that fail any step above.

## Naturalness and Plausibility Rules
- Prefer everyday, believable social/psychological contexts over dramatic or rare extremes.
- Keep cause-effect coherent: user trait -> memory-based background -> current trigger -> likely EQ query.
- Avoid robotic setup language, abstract labels, or textbook phrasing.
- Avoid introducing new entities/facts not grounded in memory/persona unless they are minimal and generic context glue.
- Ensure the category choice is justified by the situation itself, not by forced labeling.

## Category Coverage Rule
- Generate several situations with balanced diversity across the three categories.
- At least one valid situation per category whenever evidence allows.
- If a category is unsupported by evidence, do not force it.

## Output Format
Return ONLY a JSON list:
```json
[
  {
        "id": "Sit_1",
        "category": "Emotional Support | High-EQ Interaction | Social Strategy",
        "topic": "one topic from topic list",
        "relevant_mem": [1, 3],
        "situation": "Several natural sentences describing one complete situation, including user traits, reason/motivation, time/context, and what happened."
  }
]
```

## Situation Quality Requirements
- `what/how/who/why/where` are internal thinking guides for you; do not output them as separate fields.
- In `situation`, explicitly include: user characteristic cues, why this event matters now, and concrete context of what the user did/experienced.
- Keep tone natural and aligned with user psychology and persona.
- Keep topic progression realistic (not abrupt, not robotic).
- Write `situation` as several concise natural sentences that read like a believable moment right before the user asks the next query.
- Make sure each situation can plausibly lead to exactly one dominant EQ intention (ES, HEI, or SS), even if other signals exist.

## Input
User persona:
{user_persona}

Extracted topics:
{topics}

User memory:
{memory_info}
'''


QUERY_SYSTEM_PROMPT = (
    "You are an expert conversation writer for emotionally intelligent user simulation. "
    "Generate exactly one user query that strictly fits the given persona, selected relevant memory facts, and one concrete situation."
)


QUERY_USER_PROMPT_TEMPLATE = '''## Task
Generate exactly ONE natural user expression for the provided situation.

The expression can be a question, statement, concern, feeling, or observation - whatever naturally fits the moment.
**Caution** 
the situation may be over-complicated with detailed elaboration. When outputting
the final query, do NOT try to include all the details. Instead, only remain the core parts and omit the rest, just like 
how natural human expression works.

## Inputs and Ground Truth
- Persona: stable speaking style and personality
- Memory: hard constraints, cannot be contradicted
- Situation: immediate context right before the utterance

## Strict Requirements
1. Situation fidelity
- The query must naturally emerge from this situation and context.

2. Persona consistency
- Use a tone and phrasing style that sounds like this user.
- Keep language natural and authentic.

3. Memory alignment
- No conflict with relevant memory facts.

4. EQ relevance
- The query should fit one of: Emotional Support / High-EQ Interaction / Social Strategy.

5. Natural expression (critical)
- Express naturally what the user would think/feel/say in this moment.
- Do not over-explain or dump all context.
- Use implicit references when understandable.
- One clear thought or utterance; avoid listing multiple ideas.

## Output Format
Return ONLY one JSON object:
```json
{
  "category": "Emotional Support | High-EQ Interaction | Social Strategy",
  "situation_id": "Sit_1",
  "query": "...",
  "relevant_mem": [1, 3],
  "reasoning": "One sentence: why this expression naturally emerges from persona, memory, and situation."
}
```

## Quick Self-Check Before Output (silent)
- Would a real person naturally say this in this moment?
- Does it fit the persona and situation?
- Is it concise and authentic?

## Input
User persona:
{user_persona}

Relevant memory facts:
{relevant_memory}

Recent dialogue context (for language alignment):
{dialogue_context}

Situation:
{situation}
'''


def build_persona_prompt(memory_info: str, user_dialogue: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PERSONA_USER_PROMPT_TEMPLATE
            .replace("{memory_info}", memory_info or "N/A")
            .replace("{user_dialogue}", user_dialogue or "N/A"),
        },
    ]


def build_situation_prompt(memory_info: str, user_persona: str, topics: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SITUATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SITUATION_USER_PROMPT_TEMPLATE
            .replace("{memory_info}", memory_info or "N/A")
            .replace("{user_persona}", user_persona or "N/A")
            .replace("{topics}", json.dumps(topics or [], ensure_ascii=False)),
        },
    ]


def build_generation_prompt(
    user_persona: str,
    relevant_memory: str,
    situation: dict[str, Any],
    dialogue_context: str = "",
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": QUERY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": QUERY_USER_PROMPT_TEMPLATE
            .replace("{user_persona}", user_persona or "N/A")
            .replace("{relevant_memory}", relevant_memory or "N/A")
            .replace("{situation}", json.dumps(situation or {}, ensure_ascii=False, indent=2))
            .replace("{dialogue_context}", dialogue_context or "N/A"),
        },
    ]
