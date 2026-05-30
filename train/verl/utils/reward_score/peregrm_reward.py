import os
import json
import re
import asyncio
from typing import Dict, Any, List
from openai import AsyncOpenAI

STANDBY_COEFF = 0.5

_semaphore = asyncio.Semaphore(64)

train_client = AsyncOpenAI(
    api_key=os.environ.get("JUDGE_API_KEY"),
    base_url=os.environ.get("JUDGE_BASE_URL"),
)

eval_client = AsyncOpenAI(
    api_key=os.environ.get("EVAL_JUDGE_API_KEY"),
    base_url=os.environ.get("EVAL_JUDGE_BASE_URL"),
)

STANDBY_COEFF = float(os.environ.get("STANDBY_COEFF", "0.5"))
MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
EVAL_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "deepseek-v4-flash")
GRM_TRIAL = max(1, int(os.environ.get("GRM_TRIAL", "1")))

# Prompt templates for the generative reward model (GRM). During training the
# reward is judged dynamically; during evaluation fixed criteria can be supplied
# through the parquet extra_info field.

resonation_criteria = '''Measures the depth and accuracy of the responder's ability to enter the user's Internal Frame of Reference, with special emphasis on personality detection. It evaluates whether the responder captures:
1. The explicit emotion and content.
2. The causal link to the user's stable personality traits (from memory).
3. The deeper psychological need behind the reaction.'''

expression_criteria = '''Measures the quality, tone, and effectiveness of the responder's communication, with a special focus on **personalized strategy adaptation**. It evaluates whether the response demonstrates a communication strategy that is appropriately tailored to the user's personality traits and psychological needs derived from memory.'''

reception_criteria = '''Measures the interaction strictly from the **specific user's perspective**, taking into account the user's personality traits and psychological needs (derived from `memory` and `persona`). It evaluates whether the responder identified and addressed the user's Hidden Intention (the unspoken need) in a way that feels warm, safe, and supportive **for this particular user**.

* **The Key Question:** Given this user's unique personality, did the responder hit the "bullseye" of the hidden need without being intrusive? Does the response make **this user** feel supported and genuinely eager to continue the conversation?

**Instruction to the Judge:** 

1. **Safety Check:** Does this response feel warm and respectful to someone with this personality, or does it feel creepy, dismissive, or overly intrusive? (Intrusiveness = Low Score).
2. **Need Check:** Did the responder address what you (as this user) truly needed (e.g., validation, autonomy, practical advice), or did they just respond to your surface words?
3. **Engagement Check:** Based on your personality, does this response make you feel a genuine desire to reply and share more?'''

base_prompt_response_dynamic = r'''
You are an expert Psychologist and Empathy Evaluator. Your goal is to provide objective, critical, and nuanced assessments of a response quality.
You will be provided with a conversation context involving a specific User Persona and a Scenario.
Your task is to evaluate the Assistant's Response based on the **{dimension}** dimension.
Before scoring, you should first derive specific criteria for this exact user, scenario, query, and dimension. The criteria should be grounded in general emotional intelligence while using memory/persona as evidence for what would make the response emotionally effective for this user.
For example, a score of 1 means the response does not meet the criteria at all, a score of 3 means the response meets only some parts, and a score of 5 means the response perfectly meets the evaluation criteria.
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

---

### 1. Dimension Definition
**Dimension:** {dimension}
**General Criteria:** {criteria}

---

### 2. Task Context
**Memories extracted from previous dialogue:**
{memory}

**Scenario:**
{scenario}

**User Persona:**
{persona}

**User Query:**
{query}

**Assistant Response:**
{response}

---

### 3. Output Requirement
Output with three lines
Specific Criteria: <Derive concrete evaluation criteria for this exact task and user>.
Analysis: <Analyze the response based on the Specific Criteria>.
Scores: <the overall comprehensive score of the response, e.g., \boxed{{x}}>.
'''

base_prompt_response_fixed = r'''
You are an expert Psychologist and Empathy Evaluator. Your task is to judge the Assistant Response for the **{dimension}** dimension using the provided Fixed Criteria.

The Fixed Criteria were prepared in advance for this exact user, scenario, query, and dimension. Do not generate new criteria, revise the criteria, or reward anything merely because it seems generally good. Use the task context only to understand and apply the Fixed Criteria accurately.

Score strictly on a 1-5 scale:
- 1 means the response fails the Fixed Criteria or has serious emotional-intelligence problems.
- 3 means the response partially satisfies the Fixed Criteria but misses important requirements.
- 5 means the response perfectly satisfies the Fixed Criteria, including the high-weight and persona-informed requirements.

---

### 1. Fixed Evaluation Criteria
**Dimension:** {dimension}
{criteria}

---

### 2. Task Context
**Memories extracted from previous dialogue:**
{memory}

**Scenario:**
{scenario}

**User Persona:**
{persona}

**User Query:**
{query}

**Assistant Response:**
{response}

---

### 3. Output Requirement
Output exactly two lines:
Analysis: <Briefly analyze how well the response satisfies the Fixed Criteria, especially high-weight criteria and any major misses.>
Scores: <the overall comprehensive score of the response, e.g., \boxed{{x}}>.
'''

standby_user_prompt = '''You are a third-party dialogue evaluator. Your task is to evaluate the AI's response from a neutral, analytical perspective. You evaluate the response purely on content quality, reasoning, efficiency, objectivity, and factual precision.

**Context:**
**Dialogue History:**
User: {query}
AI: {response}

---

Evaluate the AI response according to the following five dimensions.
Each dimension is scored from 0 to 20.

Evaluation Criteria:

1. Content Effectiveness (0-20)
- Does the response directly address the user's concrete question or problem?
- Does it provide practical, meaningful, or insightful information?
- Does it avoid empty, repetitive, or purely generic statements?

2. Logical Structure & Clarity (0-20)
- Are the main points clear and well-organized?
- Is there coherent reasoning or explanation?
- Does it avoid logical jumps, vague claims, or unsupported assertions?

3. Communication Efficiency (0-20)
- Is the response concise and focused on the core issue?
- Does it avoid unnecessary verbosity, emotional padding, or digressions?
- Does it help move the conversation closer to resolution or clarity?

4. Objectivity & Neutrality (0-20)
- Is the response grounded in the given text rather than assumptions about emotions or intent?
- Does it avoid excessive subjectivity or value judgments?
- Does it maintain a rational, neutral tone?

5. Information Accuracy & Verifiability (0-20)
- Are the statements accurate and reasonable?
- Are claims specific, checkable, or logically justified?
- Are suggestions actionable when appropriate, and free from exaggeration?

Scoring Rules:
- Each dimension must be scored independently.
- Use the full range of scores when appropriate.
- High emotional expressiveness alone must NOT increase scores.
- Verbosity that does not add information should lower efficiency scores.

{{
    "Content Effectiveness": [0-20], 
    "Logical Structure & Clarity": [0-20], 
    "Communication Efficiency": [0-20], 
    "Objectivity & Neutrality": [0-20], 
    "Information Accuracy & Verifiability": [0-20]
}}'''

def get_default_criteria(dimension: str):
    assert dimension.lower() in ["resonation", "expression", "reception"]
    if dimension == "resonation":
        return resonation_criteria
    elif dimension == "expression":
        return expression_criteria
    return reception_criteria

def prepare_prompt_response(
    dimension: str,
    scenario: str,
    persona: str,
    query: str,
    response: str,
    memory: List[str],
    criteria: str | None = None,
    fixed_criteria: bool = False,
):
    """Build a judge prompt for one response-quality dimension."""
    assert dimension.lower() in ["resonation", "expression", "reception"]
    criteria = criteria or get_default_criteria(dimension)
    prompt_template = base_prompt_response_fixed if fixed_criteria else base_prompt_response_dynamic
    prompt = prompt_template.format(
        dimension=dimension,
        criteria=criteria,
        scenario=scenario,
        persona=persona,
        query=query,
        response=response, 
        memory="\n".join([f"{i+1}.{memory[i]}" for i in range(len(memory))])
    )

    return [{"role": "user", "content": prompt}]

def prepare_prompt_standby(scenario: str, persona: str, query: str, response: str):
    """Build the non-empathy standby judge prompt used as a penalty term."""
    prompt = standby_user_prompt.format(
        scenario=scenario,
        persona=persona,
        query=query,
        response=response
    )
    return [{"role": "user", "content": prompt}]

# Utils

def extract_boxed_score(text: str):
    """Parse a 1-5 boxed judge score and normalize it to 0-1."""
    if isinstance(text, Exception) or not isinstance(text, str):
        return 0.6
    match = re.search(r'\\boxed\{([\d.]+)\}', text)
    if match:
        num_str = match.group(1)
        return float(num_str) / 5.
    return 0.6

def calculate_standby_reward(output):
    """Parse the standby judge JSON and normalize its 0-100 score to 0-1."""
    if not output or not isinstance(output, str):
        return 0.8
    if "```" in output:
        output = output.split("```json")[-1].split("```")[0].strip().strip("\n").strip()
    try:
        res = json.loads(output, strict=False)
        score = sum([float(s) for k, s in res.items()])
        return float(score) / 100.
    except:
        return 0.8

async def get_api_response(prompt, client, model_name, max_retries=3):
    """Call an OpenAI-compatible judge endpoint with bounded retries."""
    async with _semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    temperature=0.3,
                )
                content = response.choices[0].message.content
                # print(response)
                return content.strip()
            except Exception as e:
                print(f"API Error (Trial {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    await asyncio.sleep(10)
        return ""

async def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict[str, Any]) -> float:
    """The scoring function of PERM, including four dimensions:
    
    ### Dims:
        *`Resonation`: Whether the analysis (thinking process) shows enough understanding of user's emotional state and hidden need.
        *`Expression`: Whether the response shows enough warmth, support.
        *`Reception`: Whether the response will make user feel nice.
        *`Standby`: Evaluate from the bystander's perspective to judge if the dialogue is safe, efficient ...
    """

    scenario = extra_info.get('scenario', '')
    persona = extra_info.get('persona', '')
    query = extra_info.get('query', '')
    memory = extra_info.get('memory', [''])
    fixed_criteria = extra_info.get('criteria', {}) or {}
    split = extra_info.get('split', 'train')

    if "<think>" in solution_str and "</think>" in solution_str:
        analysis_part = solution_str.split("<think>")[-1].split("</think>")[0].strip()
        response_part = solution_str.split("</think>")[-1].strip()
    else:
        analysis_part = solution_str
        response_part = solution_str

    prompts = {}
    grm_dimensions = ['resonation', 'expression', 'reception']
    if split == "train":
        for dim in grm_dimensions:
            for trial_idx in range(GRM_TRIAL):
                prompts[f"{dim}#{trial_idx}"] = prepare_prompt_response(
                    dim,
                    scenario,
                    persona,
                    query,
                    response_part,
                    memory,
                    fixed_criteria=False,
                )
        prompts['standby'] = prepare_prompt_standby(scenario, persona, query, response_part)
    else:
        for dim in grm_dimensions:
            prompts[dim] = prepare_prompt_response(
                dim,
                scenario,
                persona,
                query,
                response_part,
                memory,
                fixed_criteria.get(dim),
                fixed_criteria=True,
            )

    cur_model = MODEL if split == "train" else EVAL_MODEL
    cur_client = train_client if split == "train" else eval_client
    tasks = {dim: get_api_response(prompt, cur_client, cur_model) for dim, prompt in prompts.items()}
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    judge_score = {}
    trial_scores = {dim: [] for dim in grm_dimensions}
    for (dim, task_result) in zip(tasks.keys(), results):
        if dim == 'standby':
            judge_score[dim] = min(calculate_standby_reward(task_result), 1.)
        elif "#" in dim:
            base_dim = dim.split("#", 1)[0]
            trial_scores[base_dim].append(min(extract_boxed_score(task_result), 1.))
        else:
            judge_score[dim] = min(extract_boxed_score(task_result), 1.)

    for dim, scores in trial_scores.items():
        if scores:
            judge_score[dim] = sum(scores) / len(scores)
            if split == "train" and GRM_TRIAL > 1:
                judge_score[f"{dim}_trials"] = scores

    resonation = judge_score.get('resonation', 0.0)
    expression = judge_score.get('expression', 0.0)
    reception = judge_score.get('reception', 0.0)

    if resonation > 0 and expression > 0 and reception > 0:
        harmonic_mean = 3 / (1/resonation + 1/expression + 1/reception)
    else:
        harmonic_mean = 0.0

    standby = judge_score.get('standby', 0.0)
    if split == "train":
        score = harmonic_mean + STANDBY_COEFF * (standby - 1.0)
    else:
        score = harmonic_mean

    log_path = os.environ.get("LOG_PATH", "reward_log.jsonl")
    if split == "train":
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(judge_score, ensure_ascii=False) + "\n")

    return score
