import os
import json
import re
import asyncio
from typing import Dict, Any

from openai import AsyncOpenAI


_semaphore = asyncio.Semaphore(int(os.environ.get("JUDGE_CONCURRENCY", "8")))

train_client = AsyncOpenAI(
    api_key=os.environ.get("JUDGE_API_KEY"),
    base_url=os.environ.get("JUDGE_BASE_URL"),
)

eval_client = AsyncOpenAI(
    api_key=os.environ.get("EVAL_JUDGE_API_KEY"),
    base_url=os.environ.get("EVAL_JUDGE_BASE_URL"),
)

MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
EVAL_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "deepseek-v4-flash")


profile_system_prompt = """You are a strict personalization evaluator.
Your job is to judge whether an inferred user profile accurately captures the reference persona."""

profile_reward_prompt = """Given the reference persona and the model's inferred profile, evaluate how well the inferred profile matches the reference persona.

The inferred profile should be rewarded for:
1. Capturing stable user traits, preferences, needs, and communication style.
2. Inferring useful personalization signals from the memory and user query.
3. Avoiding unsupported, hallucinated, or overly generic claims.
4. Being specific enough to guide a personalized response.

Do not reward the response itself here. Only evaluate the inferred profile / analysis part.

---

### Reference Persona
{persona}

### Memory
{memory}

### User Query
{query}

### Model Inferred Profile / Analysis
{analysis}

---

Score from 0 to 1:
- 0 means the inferred profile is irrelevant, generic, or contradicts the persona.
- 0.5 means it captures some obvious traits but misses important personalization signals.
- 1 means it accurately and specifically captures the key persona traits and personalization needs.

Output format:
Reasoning: <brief reason>
Score: \\boxed{{x}}
"""


response_system_prompt = """You are a strict evaluator simulating the target user.
Your job is to judge whether the response feels personalized and worth continuing."""

response_reward_prompt = """You are the user described below. After reading the assistant's reply, decide whether you would want to continue the conversation.

### User Profile and Personality
{persona}

### Relevant Memory
{memory}

### Scenario
{scenario}

### User Message
{query}

### Assistant Reply
{response}

---

Evaluate the reply using these criteria:
1. Naturalness: Is it fluent, natural, and conversational?
2. Personalization: Does it fit this user's persona, memory, preferences, and emotional/practical needs?
3. Relevance: Does it directly respond to the user's message and scenario?
4. Continuation Value: Would this user feel interested, supported, or willing to keep talking?
5. Non-generic Quality: Does it avoid shallow praise, repetition, and template-like comfort?

Be strict. If the reply is generic, off-persona, irrelevant, intrusive, or not worth continuing, give 0.
If the reply is personalized enough that this user would likely continue the conversation, give 1.

Output format:
Reasoning: <brief reason>
Decision: \\boxed{{0 or 1}}
"""


def format_memory(memory) -> str:
    """Render memory lists into the numbered format used by judge prompts."""
    if isinstance(memory, list):
        return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(memory))
    return str(memory or "")


def split_solution(solution_str: str) -> tuple[str, str]:
    """Separate model analysis from final answer when <think> tags are present."""
    if "<think>" in solution_str and "</think>" in solution_str:
        analysis = solution_str.split("<think>", 1)[-1].split("</think>", 1)[0].strip()
        response = solution_str.split("</think>", 1)[-1].strip()
        return analysis, response
    return "", solution_str.strip()


def prepare_profile_prompt(persona: str, memory, query: str, analysis: str):
    """Judge whether the model inferred a useful personalized profile."""
    prompt = profile_reward_prompt.format(
        persona=persona,
        memory=format_memory(memory),
        query=query,
        analysis=analysis,
    )
    return [
        {"role": "system", "content": profile_system_prompt},
        {"role": "user", "content": prompt},
    ]


def prepare_response_prompt(scenario: str, persona: str, memory, query: str, response: str):
    """Judge whether the final response feels personalized to the target user."""
    prompt = response_reward_prompt.format(
        persona=persona,
        memory=format_memory(memory),
        scenario=scenario,
        query=query,
        response=response,
    )
    return [
        {"role": "system", "content": response_system_prompt},
        {"role": "user", "content": prompt},
    ]


def extract_boxed_number(output, default: float = 0.0) -> float:
    """Parse boxed binary/float scores from judge output."""
    if isinstance(output, Exception) or not output:
        return default
    text = str(output)
    match = re.search(r"\\boxed\{?\s*([01](?:\.\d+)?)\s*\}?", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(?:Score|Decision)\s*:\s*([01](?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default


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
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                print(f"API Error (Trial {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(2 ** attempt if attempt < max_retries - 1 else 10)
        return ""


async def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict[str, Any]) -> float:
    """RLPA-style personalization reward.

    The baseline separately evaluates:
    1. The analysis / inferred profile part.
    2. The final personalized response part.
    """

    scenario = extra_info.get("scenario", "")
    persona = extra_info.get("persona", "")
    query = extra_info.get("query", "")
    memory = extra_info.get("memory", [""])
    split = extra_info.get("split", "train")

    analysis_part, response_part = split_solution(solution_str)

    prompts = {
        "profile": prepare_profile_prompt(persona, memory, query, analysis_part),
        "response": prepare_response_prompt(scenario, persona, memory, query, response_part),
    }

    cur_model = MODEL if split == "train" else EVAL_MODEL
    cur_client = train_client if split == "train" else eval_client
    tasks = {name: get_api_response(prompt, cur_client, cur_model) for name, prompt in prompts.items()}
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    raw_outputs = dict(zip(tasks.keys(), results))
    profile_score = max(0.0, min(extract_boxed_number(raw_outputs.get("profile"), default=0.0), 1.0))
    response_score = max(0.0, min(extract_boxed_number(raw_outputs.get("response"), default=0.0), 1.0))

    score = profile_score + response_score

    return score
