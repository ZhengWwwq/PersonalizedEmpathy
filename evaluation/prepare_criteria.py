from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


resonation_criteria = '''Generates criteria for evaluating the responder's emotional attunement and empathic understanding.

The primary focus is the responder's foundational emotional intelligence: accurately recognizing the user's explicit emotion, implied emotional state, situational pressure, and underlying psychological need. Personalization should extend this foundation by using memory and persona only when they help explain the user's emotional pattern or make the empathic understanding more precise.

The generated criteria should cover:
1. Basic emotional recognition: whether the responder identifies the user's visible emotion, emotional intensity, and relevant situational trigger.
2. Empathic inference: whether the responder understands the deeper concern, unmet need, conflict, or vulnerability beneath the surface message.
3. Contextual grounding: whether the responder's interpretation is supported by the scenario and query instead of over-reading or inventing motives.
4. Personalized resonance: whether memory/persona are used to refine the understanding of this specific user without reducing them to a fixed trait label.
5. Respectful uncertainty: whether the responder leaves room for ambiguity when the user's internal state is not fully knowable.'''

expression_criteria = '''Generates criteria for evaluating the responder's emotionally intelligent communication.

The primary focus is the responder's foundational ability to express empathy in a clear, warm, respectful, and useful way. Personalization should extend this foundation by adapting tone, framing, level of directness, and support strategy to the user's known preferences, needs, and context.

The generated criteria should cover:
1. Emotional tone: whether the response sounds warm, sincere, calm, and appropriately matched to the user's emotional state.
2. Validation and support: whether it acknowledges the user's feelings and situation before shifting into advice, reframing, or problem solving.
3. Clarity and usefulness: whether the response is coherent, concrete, and helpful for the user's immediate situation.
4. Boundary and respect: whether it avoids judgment, pressure, exaggeration, false certainty, or intrusive claims about the user's inner world.
5. Personalized adaptation: whether communication style and support strategy are adjusted to memory/persona when relevant, without making the response feel scripted or overly familiar.'''

reception_criteria = '''Generates criteria for evaluating how the response is likely to be received by this particular user.

The primary focus is the user's likely felt experience of the response: whether it would feel emotionally safe, understood, respectful, and worth continuing. Personalization should extend this foundation by considering how this user's persona and memory may shape what feels supportive, intrusive, motivating, or discouraging.

The generated criteria should cover:
1. Felt understanding: whether the user would likely feel that their emotion, situation, and concern were genuinely understood.
2. Emotional safety: whether the response would likely feel respectful, nonjudgmental, non-intrusive, and safe to receive.
3. Need satisfaction: whether it addresses the user's likely immediate need, such as validation, reassurance, autonomy, perspective, encouragement, or practical direction.
4. Engagement potential: whether it would make the user more willing to continue, clarify, reflect, or share more.
5. Personalized fit: whether the response would feel appropriate for this specific user's personality, history, sensitivities, and preferences, while still meeting general standards of emotional intelligence.'''

base_prompt_criteria = r'''
You are an expert Psychologist and Empathy Evaluation Designer. Your goal is to generate objective, critical, and context-specific evaluation criteria for emotionally intelligent assistant responses.
You will be provided with a conversation context involving a specific User Persona and a Scenario.
Your task is to generate **Specific Criteria** for the **{dimension}** dimension evaluation.

The criteria must be derived from the joint meaning of the current query, scenario, persona, and memory. Start from the user's immediate emotional and practical needs in the query, then use persona and memory to infer deeper needs, sensitivities, preferred support style, and likely reception.

Foundational emotional intelligence is the baseline: emotional recognition, empathy, respect, helpfulness, and emotional safety should always matter. Personalization is the advanced layer: persona and memory should shape what "emotionally intelligent" means for this specific user in this specific moment.

Avoid criteria that reward mentioning or reusing memory by itself. The key question is not whether the responder recalls a specific fact from memory, but whether the responder uses persona/memory to understand the user more accurately, communicate more appropriately, and create a safer, warmer, more supportive interaction without becoming intrusive, presumptive, or reductive.

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

---

### 3. Output Requirement
Output only the following section:

Specific Criteria:
1. <criterion name> (<weight>%): <context-specific description of what a strong response should demonstrate>
2. <criterion name> (<weight>%): <context-specific description of what a strong response should demonstrate>
3. <criterion name> (<weight>%): <context-specific description of what a strong response should demonstrate>
...

Requirements:
- The weights must sum to 100%.
- Include 4 to 6 criteria.
- Do not create criteria that directly reward explicit memory callback, such as whether the response mentions a remembered fact.
- Treat personalization as a higher-resolution form of emotional intelligence, not as a separate goal competing with emotional intelligence.
- Make each criterion specific to the combined context of query, scenario, persona, and memory.
- Include both baseline criteria for generally emotionally intelligent responses to this query and higher-level criteria for persona/memory-informed emotional fit.
'''


DIMENSION_CRITERIA = {
    "resonation": resonation_criteria,
    "expression": expression_criteria,
    "reception": reception_criteria,
}


def _memory_value(memory_item: Any) -> str:
    """Return the public memory value from either raw strings or memory dicts."""
    if isinstance(memory_item, dict):
        return str(memory_item.get("value", ""))
    return str(memory_item)


def format_memory(memory: list[Any]) -> str:
    """Render extracted memories into the numbered text block used by prompts."""
    if not memory:
        return "No memory is provided."

    return "\n".join(f"{i}. {_memory_value(item)}" for i, item in enumerate(memory, start=1))


def extract_persona(persona: Any) -> str:
    """Normalize persona objects into a prompt-ready text profile."""
    if isinstance(persona, dict):
        return persona.get("persona_profile") or json.dumps(persona, ensure_ascii=False, indent=2)
    return str(persona or "")


def extract_scenario(query_item: dict[str, Any]) -> str:
    """Normalize the query situation field into prompt-ready text."""
    situation = query_item.get("situation", "")
    if isinstance(situation, dict):
        return situation.get("situation") or json.dumps(situation, ensure_ascii=False, indent=2)
    return str(situation or "")


def prepare_prompt_criteria(
    dimension: str,
    scenario: str,
    persona: str,
    query: str,
    memory: list[Any] | str,
) -> list[dict[str, str]]:
    """Build one prompt for generating fixed evaluation criteria."""
    dimension = dimension.lower()
    if dimension not in DIMENSION_CRITERIA:
        raise ValueError(f"dimension must be one of {', '.join(DIMENSION_CRITERIA)}")

    memory_text = memory if isinstance(memory, str) else format_memory(memory)
    prompt = base_prompt_criteria.format(
        dimension=dimension,
        criteria=DIMENSION_CRITERIA[dimension],
        scenario=scenario,
        persona=persona,
        query=query,
        memory=memory_text,
    )
    return [{"role": "user", "content": prompt}]


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file from a pathlib path."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(path: Path, data: Any) -> None:
    """Write JSON through a temporary file to avoid partial outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def split_env_list(name: str, default: str | None = None) -> list[str]:
    """Read comma-separated environment variables."""
    raw = os.getenv(name, default or "")
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values


def init_output_data(
    data: list[dict[str, Any]],
    existing_output: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create or resume the sidecar criteria output structure."""
    output_data = existing_output or []
    for session_index, session in enumerate(data):
        session_id = str(session.get("session_id") or session.get("original_sid") or "")
        queries = session.get("queries", [])

        if session_index >= len(output_data):
            output_data.append({"session_id": session_id, "criterias": []})

        output_data[session_index].setdefault("session_id", session_id)
        output_data[session_index].setdefault("criterias", [])
        criterias = output_data[session_index]["criterias"]

        for query_index, query_item in enumerate(queries):
            query_id = str(query_item.get("query_id") or "")
            if query_index >= len(criterias):
                criterias.append({"query_id": query_id})
            else:
                criterias[query_index].setdefault("query_id", query_id)

    return output_data


def build_generation_jobs(
    data: list[dict[str, Any]],
    output_data: list[dict[str, Any]],
    dimensions: list[str],
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Create pending LLM jobs for criteria that are missing or overwritten."""
    jobs = []
    for session_index, session in enumerate(data):
        persona = extract_persona(session.get("persona"))
        memory = session.get("extracted_memory") or []
        for query_index, query_item in enumerate(session.get("queries", [])):
            query_id = str(query_item.get("query_id") or "")
            existing = output_data[session_index]["criterias"][query_index]
            for dimension in dimensions:
                if not overwrite and existing.get(dimension):
                    continue
                prompt = prepare_prompt_criteria(
                    dimension=dimension,
                    scenario=extract_scenario(query_item),
                    persona=persona,
                    query=str(query_item.get("query", "")),
                    memory=memory,
                )
                jobs.append(
                    {
                        "session_index": session_index,
                        "query_index": query_index,
                        "query_id": query_id,
                        "dimension": dimension,
                        "prompt": prompt,
                    }
                )
    return jobs


def run_generation(
    output_data: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    output_path: Path,
    model: str,
    api_keys: list[str],
    base_urls: list[str],
    concurrency: int,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> None:
    """Generate criteria with an OpenAI-compatible endpoint and save results."""
    if not jobs:
        save_json_atomic(output_path, output_data)
        print("No pending criteria generation jobs.")
        return

    if not api_keys:
        raise RuntimeError("No API key found. Set OPENAI_API_KEY, or pass --api-key.")
    if not base_urls:
        base_urls = [os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")]
    if len(base_urls) == 1 and len(api_keys) > 1:
        base_urls = base_urls * len(api_keys)
    if len(base_urls) != len(api_keys):
        raise RuntimeError("The number of base URLs must be 1 or equal to the number of API keys.")

    from api_call import api_call

    results = api_call(
        model_name=model,
        user_prompt_list=[job["prompt"] for job in jobs],
        api_list=api_keys,
        base_url_list=base_urls,
        api_call_limit=concurrency,
        max_retry=max_retries,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )

    for job, result in zip(jobs, results):
        output_data[job["session_index"]]["criterias"][job["query_index"]][job["dimension"]] = result

    save_json_atomic(output_path, output_data)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for criteria generation."""
    parser = argparse.ArgumentParser(description="Generate context-specific EQ criteria with an async LLM pipeline.")
    parser.add_argument("--input", type=Path, default=Path("dataset/randomsplit/test.json"))
    parser.add_argument("--output", type=Path, default=Path("dataset/randomsplit/test_criteria.json"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--api-key", action="append", default=None, help="API key. Can be passed multiple times. Defaults to OPENAI_API_KEY comma-separated values.")
    parser.add_argument("--base-url", action="append", default=None, help="Base URL. Can be passed multiple times. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("API_CONCURRENCY", "64")))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N pending jobs for a smoke test.")
    parser.add_argument("--dimensions", nargs="+", default=list(DIMENSION_CRITERIA), choices=list(DIMENSION_CRITERIA))
    parser.add_argument("--overwrite", action="store_true", help="Regenerate criteria even when they already exist.")
    parser.add_argument("--resume", action="store_true", help="Resume from --output if it exists.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for sidecar criteria generation."""
    args = parse_args()
    data = load_json(args.input)

    if not isinstance(data, list):
        raise TypeError("Input data must be a JSON list of sessions.")

    existing_output = load_json(args.output) if args.resume and args.output.exists() else None
    if existing_output is not None and not isinstance(existing_output, list):
        raise TypeError("Output data must be a JSON list when using --resume.")

    api_keys = args.api_key or split_env_list("OPENAI_API_KEY")
    base_urls = args.base_url or split_env_list("OPENAI_BASE_URL")
    output_data = init_output_data(data, existing_output)
    jobs = build_generation_jobs(
        data=data,
        output_data=output_data,
        dimensions=args.dimensions,
        overwrite=args.overwrite,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]

    total_queries = sum(len(session.get("queries", [])) for session in data)
    print(f"Loaded {len(data)} sessions, {total_queries} queries.")
    print(f"Pending LLM jobs: {len(jobs)} ({', '.join(args.dimensions)}).")
    print(f"Output: {args.output}")

    run_generation(
        output_data=output_data,
        jobs=jobs,
        output_path=args.output,
        model=args.model,
        api_keys=api_keys,
        base_urls=base_urls,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
