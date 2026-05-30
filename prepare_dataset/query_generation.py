from api_call import api_call
import json
from typing import Any
from query_prompt import build_generation_prompt, build_persona_prompt, build_situation_prompt


DEFAULT_MODEL = "deepseek-chat"


def _extract_session_text(
    record: dict[str, Any],
    max_turns: int | None = None,
    recent_turns: int = 3,
    only_user: bool = False,
) -> str:
    sessions = record.get("sessions", [])
    if not sessions:
        return ""

    all_turns = sessions[0].get("turns", [])
    if not isinstance(all_turns, list):
        return ""

    turn_num = len(all_turns) if max_turns is None else min(max_turns, len(all_turns))
    window = max(0, recent_turns)
    selected_turns = all_turns[:turn_num][-window:] if window else []

    lines = []
    for turn in selected_turns:
        role = turn.get("role", "unknown")
        if only_user and str(role).lower() != "user":
            continue
        text = str(turn.get("text", "")).strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _build_persona_dialogue(
    record: dict[str, Any],
    max_chars: int = 5000,
    recent_turns: int = 12,
) -> str:
    full_dialogue = _extract_session_text(record, recent_turns=recent_turns, only_user=False)
    if len(full_dialogue) <= max_chars:
        return full_dialogue

    user_only_dialogue = _extract_session_text(record, recent_turns=recent_turns, only_user=True)
    if len(user_only_dialogue) <= max_chars:
        return user_only_dialogue

    return user_only_dialogue[-max_chars:]


def _extract_memory_items(record: dict[str, Any], max_items: int | None = None) -> list[dict[str, Any]]:
    memory_items = record.get("memory_items", [])
    if not isinstance(memory_items, list):
        return []
    if max_items is None:
        return memory_items
    return memory_items[: max(0, max_items)]


def _extract_memory_text(record: dict[str, Any], max_items: int | None = None) -> str:
    lines = []
    for mem_position, mem in enumerate(_extract_memory_items(record, max_items=max_items), start=1):
        label = mem.get("label", "")
        value = str(mem.get("value", "")).strip()
        suggestion = mem.get("label_suggestion", "")
        if value:
            lines.append(
                f"mem_position={mem_position}; label={label}; possible type={suggestion}; value={value}"
            )
    return "\n".join(lines)


def _extract_relevant_memory_text(record: dict[str, Any], relevant_mem: Any) -> tuple[list[int], str]:
    positions: list[int] = []
    if isinstance(relevant_mem, list):
        raw_positions = relevant_mem
    else:
        raw_positions = []
        text = str(relevant_mem).strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    raw_positions = parsed
            except Exception:
                raw_positions = []

    for pos in raw_positions:
        try:
            p = int(pos)
            if p > 0 and p not in positions:
                positions.append(p)
        except (TypeError, ValueError):
            continue

    memory_items = _extract_memory_items(record)
    if not positions:
        return [], _extract_memory_text(record)

    lines = []
    for mem_position in positions:
        idx = mem_position - 1
        if idx < 0 or idx >= len(memory_items):
            continue
        mem = memory_items[idx]
        label = mem.get("label", "")
        value = str(mem.get("value", "")).strip()
        suggestion = mem.get("label_suggestion", "")
        if value:
            lines.append(
                f"mem_position={mem_position}; label={label}; possible type={suggestion}; value={value}"
            )

    if not lines:
        return positions, _extract_memory_text(record)
    return positions, "\n".join(lines)


def prepare_persona_prompt(record: dict[str, Any]) -> list[dict[str, str]]:
    session_text = _build_persona_dialogue(record)
    memory_text = _extract_memory_text(record)
    return build_persona_prompt(memory_text, session_text)


def _parse_plain_response(response: str | None) -> str | None:
    if not response:
        return None
    text = response.strip()
    if not text:
        return None
    if "```" not in text:
        return text

    try:
        if "```json" in text:
            extracted = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```text" in text:
            extracted = text.split("```text", 1)[1].split("```", 1)[0].strip()
        elif "```markdown" in text:
            extracted = text.split("```markdown", 1)[1].split("```", 1)[0].strip()
        else:
            extracted = text.split("```", 1)[1].split("```", 1)[0].strip()
    except Exception:
        extracted = text

    # Some models output a language tag as first line inside fenced content.
    lines = extracted.splitlines()
    if lines and lines[0].strip().lower() in {"json", "text", "markdown", "md"}:
        extracted = "\n".join(lines[1:]).strip()

    return extracted


def _parse_json_obj(response: str | None) -> dict[str, Any] | None:
    text = _parse_plain_response(response)
    if not text:
        return None
    try:
        result = json.loads(text, strict=False)
        return result if isinstance(result, dict) else None
    except Exception:
        pass

    try:
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        result = json.loads(text, strict=False)
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def parse_response(response: str | None) -> list[dict[str, Any]] | None:
    if not response:
        return None

    text = _parse_plain_response(response)
    if not text:
        return None

    try:
        result = json.loads(text, strict=False)
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)]
        if isinstance(result, dict):
            return [result]
        return None
    except Exception:
        pass

    try:
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        result = json.loads(text, strict=False)
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)]
        if isinstance(result, dict):
            return [result]
        return None
    except Exception:
        return None


def _merge_usage(*usages: dict[str, Any] | None) -> dict[str, Any] | None:
    valid = [u for u in usages if isinstance(u, dict)]
    if not valid:
        return None

    def _to_int(value: Any) -> int:
        try:
            if value is None:
                return 0
            return int(value)
        except (TypeError, ValueError):
            return 0

    return {
        "input_tokens": sum(_to_int(u.get("input_tokens")) for u in valid),
        "output_tokens": sum(_to_int(u.get("output_tokens")) for u in valid),
        "total_tokens": sum(_to_int(u.get("total_tokens")) for u in valid),
    }


def _extract_persona_and_topics(persona_obj: dict[str, Any] | None) -> tuple[bool, str, list[str]]:
    if not isinstance(persona_obj, dict):
        return False, "", []

    is_deep_raw = persona_obj.get("is_deep_persona", False)
    if isinstance(is_deep_raw, bool):
        is_deep = is_deep_raw
    elif isinstance(is_deep_raw, str):
        is_deep = is_deep_raw.strip().lower() in {"true", "1", "yes", "y"}
    else:
        is_deep = bool(is_deep_raw)

    persona_profile = str(persona_obj.get("persona_profile", "")).strip()
    topics_raw = persona_obj.get("topics", [])

    if isinstance(topics_raw, list):
        topics = [str(t).strip() for t in topics_raw if str(t).strip()]
    elif isinstance(topics_raw, str):
        try:
            parsed_topics = json.loads(topics_raw)
            if isinstance(parsed_topics, list):
                topics = [str(t).strip() for t in parsed_topics if str(t).strip()]
            else:
                topics = [topics_raw.strip()] if topics_raw.strip() else []
        except Exception:
            topics = [topics_raw.strip()] if topics_raw.strip() else []
    else:
        topics = []

    return is_deep, persona_profile, topics


def preview_situation_prompt_effect(
    record: dict[str, Any],
    model_name: str = DEFAULT_MODEL,
    api_list: list[str] | None = None,
    base_url_list: list[str] | None = None,
    api_call_limit: int = 20,
    max_retry: int = 3,
) -> dict[str, Any]:
    if not api_list:
        raise ValueError("api_list is required")
    if not base_url_list:
        raise ValueError("base_url_list is required")

    persona_prompt = prepare_persona_prompt(record)
    # Stage 1 call: infer whether the record has enough persona depth.
    persona_responses = api_call(
        model_name=model_name,
        user_prompt_list=[persona_prompt],
        api_list=api_list,
        base_url_list=base_url_list,
        api_call_limit=api_call_limit,
        max_retry=max_retry,
        max_completion_tokens=2048,
        return_usage=True,
    )

    persona_obj: dict[str, Any] | None = None
    persona_usage: dict[str, Any] | None = None
    if persona_responses and isinstance(persona_responses[0], dict):
        content = persona_responses[0].get("content")
        persona_obj = _parse_json_obj(content if isinstance(content, str) else None)
        usage = persona_responses[0].get("usage")
        persona_usage = usage if isinstance(usage, dict) else None

    is_deep_persona, persona_profile, topics = _extract_persona_and_topics(persona_obj)

    result: dict[str, Any] = {
        "persona": persona_obj,
        "persona_usage": persona_usage,
        "situations": [],
        "situation_usage": None,
        "skipped": False,
        "skip_reason": "",
    }

    if not is_deep_persona:
        result["skipped"] = True
        result["skip_reason"] = "persona gate not passed"
        return result

    if not persona_profile:
        result["skipped"] = True
        result["skip_reason"] = "persona_profile is empty"
        return result

    memory_text = _extract_memory_text(record)
    situation_prompt = build_situation_prompt(memory_text, persona_profile, topics)
    # Stage 2 call: generate candidate situations for the accepted persona.
    situation_responses = api_call(
        model_name=model_name,
        user_prompt_list=[situation_prompt],
        api_list=api_list,
        base_url_list=base_url_list,
        api_call_limit=api_call_limit,
        max_retry=max_retry,
        max_completion_tokens=2048,
        return_usage=True,
    )

    if situation_responses and isinstance(situation_responses[0], dict):
        content = situation_responses[0].get("content")
        result["situations"] = parse_response(content if isinstance(content, str) else None) or []
        usage = situation_responses[0].get("usage")
        result["situation_usage"] = usage if isinstance(usage, dict) else None

    return result


def generate_batch(
    records: list[dict[str, Any]],
    model_name: str,
    api_list: list[str],
    base_url_list: list[str],
    api_call_limit: int = 20,
    max_retry: int = 3,
    return_usage: bool = False,
    return_stage_debug: bool = False,
) -> list[Any]:
    """
    Pipeline with batched parallel calls by stage:
    1) persona for all records (parallel in one api_call)
    2) situations for valid personas (parallel in one api_call)
    3) query per situation (flattened and parallel in one api_call)

    Empty persona/situations/query are preserved as empty outputs per record.
    """
    n = len(records)
    if n == 0:
        return []

    # Keep final per-record outputs; default empty.
    queries_per_record: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    usage_parts: list[list[dict[str, Any] | None]] = [[] for _ in range(n)]
    stage_debug_per_record: list[dict[str, Any]] = [
        {
            "persona": {},
            "situation": {},
            "query_tasks": [],
        }
        for _ in range(n)
    ]

    # Stage 1: persona (parallel).
    persona_prompts = [prepare_persona_prompt(record) for record in records]
    # Stage 1 call: batch persona extraction.
    persona_responses = api_call(
        model_name=model_name,
        user_prompt_list=persona_prompts,
        api_list=api_list,
        base_url_list=base_url_list,
        api_call_limit=api_call_limit,
        max_retry=max_retry,
        max_completion_tokens=2048,
        return_usage=return_usage,
    )

    persona_profiles: list[str] = [""] * n
    persona_topics: list[list[str]] = [[] for _ in range(n)]
    persona_pass: list[bool] = [False] * n

    for i, resp in enumerate(persona_responses):
        content: str | None = None
        usage: dict[str, Any] | None = None
        if return_usage and isinstance(resp, dict):
            content = resp.get("content") if isinstance(resp.get("content"), str) else None
            usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else None
        elif isinstance(resp, str):
            content = resp

        persona_obj = _parse_json_obj(content)
        is_deep, persona_profile, topics = _extract_persona_and_topics(persona_obj)
        persona_profiles[i] = persona_profile
        persona_topics[i] = topics
        persona_pass[i] = bool(is_deep and persona_profile)
        usage_parts[i].append(usage)
        stage_debug_per_record[i]["persona"] = {
            "prompt": persona_prompts[i],
            "raw_content": content,
            "parsed": persona_obj,
            "usage": usage,
            "is_deep_persona": is_deep,
            "persona_profile": persona_profile,
            "topics": topics,
            "passed": persona_pass[i],
        }
        stage_debug_per_record[i]["situation"] = {
            "ran": persona_pass[i],
            "skip_reason": "" if persona_pass[i] else "persona gate not passed",
        }

    # Stage 2: situations (parallel for persona_pass).
    situation_target_indices = [i for i in range(n) if persona_pass[i]]
    situations_per_record: list[list[dict[str, Any]]] = [[] for _ in range(n)]

    if situation_target_indices:
        situation_prompts = [
            build_situation_prompt(
                _extract_memory_text(records[i]),
                persona_profiles[i],
                persona_topics[i],
            )
            for i in situation_target_indices
        ]
        # Stage 2 call: batch situation generation only for accepted personas.
        situation_responses = api_call(
            model_name=model_name,
            user_prompt_list=situation_prompts,
            api_list=api_list,
            base_url_list=base_url_list,
            api_call_limit=api_call_limit,
            max_retry=max_retry,
            max_completion_tokens=2048,
            return_usage=return_usage,
        )

        for j, resp in enumerate(situation_responses):
            rec_idx = situation_target_indices[j]
            content: str | None = None
            usage: dict[str, Any] | None = None
            if return_usage and isinstance(resp, dict):
                content = resp.get("content") if isinstance(resp.get("content"), str) else None
                usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else None
            elif isinstance(resp, str):
                content = resp

            situations_per_record[rec_idx] = parse_response(content) or []
            usage_parts[rec_idx].append(usage)
            stage_debug_per_record[rec_idx]["situation"] = {
                "ran": True,
                "skip_reason": "",
                "prompt": situation_prompts[j],
                "raw_content": content,
                "parsed": situations_per_record[rec_idx],
                "usage": usage,
            }

    # Stage 3: queries (parallel across all situations).
    task_to_record: list[int] = []
    task_to_query_slot: list[int] = []
    query_prompts: list[list[dict[str, str]]] = []
    task_meta: list[dict[str, Any]] = []

    for rec_idx, situations in enumerate(situations_per_record):
        if not situations:
            continue
        record = records[rec_idx]
        persona_profile = persona_profiles[rec_idx]
        recent_dialogue = _build_persona_dialogue(record, max_chars=2000, recent_turns=5)
        for situation in situations:
            if not isinstance(situation, dict):
                continue
            relevant_positions, relevant_memory_text = _extract_relevant_memory_text(
                record,
                situation.get("relevant_mem", []),
            )
            query_prompts.append(
                build_generation_prompt(
                    user_persona=persona_profile,
                    relevant_memory=relevant_memory_text,
                    situation=situation,
                    dialogue_context=recent_dialogue,
                )
            )
            task_to_record.append(rec_idx)
            task_meta.append(
                {
                    "situation_id": str(situation.get("id", "")).strip() or "Sit",
                    "category": str(situation.get("category", "")).strip(),
                    "relevant_mem": relevant_positions,
                    "situation": situation,
                }
            )
            query_slot = len(stage_debug_per_record[rec_idx]["query_tasks"])
            task_to_query_slot.append(query_slot)
            stage_debug_per_record[rec_idx]["query_tasks"].append(
                {
                    "task_index": len(query_prompts) - 1,
                    "situation": situation,
                    "query_prompt": query_prompts[-1],
                    "meta": task_meta[-1],
                    "raw_content": None,
                    "parsed": None,
                    "usage": None,
                }
            )

    if query_prompts:
        # Stage 3 call: batch final query generation across all situations.
        query_responses = api_call(
            model_name=model_name,
            user_prompt_list=query_prompts,
            api_list=api_list,
            base_url_list=base_url_list,
            api_call_limit=api_call_limit,
            max_retry=max_retry,
            max_completion_tokens=2048,
            return_usage=return_usage,
        )

        for k, resp in enumerate(query_responses):
            rec_idx = task_to_record[k]
            meta = task_meta[k]
            query_slot = task_to_query_slot[k]

            content: str | None = None
            usage: dict[str, Any] | None = None
            if return_usage and isinstance(resp, dict):
                content = resp.get("content") if isinstance(resp.get("content"), str) else None
                usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else None
            elif isinstance(resp, str):
                content = resp

            parsed = parse_response(content)
            if parsed:
                one_query = parsed[0] if isinstance(parsed[0], dict) else None
                if one_query:
                    if not one_query.get("situation_id"):
                        one_query["situation_id"] = meta["situation_id"]
                    if not one_query.get("category") and meta["category"]:
                        one_query["category"] = meta["category"]
                    if not one_query.get("relevant_mem"):
                        one_query["relevant_mem"] = meta["relevant_mem"]
                    if not isinstance(one_query.get("situation"), dict):
                        one_query["situation"] = meta["situation"]
                    if not one_query.get("dialogue_context"):
                        one_query["dialogue_context"] = _build_persona_dialogue(
                            records[rec_idx], max_chars=2000, recent_turns=5
                        )
                    queries_per_record[rec_idx].append(one_query)

            task = stage_debug_per_record[rec_idx]["query_tasks"][query_slot]
            task["raw_content"] = content
            task["parsed"] = parsed
            task["usage"] = usage

            usage_parts[rec_idx].append(usage)

    for i in range(n):
        stage_debug_per_record[i]["result"] = {
            "query_count": len(queries_per_record[i]),
            "queries": queries_per_record[i],
            "merged_usage": _merge_usage(*usage_parts[i]),
        }

    if return_usage:
        if return_stage_debug:
            return [
                (
                    queries_per_record[i],
                    _merge_usage(*usage_parts[i]),
                    stage_debug_per_record[i],
                )
                for i in range(n)
            ]
        return [
            (
                queries_per_record[i],
                _merge_usage(*usage_parts[i]),
            )
            for i in range(n)
        ]

    if return_stage_debug:
        return [
            (
                queries_per_record[i],
                stage_debug_per_record[i],
            )
            for i in range(n)
        ]

    return queries_per_record


def generate(
    record: dict[str, Any],
    model_name: str = DEFAULT_MODEL,
    api_list: list[str] | None = None,
    base_url_list: list[str] | None = None,
    api_call_limit: int = 20,
    max_retry: int = 3,
    return_usage: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]] | None, dict[str, Any] | None] | None:
    if not api_list:
        raise ValueError("api_list is required")
    if not base_url_list:
        raise ValueError("base_url_list is required")

    results = generate_batch(
        records=[record],
        model_name=model_name,
        api_list=api_list,
        base_url_list=base_url_list,
        api_call_limit=api_call_limit,
        max_retry=max_retry,
        return_usage=return_usage,
    )
    return results[0] if results else (([], None) if return_usage else [])
