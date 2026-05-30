from api_call import api_call
import json
from typing import Any


DEFAULT_MODEL = "deepseek-chat"


def _extract_generated_query_info(query: str | dict[str, Any]) -> tuple[str, str]:
	if isinstance(query, dict):
		query_text = str(query.get("query", "")).strip()
		reason_text = str(query.get("reasoning", "")).strip()
		return query_text, reason_text
	return str(query).strip(), ""


def _parse_relevant_mem_positions(query: str | dict[str, Any]) -> set[int]:
	if not isinstance(query, dict):
		return set()

	relevant_mem = query.get("relevant_mem", [])
	if isinstance(relevant_mem, list):
		positions = relevant_mem
	else:
		positions = []
		text = str(relevant_mem).strip()
		if text:
			try:
				parsed = json.loads(text)
				if isinstance(parsed, list):
					positions = parsed
			except Exception:
				positions = []

	result = set()
	for position in positions:
		try:
			result.add(int(position))
		except (TypeError, ValueError):
			continue
	return result


def _build_memory_info(query: str | dict[str, Any], record: dict[str, Any], max_items: int | None) -> str:
	memory_items = record.get("memory_items", [])
	if not isinstance(memory_items, list):
		return ""

	items_num = len(memory_items) if max_items is None else min(max_items, len(memory_items))
	relevant_positions = _parse_relevant_mem_positions(query)

	relevant_lines = []
	irrelevant_lines = []
	for mem_position, mem in enumerate(memory_items[:items_num], start=1):
		label = mem.get("label", "")
		suggestion = mem.get("label_suggestion", "")
		value = str(mem.get("value", "")).strip()
		if value:
			line = f"mem_position={mem_position}; label={label}; possible type={suggestion}; value={value}"
			if mem_position in relevant_positions:
				relevant_lines.append(line)
			else:
				irrelevant_lines.append(line)

	sections = []
	if relevant_lines:
		sections.append("Relevant Memory:\n" + "\n".join(relevant_lines))
	if irrelevant_lines:
		sections.append("Irrelevant Memory:\n" + "\n".join(irrelevant_lines))
	return "\n\n".join(sections)


def _extract_persona_and_context(query: str | dict[str, Any], record: dict[str, Any]) -> tuple[str, str, str]:
	"""Extract persona, situation and dialogue_context from query or record."""
	persona_profile = ""
	situation_text = ""
	dialogue_context = ""
	
	if isinstance(query, dict):
		dialogue_context = str(query.get("dialogue_context", "")).strip()
		
		situation = query.get("situation", {})
		if isinstance(situation, dict):
			situation_text = str(situation.get("situation", "")).strip()
	
	# Try to get persona from record (from stage_debug if available)
	if "stage_debug" in record and isinstance(record["stage_debug"], dict):
		stage_debug = record["stage_debug"]
		if "persona" in stage_debug and isinstance(stage_debug["persona"], dict):
			persona_profile = str(stage_debug["persona"].get("persona_profile", "")).strip()
	
	return persona_profile, situation_text, dialogue_context


def prepare_prompt(
	record: dict[str, Any],
	query: str | dict[str, Any],
	max_memory_items: int | None = 8,
) -> list[dict[str, str]]:
	memory_text = _build_memory_info(query, record, max_items=max_memory_items)
	generated_query, reason_process = _extract_generated_query_info(query)
	persona_profile, situation_text, dialogue_context = _extract_persona_and_context(query, record)

	system_prompt = (
		"You are an expert in social intelligence and empathy-driven dialogue analysis. "
		"Judge whether the given query is high-EQ related for this user context."
	)

	user_prompt = '''## Task 
Based on the provided **User Memory**, **Persona**, **Situation Context**, **Dialogue Context**, and the **Generated Query**, evaluate the interaction quality and determine if the AI assistant needs to provide emotional support or a high-EQ response. Details are provided as below.

### Instructions (IMPORTANT - Must Check Alignment)
1. **Extract key traits from Persona**: What are the user's main characteristics, emotional patterns, speaking style?
2. **Check Query Against Persona**: Does the query sound like this person? Does it match their values, concerns, and way of expressing themselves?
3. **Check Query Against Situation**: Does the query appropriately respond to the situation described? Is the timing and emotional tone right?
4. **Check Query Against Memory**: Are the facts and background consistent? Does it align with what the user has shared?
5. For "language" field: Read Dialogue Context first and identify what language the USER actually speaks. Mark that language.
- Analyze memory, persona, and situation to understand the user's context
### Criteria for "logical consistency"
    - TRUE(1): The query is likely to happen in scenarios like user-assitant interactions. The query can be answered by given relevant memory logically and reasonably. The reasoning is reasonable and correct. There's no factual errors or conflict between query and memories.
    - FALSE(0): The query is impossible to happen in user-assitant interactions.There is **absolutely** no convincing and solid relationship between query and memories. The reasoning has critical flaws in logical or factual basis. There's conflict between query and memories.

### Criteria for "role awareness" (Check Against Persona + Situation)
    - TRUE(1): Query MATCHES the persona (same emotional tone, values, concerns). Query FITS the situation (appropriate timing and emotional response). Query sounds natural like this specific person, not generic. No contradiction between query and memory/persona.
    - FALSE(0): Query CONTRADICTS persona (mismatched values/style/tone). Query DOESN'T FIT situation (wrong response, incorrect timing). Query sounds bot-generated or generic, not like this person. Contains facts that contradict memory or persona.

### Criteria for "language" (CRITICAL - Use Dialogue Language)
- The "language" field MUST reflect the ORIGINAL DIALOGUE LANGUAGE, NOT the generated query language
- Steps:
  1. Look at the "Dialogue Context" section - what language(s) does the user actually use?
  2. If dialogue is pure English → "English"
  3. If dialogue is pure Chinese/Russian/Spanish/French/Korean/Japanese/German → use that language
  4. If dialogue mixes languages (e.g., Russian + English), choose the PRIMARY one (used most, or used in emotional content)
  5. If cannot determine → "Other"

### Criteria for "emotional_support": true (1)
- User is seeking psychological support, venting feelings, or sharing emotional issues
- Query involves personal contexts (family, social issues, personal history)  
- A purely factual response would be cold or insensitive given the user's background

### Criteria for "emotional_support": false (0)
- **Purely Functional**: The query is strictly technical, factual, or informational (e.g., coding, math, general knowledge).
- **No Memory Needed**: The assistant can fulfill the request perfectly without knowing the user's personal history or psychological state.
- **Objective Interaction**: The interaction is tool-like and lacks any personal or emotional vulnerability.

---

### Input Format:
User Persona:
{persona_profile}

Situation Context (why user asks now):
{situation_text}

Dialogue Context (recent turns for language/tone alignment):
{dialogue_context}

Memory info:
{memory_info}

Generated query: 
{generated_query}

Reasoning: 
{reason_process}

### Output Format:
Return ONLY a JSON object:
```json
{
    "emotional_support": boolean,
    "logical_consistency": boolean,
    "role_awareness": boolean,
    "language": "English | Chinese | Japanese | Korean | Spanish | French | German | Other",
    "reason": "xxx"
}
```
---
Here are some examples:

Case1: Seeking Support:
**User Memory**: I just want to be with her, I'm not request anything difficult for her. I respect everything of her
**Current Query**: Romantic feelings for a married woman
**Reason**: The user is expressing romantic feelings for a married woman, which requires emotional support.


Case2: Memory-Integrated EQ:
**User Memory**: I invited him to dinner and karaoke, and he never directly refused.
**Current Query**: Romantic interest in a specific boy
Experiencing emotional distress from being restricted to a chat-only mode
Persistent curiosity about the boy's contradictory behavior (profile visits vs. chat avoidance)
Ambivalence about maintaining contact (delete vs. keep connection)
**Reason**: The user is expressing romantic interest in a specific boy, but he is not interested in chatting with him. He is more interested in the boy's contradictory behavior (profile visits vs. chat avoidance).


Case3: High-EQ Necessity:
**User Memory**: 8I want bots to play with. Because I dont have friend :
**Current Query**: No friends to play with
**Reason**: The user is expressing that he does not have any friends to play with, which requires a high-EQ response.
'''

	return [
		{"role": "system", "content": system_prompt},
		{
			"role": "user",
			"content": user_prompt
			.replace('{persona_profile}', persona_profile or 'N/A')
			.replace('{situation_text}', situation_text or 'N/A')
			.replace('{dialogue_context}', dialogue_context or 'N/A')
			.replace('{memory_info}', memory_text or 'N/A')
			.replace('{generated_query}', generated_query or 'N/A')
			.replace('{reason_process}', reason_process or 'N/A'),
		},
	]


def parse_response(response: str | None) -> dict[str, Any] | None:
	if not response:
		return None

	text = response.strip()
	try:
		return json.loads(text, strict=False)
	except Exception:
		pass

	try:
		if "```json" in text:
			text = text.split("```json", 1)[1].split("```", 1)[0].strip()
		elif "```" in text:
			text = text.split("```", 1)[1].split("```", 1)[0].strip()
		return json.loads(text, strict=False)
	except Exception:
		return None


def inspect_query(
	record: dict[str, Any],
	query: str | dict[str, Any],
	model_name: str = DEFAULT_MODEL,
	api_list: list[str] | None = None,
	base_url_list: list[str] | None = None,
	api_call_limit: int = 20,
	max_retry: int = 3,
	max_memory_items: int | None = 8,
	return_usage: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any] | None, dict[str, Any] | None] | None:
	"""Inspect one generated query with record context and judge if it is high-EQ related."""
	if not api_list:
		raise ValueError("api_list is required")
	if not base_url_list:
		raise ValueError("base_url_list is required")

	prompt = prepare_prompt(
		record=record,
		query=query,
		max_memory_items=max_memory_items,
	)

	# Judge one generated query with the configured OpenAI-compatible endpoint.
	responses = api_call(
		model_name=model_name,
		user_prompt_list=[prompt],
		api_list=api_list,
		base_url_list=base_url_list,
		api_call_limit=api_call_limit,
		max_retry=max_retry,
		max_completion_tokens=1024,
		return_usage=return_usage,
	)

	if not responses:
		return (None, None) if return_usage else None

	response = responses[0]
	if return_usage and isinstance(response, dict):
		content = response.get("content")
		usage = response.get("usage")
		return parse_response(content if isinstance(content, str) else None), usage if isinstance(usage, dict) else None
	return parse_response(response if isinstance(response, str) else None)


def inspect_batch(
	items: list[dict[str, Any]],
	model_name: str,
	api_list: list[str],
	base_url_list: list[str],
	api_call_limit: int = 20,
	max_retry: int = 3,
	max_memory_items: int | None = 8,
	return_usage: bool = False,
) -> list[dict[str, Any] | None] | list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
	"""
	Batch inspect.

	Each item should contain:
	- item["record"]: dataset record
	- item["query"]: generated query string or generated query object
	"""
	prompts = [
		prepare_prompt(
			record=item.get("record", {}),
			query=item.get("query", ""),
			max_memory_items=max_memory_items,
		)
		for item in items
	]

	# Judge all prompts in one API batch while preserving input order.
	responses = api_call(
		model_name=model_name,
		user_prompt_list=prompts,
		api_list=api_list,
		base_url_list=base_url_list,
		api_call_limit=api_call_limit,
		max_retry=max_retry,
		max_completion_tokens=1024,
		return_usage=return_usage,
	)

	if return_usage:
		results = []
		for resp in responses:
			if isinstance(resp, dict):
				content = resp.get("content")
				usage = resp.get("usage")
				results.append(
					(
						parse_response(content if isinstance(content, str) else None),
						usage if isinstance(usage, dict) else None,
					)
				)
			else:
				results.append((parse_response(resp if isinstance(resp, str) else None), None))
		return results

	return [parse_response(resp if isinstance(resp, str) else None) for resp in responses]
