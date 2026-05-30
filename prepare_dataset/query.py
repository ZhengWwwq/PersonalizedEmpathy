from query_generation import generate_batch, preview_situation_prompt_effect
from query_inspection import inspect_batch
import json
from typing import Any
import os
from final_filter import final_filter,store_in_language,FINAL

FILE = "./dataset/ei_trainset/raw_filtered.json"
GENERATED_FILE = "./dataset/ei_trainset/generated_queries.json"
INSPECTION_FILE = "./dataset/ei_trainset/inspection_results.json"
USAGE_SUMMARY_FILE = "./dataset/ei_trainset/query_usage_summary.json"
SITUATION_FILE = "./dataset/ei_trainset/generated_situations.json"
SITUATION_USAGE_SUMMARY_FILE = "./dataset/ei_trainset/situation_usage_summary.json"
STAGE_DEBUG_FILE = "./dataset/ei_trainset/generation_stage_debug.json"


def _split_env(name: str, default: str = "") -> list[str]:
	"""Read comma-separated environment variables without storing secrets in code."""
	value = os.environ.get(name, default)
	return [item.strip() for item in value.split(",") if item.strip()]


# API configuration. Put real credentials in environment variables, never in Git.
API_LIST = _split_env("DATA_API_KEYS")
BASE_URL_LIST = _split_env("DATA_BASE_URLS", "https://api.deepseek.com")
MODEL_NAME = os.environ.get("DATA_MODEL_NAME", "deepseek-chat")
RUN_MODE = os.environ.get("DATA_RUN_MODE", "full_pipeline")  # situation_only | generation | full_pipeline


def load_records(file_path: str) -> list[dict[str, Any]]:
	"""Load records from JSON file."""
	if not os.path.exists(file_path):
		print(f"File not found: {file_path}")
		return []
	with open(file_path, "r", encoding="utf-8") as f:
		return json.load(f)


def save_json(file_path: str, data: Any) -> None:
	with open(file_path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


def _require_api_config() -> None:
	if not API_LIST:
		raise ValueError("Set DATA_API_KEYS before running query.py")
	if len(BASE_URL_LIST) == 1 and len(API_LIST) > 1:
		BASE_URL_LIST.extend(BASE_URL_LIST * (len(API_LIST) - 1))
	if len(BASE_URL_LIST) != len(API_LIST):
		raise ValueError("DATA_BASE_URLS must contain one URL or the same count as DATA_API_KEYS")


def _safe_int(value: Any) -> int:
	try:
		if value is None:
			return 0
		return int(value)
	except (TypeError, ValueError):
		return 0


def _summarize_usage_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
	total_calls = len(entries)
	entries_with_usage = [entry for entry in entries if isinstance(entry.get("usage"), dict)]
	input_tokens = sum(_safe_int(entry["usage"].get("input_tokens")) for entry in entries_with_usage)
	output_tokens = sum(_safe_int(entry["usage"].get("output_tokens")) for entry in entries_with_usage)
	total_tokens = sum(_safe_int(entry["usage"].get("total_tokens")) for entry in entries_with_usage)

	return {
		"total_calls": total_calls,
		"calls_with_usage": len(entries_with_usage),
		"calls_without_usage": total_calls - len(entries_with_usage),
		"input_tokens": input_tokens,
		"output_tokens": output_tokens,
		"total_tokens": total_tokens,
		"avg_input_tokens_per_call": input_tokens / total_calls if total_calls else 0,
		"avg_output_tokens_per_call": output_tokens / total_calls if total_calls else 0,
		"avg_total_tokens_per_call": total_tokens / total_calls if total_calls else 0,
	}


def _build_usage_summary(generation_usage: list[dict[str, Any]], inspection_usage: list[dict[str, Any]]) -> dict[str, Any]:
	overall_usage = generation_usage + inspection_usage
	return {
		"generation": _summarize_usage_entries(generation_usage),
		"inspection": _summarize_usage_entries(inspection_usage),
		"overall": _summarize_usage_entries(overall_usage),
		"generation_calls": generation_usage,
		"inspection_calls": inspection_usage,
	}


def _print_usage_summary(usage_summary: dict[str, Any]) -> None:
	for stage in ["generation", "inspection", "overall"]:
		stats = usage_summary.get(stage, {})
		print(
			f"{stage}: "
			f"calls={stats.get('total_calls', 0)}, "
			f"total_tokens={stats.get('total_tokens', 0)}, "
			f"avg_total_tokens_per_call={stats.get('avg_total_tokens_per_call', 0):.2f}, "
			f"avg_input_tokens_per_call={stats.get('avg_input_tokens_per_call', 0):.2f}, "
			f"avg_output_tokens_per_call={stats.get('avg_output_tokens_per_call', 0):.2f}"
		)


def _count_generated_queries(generated_queries: list[dict[str, Any]]) -> int:
	return sum(len(item.get("queries", [])) for item in generated_queries)


def _count_inspected_queries(inspected_records: list[dict[str, Any]]) -> int:
	return sum(len(item.get("queries_inspected", [])) for item in inspected_records)


def query_generation(
	start_index: int = 0,
	end_index: int | None = None,
	batch_size: int = 50,
	source_files: list[str] | None = None,
	session_ids: list[str] | None = None,
):
	"""
	Generate queries from records in raw_filtered.json.
	The internal pipeline in `generate_batch` is already:
	persona -> situations -> per-situation query.
	start_index / end_index control which slice of records to process.
	batch_size controls how many records are sent to the LLM in parallel per batch.
	source_files: if provided, only process records whose source_file is in this list.
	session_ids: if provided, only process records that contain any matching session_id.
	When both filters are given, a record passes if it matches either one (OR logic).
	When both are None, no extra filtering is applied.
	"""
	print("Loading records from", FILE)
	records_data = load_records(FILE)

	if not records_data:
		print("No records found.")
		return [], []

	end_index = end_index if end_index is not None else len(records_data)
	slice_data = records_data[start_index:end_index]

	if source_files is not None or session_ids is not None:
		def _passes_filter(item: dict) -> bool:
			if source_files is not None and item.get("source_file") in source_files:
				return True
			if session_ids is not None:
				for s in item.get("record", {}).get("sessions", []):
					if s.get("session_id") in session_ids:
						return True
			return False
		slice_data = [item for item in slice_data if _passes_filter(item)]

	total = len(slice_data)
	print(f"Loaded {len(records_data)} records total. Processing [{start_index}, {end_index}) - {total} records after filter.")

	generated_queries = []
	generation_usage = []
	stage_debug_results = []

	for batch_start in range(0, total, batch_size):
		batch_end = min(batch_start + batch_size, total)
		batch = slice_data[batch_start:batch_end]
		global_start = start_index + batch_start
		global_end = start_index + batch_end - 1

		print(f"\n[Generation] Records {global_start}~{global_end} ({batch_end - batch_start} records) | total progress: {batch_end}/{total}...")

		records, meta = [], []
		for local_idx, item in enumerate(batch):
			record = item.get("record", {})
			source_file = item.get("source_file", "unknown")
			if not record:
				continue
			records.append(record)
			meta.append((source_file, record, global_start + local_idx))

		if not records:
			continue

		results = generate_batch(
			records=records,
			model_name=MODEL_NAME,
			api_list=API_LIST,
			base_url_list=BASE_URL_LIST,
			api_call_limit=20,
			max_retry=3,
			return_usage=True,
			return_stage_debug=True,
		)

		success = 0
		for (source_file, record, global_idx), (queries, usage, stage_debug) in zip(meta, results):
			persona_debug = stage_debug.get("persona", {}) if isinstance(stage_debug, dict) else {}
			persona_parsed = persona_debug.get("parsed") if isinstance(persona_debug, dict) else None

			generation_usage.append({
				"stage": "generation",
				"source_file": source_file,
				"record_id": record.get("line_index", global_idx),
				"usage": usage,
			})
			stage_debug_results.append({
				"source_file": source_file,
				"record_id": record.get("line_index", global_idx),
				"stage_debug": stage_debug,
			})
			if queries:
				success += 1
				generated_queries.append({
					"source_file": source_file,
					"record": record,
					"persona": persona_parsed,
					"queries": queries,
					"generation_usage": usage,
				})

		print(f"  -> {success}/{len(records)} succeeded | accumulated: {len(generated_queries)} records")

	save_json(GENERATED_FILE, generated_queries)
	save_json(STAGE_DEBUG_FILE, stage_debug_results)
	generated_query_count = _count_generated_queries(generated_queries)
	print(f"\n[Summary][Generation] records_with_queries={len(generated_queries)} | total_queries={generated_query_count}")
	print(f"\n[OK] Generated queries saved to {GENERATED_FILE} ({len(generated_queries)} records)")
	print(f"[OK] Stage debug saved to {STAGE_DEBUG_FILE} ({len(stage_debug_results)} records)")
	return generated_queries, generation_usage


def situation_generation(
	start_index: int = 0,
	end_index: int | None = 100,
	batch_size: int = 20,
	source_files: list[str] | None = None,
	session_ids: list[str] | None = None,
):
	"""
	Generate persona + situations for quick debugging.
	Results are saved to SITUATION_FILE and usage summary to SITUATION_USAGE_SUMMARY_FILE.
	"""
	print("Loading records from", FILE)
	records_data = load_records(FILE)

	if not records_data:
		print("No records found.")
		return [], {}

	end_index = end_index if end_index is not None else len(records_data)
	slice_data = records_data[start_index:end_index]

	if source_files is not None or session_ids is not None:
		def _passes_filter(item: dict) -> bool:
			if source_files is not None and item.get("source_file") in source_files:
				return True
			if session_ids is not None:
				for s in item.get("record", {}).get("sessions", []):
					if s.get("session_id") in session_ids:
						return True
			return False
		slice_data = [item for item in slice_data if _passes_filter(item)]

	total = len(slice_data)
	print(f"Loaded {len(records_data)} records total. Situation preview on [{start_index}, {end_index}) - {total} records after filter.")

	situation_results: list[dict[str, Any]] = []
	persona_usage_entries: list[dict[str, Any]] = []
	situation_usage_entries: list[dict[str, Any]] = []

	for batch_start in range(0, total, batch_size):
		batch_end = min(batch_start + batch_size, total)
		print(f"[Situation] Records {batch_start + 1}~{batch_end} / {total}...")
		for local_idx, item in enumerate(slice_data[batch_start:batch_end], start=batch_start):
			record = item.get("record", {})
			source_file = item.get("source_file", "unknown")
			global_idx = start_index + local_idx
			if not record:
				continue

			preview = preview_situation_prompt_effect(
				record=record,
				model_name=MODEL_NAME,
				api_list=API_LIST,
				base_url_list=BASE_URL_LIST,
				api_call_limit=20,
				max_retry=3,
			)

			persona_usage_entries.append({
				"stage": "persona",
				"source_file": source_file,
				"record_id": record.get("line_index", global_idx),
				"usage": preview.get("persona_usage"),
			})
			situation_usage_entries.append({
				"stage": "situation",
				"source_file": source_file,
				"record_id": record.get("line_index", global_idx),
				"usage": preview.get("situation_usage"),
			})

			situation_results.append({
				"source_file": source_file,
				"record_id": record.get("line_index", global_idx),
				"persona": preview.get("persona"),
				"situations": preview.get("situations", []),
				"skipped": preview.get("skipped", False),
				"skip_reason": preview.get("skip_reason", ""),
				"persona_usage": preview.get("persona_usage"),
				"situation_usage": preview.get("situation_usage"),
			})

	save_json(SITUATION_FILE, situation_results)
	print(f"\n[OK] Situation results saved to {SITUATION_FILE} ({len(situation_results)} records)")

	usage_summary = {
		"persona": _summarize_usage_entries(persona_usage_entries),
		"situation": _summarize_usage_entries(situation_usage_entries),
		"overall": _summarize_usage_entries(persona_usage_entries + situation_usage_entries),
		"persona_calls": persona_usage_entries,
		"situation_calls": situation_usage_entries,
	}
	save_json(SITUATION_USAGE_SUMMARY_FILE, usage_summary)
	print(f"[OK] Situation usage summary saved to {SITUATION_USAGE_SUMMARY_FILE}")

	return situation_results, usage_summary


def query_inspection(
	generated_queries: list[dict[str, Any]] | None = None,
	generation_usage: list[dict[str, Any]] | None = None,
	batch_size: int = 100,
):
	"""
	Inspect generated queries using the inspection module.
	All queries are flattened and sent to the LLM in parallel batches.
	batch_size controls how many queries are sent per batch.
	"""
	if generated_queries is None:
		if not os.path.exists(GENERATED_FILE):
			print(f"File not found: {GENERATED_FILE}")
			return [], {}
		with open(GENERATED_FILE, "r", encoding="utf-8") as f:
			generated_queries = json.load(f)

	if generation_usage is None:
		generation_usage = [
			{
				"stage": "generation",
				"source_file": rec_data.get("source_file", "unknown"),
				"record_id": rec_data.get("record", {}).get("line_index", rec_idx),
				"usage": rec_data.get("generation_usage"),
			}
			for rec_idx, rec_data in enumerate(generated_queries)
		]

	# Flatten all (record, query) pairs for batch processing
	flat_items: list[dict[str, Any]] = []
	flat_meta: list[tuple] = []  # (rec_idx, q_idx, source_file, record, query_obj)
	for rec_idx, rec_data in enumerate(generated_queries):
		record = rec_data["record"]
		source_file = rec_data["source_file"]
		for q_idx, query_obj in enumerate(rec_data.get("queries", [])):
			flat_items.append({"record": record, "query": query_obj})
			flat_meta.append((rec_idx, q_idx, source_file, record, query_obj))

	total_queries = len(flat_items)
	print(f"Inspecting {total_queries} queries across {len(generated_queries)} records...")

	flat_results: list[dict[str, Any] | None] = []
	flat_usages: list[dict[str, Any] | None] = []

	for batch_start in range(0, total_queries, batch_size):
		batch_end = min(batch_start + batch_size, total_queries)
		print(f"\n[Inspection] Queries {batch_start + 1}~{batch_end} / {total_queries}...")

		batch_results = inspect_batch(
			items=flat_items[batch_start:batch_end],
			model_name=MODEL_NAME,
			api_list=API_LIST,
			base_url_list=BASE_URL_LIST,
			api_call_limit=20,
			max_retry=3,
			max_memory_items=8,
			return_usage=True,
		)

		success = 0
		for result, usage in batch_results:
			flat_results.append(result)
			flat_usages.append(usage)
			if result:
				success += 1

		print(f"  -> {success}/{len(batch_results)} succeeded | accumulated: {sum(1 for r in flat_results if r)}")

	# Re-aggregate flat results back to per-record structure
	per_record: list[dict[str, Any]] = [
		{
			"source_file": rec_data["source_file"],
			"record_id": rec_data["record"].get("line_index", rec_idx),
			"generation_usage": rec_data.get("generation_usage"),
			"queries_inspected": [],
		}
		for rec_idx, rec_data in enumerate(generated_queries)
	]

	inspection_usage: list[dict[str, Any]] = []
	for (rec_idx, q_idx, source_file, record, query_obj), result, usage in zip(
		flat_meta, flat_results, flat_usages
	):
		inspection_usage.append({
			"stage": "inspection",
			"source_file": source_file,
			"record_id": record.get("line_index", rec_idx),
			"query_index": q_idx,
			"query": query_obj.get("query", "") if isinstance(query_obj, dict) else str(query_obj),
			"usage": usage,
		})
		if result:
			per_record[rec_idx]["queries_inspected"].append({
				"query_index": q_idx,
				"query": query_obj,
				"inspection": result,
				"inspection_usage": usage,
			})

	save_json(INSPECTION_FILE, per_record)
	print(f"\n[OK] Inspection results saved to {INSPECTION_FILE}")

	usage_summary = _build_usage_summary(generation_usage, inspection_usage)
	save_json(USAGE_SUMMARY_FILE, usage_summary)
	print(f"[OK] Usage summary saved to {USAGE_SUMMARY_FILE}")
	_print_usage_summary(usage_summary)

	inspected_records_with_results = sum(1 for rec in per_record if rec.get("queries_inspected"))
	inspected_query_count = _count_inspected_queries(per_record)
	print(
		f"[Summary][Inspection] input_records={len(generated_queries)} | "
		f"input_queries={total_queries} | records_with_passed_queries={inspected_records_with_results} | "
		f"passed_queries={inspected_query_count}"
	)

	return per_record, usage_summary


if __name__ == "__main__":
	_require_api_config()
	print("=" * 60)
	print("Query Generation & Inspection Pipeline")
	print("=" * 60)

	files = []
	file_name = [f"{file}.json" for file in files]
	sess_name = []

	pipeline_outputs: dict[str, Any] = {
		"mode": RUN_MODE,
		"situations": [],
		"situation_usage": {},
		"generated": [],
		"generation_usage": [],
		"inspected": [],
		"usage_summary": {},
	}

	if RUN_MODE == "situation_only":
		situations, situation_usage = situation_generation(
			start_index=0,
			end_index=50,
			source_files=file_name if file_name else None,
			session_ids=sess_name if sess_name else None,
		)
		pipeline_outputs["situations"] = situations
		pipeline_outputs["situation_usage"] = situation_usage

	elif RUN_MODE == "generation":
		generated, generation_usage = query_generation(
			start_index=0,
			end_index=50,
			source_files=file_name if file_name else None,
			session_ids=sess_name if sess_name else None,
		)
		pipeline_outputs["generated"] = generated
		pipeline_outputs["generation_usage"] = generation_usage
		print(
			f"[Pipeline] After generation: records_with_queries={len(generated)} | "
			f"total_queries={_count_generated_queries(generated)}"
		)

	elif RUN_MODE == "full_pipeline":
		generated, generation_usage = query_generation(
			start_index=0,
			source_files=file_name if file_name else None,
			session_ids=sess_name if sess_name else None,
		)
		pipeline_outputs["generated"] = generated
		pipeline_outputs["generation_usage"] = generation_usage
		print(
			f"[Pipeline] After generation: records_with_queries={len(generated)} | "
			f"total_queries={_count_generated_queries(generated)}"
		)

		if generated:
			inspected, usage_summary = query_inspection(generated, generation_usage)
			pipeline_outputs["inspected"] = inspected
			pipeline_outputs["usage_summary"] = usage_summary
			print(
				f"[Pipeline] After inspection: records_with_passed_queries="
				f"{sum(1 for rec in inspected if rec.get('queries_inspected'))} | "
				f"passed_queries={_count_inspected_queries(inspected)}"
			)

			(g1, g2, g3), total, filtered_count, filtered_records = final_filter()
			print(f"\nFilter rate: {filtered_count}/{total} ({(filtered_count/total*100) if total else 0:.2f}%)")
			print(f"High-EQ Interaction: {g1}")
			print(f"Emotional Support: {g2}")
			print(f"Social Strategy: {g3}")
			print(
				f"[Pipeline] After final_filter: total_candidates={total} | "
				f"filtered_queries={filtered_count} | output_records={len(filtered_records)}"
			)
			store_in_language(filtered_records, FINAL)
			print(f"Saved language-split files to: {FINAL}")

	else:
		raise ValueError("RUN_MODE must be one of: situation_only, generation, full_pipeline")

	print(f"\nDone. Mode={RUN_MODE} | generated={len(pipeline_outputs['generated'])} | inspected={len(pipeline_outputs['inspected'])} | situations={len(pipeline_outputs['situations'])}")
