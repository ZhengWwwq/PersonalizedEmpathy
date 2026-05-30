import os
import json
from typing import Any

DIR = "./dataset/ei_trainset/"

query = DIR + "generated_queries.json"
inspection = DIR + "inspection_results.json"

FINAL = DIR + "final_data/"

def build_record(filtered_query: list, qrecord: dict[str, Any], irecord: dict[str, Any]) -> dict[str, Any]:
    """Build one final public dataset record from generated queries and inspections."""
    language = irecord["queries_inspected"]
    if language.__len__() > 0:
        language = language[0]["inspection"].get("language", "Other")
    else:   
        language = "Other"
    source = qrecord.get("source_file", "")
    query = []
    source_queries = qrecord.get("queries", [])
    if isinstance(source_queries, list):
        for idx in filtered_query:
            try:
                query_idx = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= query_idx < len(source_queries):
                query.append(source_queries[query_idx])
    for item in query:
        item["query_id"]=""
        if isinstance(item, dict):
            item.pop("dialogue_context", None)
    persona = qrecord.get("persona", {})
    original_id = qrecord["record"]["sessions"][0].get("session_id", "")
    turn = qrecord["record"]["sessions"][0].get("turns", [])
    conversation = []
    for t in turn:
        role = t.get("role", "")
        text = t.get("text", "")
        conversation.append({
            "role": role,
            "text": text
        })
    memory_items = qrecord["record"].get("memory_items", [])
    memory = []
    for m in memory_items:
        label = m.get("label", "")
        value = m.get("value", "")
        suggestion = m.get("label_suggestion", "")
        evidence = m.get("evidence", "")
        memory.append({
            "label": label,
            "label_suggestion": suggestion,
            "value": value,
            "evidence": evidence
        })
    return {
        "language": language,
        "source_file": source,
        "original_sid": original_id,
        "session_id": "",
        "persona": persona,
        "queries": query,
        "conversation": conversation,
        "extracted_memory": memory,
    }


def final_filter():
    """Keep only inspected queries that pass all three EQ-quality gates."""
    with open(query, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    with open(inspection, "r", encoding="utf-8") as f:
        inspection_data = json.load(f)

    filtered = []
    total = 0
    filtered_count = 0
    g1 =0
    g2 =0
    g3 =0
    for q_item, i_item in zip(query_data, inspection_data):
        if not isinstance(q_item, dict) or not isinstance(i_item, dict):
            continue
        
        if not q_item.get("source_file") == i_item.get("source_file"):
            print("Critical error: source_file mismatch between query and inspection data.")
            break
        
        inspection_results = i_item.get("queries_inspected", [])
        total += len(inspection_results)
       
        filtered_query = []
        for idx, inspect_obj in enumerate(inspection_results):
            if not isinstance(inspect_obj, dict):
                continue
            
            result = inspect_obj.get("inspection",{})
            if result:
                es = result.get("emotional_support", False)
                lc = result.get("logical_consistency", False)
                ra = result.get("role_awareness", False)
                if es and lc and ra:
                    query_idx = inspect_obj.get("query_index", idx)
                    filtered_query.append(query_idx)
                    query_t = inspect_obj.get("query", {})
                    category = query_t.get("category", "") if isinstance(query_t, dict) else ""
                    if category == "High-EQ Interaction":
                        g1 += 1
                    elif category == "Emotional Support":
                        g2 += 1
                    elif category == "Social Strategy":
                        g3 += 1

        if filtered_query:
            filtered.append(build_record(filtered_query, q_item, i_item))
            filtered_count += len(filtered_query)
    return (g1,g2,g3),total, filtered_count, filtered

def store_in_language(filtered: list, output_dir: str):
    """Write final records into language-specific JSON files with stable IDs."""
    os.makedirs(output_dir, exist_ok=True)

    # Remove historical language output files so each run is isolated.
    for name in os.listdir(output_dir):
        if not name.endswith(".json"):
            continue
        file_path = os.path.join(output_dir, name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    language_groups = {}
    for record in filtered:
        language = record.get("language", "Other")
        if language not in language_groups:
            language_groups[language] = []
        # Remove language field in final output records.
        record_without_language = dict(record)
        record_without_language.pop("language", None)
        language_groups[language].append(record_without_language)

    for language, records in language_groups.items():
        safe_language = str(language).strip() or "Other"
        safe_language = safe_language.replace("/", "_").replace(" ", "_")

        # Fill session_id/query_id by index within each language file.
        lang_prefix = safe_language.lower()
        for session_idx, record in enumerate(records):
            session_id = f"{lang_prefix}_{session_idx}"
            record["session_id"] = session_id

            queries = record.get("queries", [])
            if isinstance(queries, list):
                for query_idx, query_item in enumerate(queries):
                    if isinstance(query_item, dict):
                        new_query_item = {"query_id": f"{session_id}:{query_idx}"}
                        for key, value in query_item.items():
                            if key != "query_id":
                                new_query_item[key] = value
                        queries[query_idx] = new_query_item

        output_file = os.path.join(output_dir, f"{safe_language}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    (g1, g2, g3), total, filtered_count, filtered_records = final_filter()
    store_in_language(filtered_records, FINAL)
    print(f"Saved language-split files to: {FINAL}")
    print(f"Filtered counts - High-EQ: {g1}, Emotional Support: {g2}, Social Strategy: {g3}")
