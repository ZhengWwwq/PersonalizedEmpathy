from filter_list import intent_keyword_list, memory_label_list, key_words_for_unmatched
import json
import os
from api_call import api_call

DIR = "./dataset/by_label_json/"


def multiple_records():
    """Print source files that contain more than one raw record."""
    for file_name in os.listdir(DIR):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(DIR, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, list):
            continue

        if len(data) > 1:
            print(f"{file_name} has {len(data)} records")

def filter_intent():
    """Keep records whose ranked intent category/subtype matches the EQ-related allowlist."""
    filtered = []
    keywords = [k.lower() for k in intent_keyword_list]
    cnt = 0

    for file_name in os.listdir(DIR):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(DIR, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, list):
            continue

        for record in data:
            intents_ranked = record.get("intents_ranked", [])
            matched = False
            cnt +=1
            for intent in intents_ranked:
                category = str(intent.get("intent_category", "")).lower()
                subtype = str(intent.get("intent_subtype", "")).lower()

                if any((kw in category) or (kw in subtype) for kw in keywords):
                    matched = True
                    break

            if matched:
                filtered.append({
                    "source_file": file_name,
                    "record": record,
                })

    return filtered.__len__(),filtered,cnt

def filter_memory(filt_records: list):
    """Keep records with memory labels likely to support personalized empathy queries."""
    filtered = []
    label_prefixes = [k.lower() for k in memory_label_list]
    unmatched_keywords = [k.lower() for k in key_words_for_unmatched]

    for item in filt_records:
        record = item["record"]
        source_file = item["source_file"]
        memories = record.get("memory_items", [])
        matched = False

        for memory in memories:
            label = str(memory.get("label", "")).lower()
            label_suggestion = str(memory.get("label_suggestion", "")).lower()

            # Rule 1: mapped labels must start with one of the selected memory prefixes.
            if label and label != "unmapped":
                if any(label.startswith(prefix) for prefix in label_prefixes):
                    matched = True
                    break

            # Rule 2: unmapped labels can pass if the suggested label contains an EQ keyword.
            if label == "unmapped":
                if any(keyword in label_suggestion for keyword in unmatched_keywords):
                    matched = True
                    break

        if matched:
            filtered.append({
                "source_file": source_file,
                "record": record,
            })

    return filtered.__len__(),filtered

if __name__ == "__main__":
    intent_count, filtered_records, total_records = filter_intent()
    left_count, memory_filtered_records = filter_memory(filtered_records)
    print(left_count)
    
    with open("./dataset/ei_trainset/raw_filtered.json", "w", encoding="utf-8") as f:
        json.dump(memory_filtered_records, f, ensure_ascii=False, indent=2)
