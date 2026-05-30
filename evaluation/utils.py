import json
import re


def load_json(file_path: str):
    """Load a UTF-8 JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_jsonl(file_path: str):
    """Load a UTF-8 JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def extract_score(output):
    """Parse a JSON judge output, including fenced JSON responses."""
    if isinstance(output, tuple):
        output = output[0]
    if "```" in output:
        output = output.split("```json")[-1].split("```")[0].strip().strip("\n").strip()
    try:
        return json.loads(output, strict=False)
    except:
        return None


def extract_boxed_score(text: str) -> int | float | None:
    """Extract a numeric score from LaTeX-style \\boxed{...} judge output."""
    match = re.search(r'\\boxed\{([\d.]+)\}', text)
    if match:
        num_str = match.group(1)
        return float(num_str)
    return None


def save_json(data, file_path: str):
    """Write pretty UTF-8 JSON."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def sort_dataset_by_session_id(dataset):
    """Sort language-split records by their generated session id."""
    def get_sort_key(item):
        session_id = item["session_id"]
        parts = session_id.split("_")
        num = int(parts[-1])
        prefix = "_".join(parts[:-1])
        return (prefix, num)
    
    dataset.sort(key=get_sort_key)
    return dataset


def preprocess_data(dataset, predict):
    """Flatten dataset sessions and prediction responses into evaluation examples."""
    assert len(dataset) == len(predict)
    processed_data = []
    for i in range(len(dataset)):
        assert len(dataset[i]['queries']) == len(predict[i]['responses'])
        for j in range(len(dataset[i]['queries'])):
            item = {
                "session_id": dataset[i]['session_id'], 
                "query_id": dataset[i]['queries'][j]['query_id'],
                "memory": "\n".join(["*" + mem_item['value'] for mem_item in dataset[i]['extracted_memory']]), 
                "scenario": dataset[i]['queries'][j]['situation']['situation'], 
                "persona": dataset[i]['persona']['persona_profile'], 
                "query": dataset[i]['queries'][j]['query'], 
                "criteria": dataset[i]['queries'][j].get('criteria', ''),
                "response": predict[i]['responses'][j]['response']
            }
            processed_data.append(item)
    return processed_data
