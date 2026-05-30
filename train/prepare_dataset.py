"""Convert the final empathy dataset JSON into verl-compatible parquet."""
import json
import pandas as pd
import argparse
from typing import Dict

# The PPO trainer expects a parquet file where each row contains a chat prompt
# and extra_info fields consumed by the custom reward function.

user_prompt_template = '''You are a helpful, warm, and empathetic AI assistant.
You will be provided with the extracted memories from the previous dialogue. Your task is to generate a response to the user.
---

**Memory extracted from previous conversation:** {memory}

**User Query:** {query}
'''

def make_map_fn(split: str):
    """Create a row mapper that tags examples with the train/validation split."""
    def process_fn(example: Dict, idx: int):
        """Map one flattened query example into verl's DataProto-friendly schema."""
        persona = example['persona']
        scenario = example['scenario']
        memory = example['memory']
        
        query = example['query']

        prompt = [
            {"role": "user", "content": user_prompt_template.format(
                memory="\n".join([f"{i+1}.{memory[i]}" for i in range(len(memory))]),
                query=query
            )}
        ]

        return {
            "data_source": "Ours_perm_format",
            "prompt": prompt,
            "ability": "empathy",
            "reward_model": {
                "style": "rule",
                "ground_truth": ""
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "persona": persona, 
                "scenario": scenario, 
                "memory": memory,
                "query": query
            }
        }
    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_path')
    parser.add_argument('--split', default="train")
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        anchor_data = json.load(f)
    processed_data = []
    for item in anchor_data:
        for query in item['queries']:
            new_item = {
                "persona": item['persona']['persona_profile'], 
                "scenario": query['situation']['situation'], 
                "query": query['query'], 
                "memory": [mem_item['value'] for mem_item in item['extracted_memory']], 
                "criteria": query['criteria']
            }
            processed_data.append(new_item)

    df = pd.DataFrame(processed_data)

    data_processed = [make_map_fn(args.split)(row.to_dict(), i) for i, row in df.iterrows()]

    pd.DataFrame(data_processed).to_parquet(args.output_path)
