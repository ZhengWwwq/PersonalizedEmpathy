# Dataset Preparation

This directory builds the Personalized Empathy training data from raw memory/dialogue records.

## Pipeline

1. `filter.py`
   - Reads raw label-split JSON files from `./dataset/by_label_json/`.
   - Keeps records with EQ-related intent labels.
   - Keeps records whose memory labels are useful for personalized emotional or social-support scenarios.
   - Writes `./dataset/ei_trainset/raw_filtered.json`.

2. `query.py`
   - Runs the full generation pipeline:
     - persona extraction
     - situation generation
     - user query generation
     - generated query inspection
     - final filtering and language split
   - Main outputs:
     - `generated_queries.json`
     - `inspection_results.json`
     - `query_usage_summary.json`
     - `final_data/<Language>.json`

3. `final_filter.py`
   - Keeps only queries that pass all inspection gates:
     - `emotional_support`
     - `logical_consistency`
     - `role_awareness`
   - Removes temporary dialogue context from public outputs.
   - Assigns stable `session_id` and `query_id` values by language file.

## Configuration

Do not put API keys in source files. Export them before running:

```bash
export DATA_API_KEYS="key1,key2"
export DATA_BASE_URLS="https://api.deepseek.com"
export DATA_MODEL_NAME="deepseek-chat"
export DATA_RUN_MODE="full_pipeline"
```

`DATA_RUN_MODE` can be:

- `situation_only`
- `generation`
- `full_pipeline`

If you provide multiple API keys and one base URL, the URL is reused for all keys. If you provide multiple base URLs, the count must match `DATA_API_KEYS`.

## Run

From this directory:

```bash
python filter.py
python query.py
```

For small debugging runs, edit the `start_index`, `end_index`, and optional `source_files` / `session_ids` arguments in the `query.py` main block.

## Expected Local Layout

```text
prepare_dataset/
  api_call.py
  filter.py
  filter_list.py
  final_filter.py
  query.py
  query_generation.py
  query_inspection.py
  query_prompt.py
  dataset/
    by_label_json/
    ei_trainset/
      raw_filtered.json
      generated_queries.json
      inspection_results.json
      final_data/
```

The `dataset/` directory is ignored by Git because it can contain raw/private data and generated outputs.
