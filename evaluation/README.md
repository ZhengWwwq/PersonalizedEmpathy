# Evaluation

This directory evaluates model responses with fixed PERM criteria.

The workflow has two stages:

1. `prepare_criteria.py` generates context-specific criteria for each query and PERM dimension.
2. `eval.py` scores model predictions against those fixed criteria.

## Files

- `prepare_criteria.py`: generates sidecar criteria for `resonation`, `expression`, and `reception`.
- `eval.py`: runs LLM-as-judge evaluation with fixed criteria.
- `eval.sh`: environment-variable driven evaluation entry point.
- `api_call.py`: async OpenAI-compatible API caller.
- `utils.py`: JSON loading, preprocessing, and score parsing helpers.

## Generate Fixed Criteria

Set API credentials through environment variables:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-chat"
```

Run criteria generation:

```bash
python prepare_criteria.py \
  --input ../prepare_dataset/dataset/ei_trainset/final_data/English.json \
  --output ./criteria/English_criteria.json \
  --model "${OPENAI_MODEL}" \
  --concurrency 64
```

Use `--limit N` for a smoke test and `--resume` to continue from an existing output file.

## Run Evaluation

`eval.py` expects:

- a dataset JSON file in the final Personalized Empathy format,
- a prediction JSON file with `responses` aligned to each query,
- either criteria embedded under each query's `criteria` field or a sidecar criteria file from `prepare_criteria.py`.

Example:

```bash
export EVAL_DATASET_PATH="../prepare_dataset/dataset/ei_trainset/final_data/English.json"
export EVAL_PREDICT_PATH="./predictions/model_outputs.json"
export EVAL_CRITERIA_PATH="./criteria/English_criteria.json"
export EVAL_API_KEY="your_api_key"
export EVAL_BASE_URL="https://api.deepseek.com"
export EVAL_MODEL="deepseek-chat"

bash eval.sh
```

The script writes results next to the prediction file with the suffix `_fix_criteria_res.json`.

## Output Metrics

The evaluator reports three PERM dimensions:

- `resonation`: emotional attunement and empathic understanding,
- `expression`: emotionally intelligent communication,
- `reception`: likely user-side reception and willingness to continue.

The final score is the average normalized score across these three dimensions.

## Privacy

Do not commit prediction files, raw datasets, criteria generated from private data, API keys, judge logs, or local machine paths.
