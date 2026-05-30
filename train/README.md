# Training

This directory contains the verl-based PPO training code for Personalized Empathy.

## Main Files

- `prepare_dataset.py`: converts final dataset JSON files into parquet rows accepted by verl.
- `train_llm.sh`: launches GRPO/PPO training with the custom reward function.
- `verl/utils/reward_score/peregrm_reward.py`: main Personalized Empathy reward, judged on resonation, expression, reception, and a standby quality penalty.
- `verl/utils/reward_score/rlpa_reward.py`: simpler personalization baseline reward.

The rest of `verl/`, `docker/`, `scripts/`, and `tests/` are the underlying training framework and utilities.

## Prepare Parquet Data

Input JSON should come from `prepare_dataset/dataset/ei_trainset/final_data/<Language>.json`.

```bash
python prepare_dataset.py \
  --input_file ../prepare_dataset/dataset/ei_trainset/final_data/English.json \
  --output_path ./data/train.parquet \
  --split train
```

Run the same command with `--split val` to prepare validation data.

## Required Environment Variables

`train_llm.sh` intentionally reads secrets and machine-specific paths from environment variables:

```bash
export TRAIN_FILES="/path/to/train.parquet"
export VAL_FILES="/path/to/val.parquet"
export MODEL_PATH="/path/to/base/model/or/hf-repo"

export JUDGE_API_KEY="..."
export JUDGE_BASE_URL="..."
export JUDGE_MODEL="..."

export EVAL_JUDGE_API_KEY="..."
export EVAL_JUDGE_BASE_URL="..."
export EVAL_JUDGE_MODEL="..."

export WANDB_API_KEY="..."          # optional
export EXPERIMENT_NAME="perm-run"   # optional
```

## Run Training

```bash
bash train_llm.sh
```

Training logs, checkpoints, and Weights & Biases state are ignored by Git. Keep model weights and private parquet files outside the repository or in ignored local directories.
