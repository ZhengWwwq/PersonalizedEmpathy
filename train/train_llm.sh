# Launch PPO training for the Personalized Empathy reward model.
# Configure paths, model names, and API keys through environment variables.
unset ROCR_VISIBLE_DEVICES

# export RAY_LOG_TO_STDERR=1
# export RAY_DEBUG=1

export GRM_TRIAL=4

export JUDGE_API_KEY="${JUDGE_API_KEY:?Set JUDGE_API_KEY}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:?Set JUDGE_BASE_URL}"
export JUDGE_MODEL="${JUDGE_MODEL:?Set JUDGE_MODEL}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"

export EVAL_JUDGE_MODEL="${EVAL_JUDGE_MODEL:?Set EVAL_JUDGE_MODEL}"
export EVAL_JUDGE_API_KEY="${EVAL_JUDGE_API_KEY:?Set EVAL_JUDGE_API_KEY}"
export EVAL_JUDGE_BASE_URL="${EVAL_JUDGE_BASE_URL:?Set EVAL_JUDGE_BASE_URL}"

TRAIN_FILES="${TRAIN_FILES:?Set TRAIN_FILES to the training parquet path}"
VAL_FILES="${VAL_FILES:?Set VAL_FILES to the validation parquet path}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the base model path or HF repo id}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-personalized-empathy-grpo}"

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_max_samples=-1 \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='PereGRM' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    custom_reward_function.path="verl/utils/reward_score/peregrm_reward.py" \
    custom_reward_function.name=compute_score $@
