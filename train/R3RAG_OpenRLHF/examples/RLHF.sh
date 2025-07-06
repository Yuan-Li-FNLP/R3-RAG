set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

training_commands="openrlhf.cli.train_ppo \
   --pretrain "your_R3-RAG-CS_model_path" \
   --remote_rm_url http://localhost:5000/get_reward \
   --save_path "R3-RAG_save_path" \
   --ckpt_path "checkpoint_save_path" \
   --save_hf_ckpt \
   --use_tensorboard "tensorboard_log_dir" \
   --save_steps 8 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 4 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 64  \
   --n_samples_per_prompt 4 \
   --max_epochs 1 \
   --num_episodes 4 \
   --prompt_max_len 4096 \
   --generate_max_len 512 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data "R3-RAG_RLTraingData" \
   --input_key context_messages \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --advantage_estimator gae \
   --load_checkpoint \
   --normalize_reward"

if [[ ${1} != "slurm" ]]; then
    deepspeed --num_gpus=8 --module $training_commands
fi
