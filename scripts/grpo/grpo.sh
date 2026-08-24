# Warning: do not enable these settings on NVLink-equipped machines such as L20Z.
# They disable high-speed communication and can substantially slow training.
# Use them only without NVLink or to work around Docker shared-memory errors.
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

export MAX_PIXELS=100352  # Leave memory available for an 8192-token sequence.
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set this to the GPU IDs used for training.
export NPROC_PER_NODE=4                      # Must match the number of GPUs listed above.

swift rlhf \
  --rlhf_type grpo \
  --model "Path to your model" \
  --train_type lora \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8000 \
  --async_generate true \
  --dataset 'grpo_train_json/BarGraph' \
            'grpo_train_json/AreaGraph' \
            'grpo_train_json/LineGraph' \
            'grpo_train_json/PieChart' \
            'grpo_train_json/quartz_normalized' \
  --external_plugins "Path to your plugin" \
  --reward_funcs svg_pipeline \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --num_train_epochs 1 \
  --max_length 8192 \
  --max_completion_length 8192 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --eval_steps 20 \
  --save_steps 20 \
  --learning_rate 1e-6 \
  --logging_steps 1 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 0 \
  --num_generations 4 \
  --deepspeed zero2 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 80 \
  --log_completions true \
  --offload_optimizer true \
  --offload_model true \
  --beta 0.005 \
  --max_grad_norm 0.5 \
  --report_to wandb \
  2>&1 | tee output_grpo/grpo_run_$(date +%Y%m%d_%H%M%S).txt
