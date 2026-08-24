export MAX_PIXELS=100352
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4

swift rollout \
  --model "Path to your model" \
  --vllm_tensor_parallel_size 4 \
  --vllm_data_parallel_size 1 \
  --vllm_max_model_len 8192 \
  --port 8000
