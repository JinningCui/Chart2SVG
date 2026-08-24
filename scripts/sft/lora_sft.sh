# 22GB
# qwen3: https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo1.sh
# export MASTER_PORT=23456
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
swift sft \
    --model "Path to your model" \
    --model_type qwen3_vl \
    --train_type lora \
    --dataset "Path to your dataset" \
              "Path to your dataset" \
              "Path to your dataset" \
              "Path to your dataset" \
              "Path to your dataset" \
              "Path to your dataset" \
              "Path to your dataset" \
              "Path to your dataset" \
              "Path to your dataset" \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --modules_to_save embed_tokens lm_head \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 8192 \
    --truncation_strategy right \
    --output_dir output_sft_nouse \
    --system 'You are a world-class SVG Expert and Data Visualization Engineer. Your primary objective is to interpret rasterized chart images and reconstruct them into high-quality, semantically correct SVG code.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
