#! /bin/bash
# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export WANDB_PROJECT=lifedb-lora
export WANDB_RUN_ID=flow-generated-trial-11
export WANDB_RESUME=allow
export ABS_PATH=/home/ubuntu
export PYTHONPATH="$ABS_PATH/BELLE/train"
model_name_or_path=/home/ubuntu/lifeDB/python/BELLE-Llama2-13B-chat-0.4M-gptq-4bit/ # or bloomz-7b1-mt
model_name_or_path=BELLE-2/BELLE-Llama2-13B-chat-0.4M

train_file=/home/ubuntu/BELLE/train_dataset-version-1.json
validation_file=/home/ubuntu/BELLE/validation_dataset-version-1.json
output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}-${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=2048

#FT
# torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --deepspeed configs/deepspeed_config.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 8e-6 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --use_flash_attention
#    # --resume_from_checkpoint ...


#LoRA with 8bit
torchrun --nproc_per_node 4 src/entry_point/sft_train.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora \
    --use_int8_training \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 8e-6 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    # --resume_from_checkpoint ${output_dir}/checkpoint-336
    # --use_flash_attention

# LoRA without 8bit
# torchrun --nproc_per_node 4 src/entry_point/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --use_lora \
#     --deepspeed configs/deepspeed_config_stage3.json \
#     --lora_config configs/lora_config_llama.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 3e-4 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.01 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#     --overwrite_output_dir
#    # --use_flash_attention
#    # --resume_from_checkpoint ...

