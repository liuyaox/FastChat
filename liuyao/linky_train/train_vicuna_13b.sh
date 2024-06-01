MODEL_PATH=/mnt/data/gyl/pretrained_models/vicuna-13b-v1.5
DATA_PATH=/mnt/data/gyl/data/koala-sft/linky_v1/train.json
EVAL_DATA_PATH=/mnt/data/gyl/data/koala-sft/linky_v1/dev.json
OUTPUT_DIR=/mnt/data/gyl/checkpoints/koala_vicuna1.5_13b_linky_v1

# 13b-v1.5 vicuna-v0.3
torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --report_to "wandb" \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --data_format sft