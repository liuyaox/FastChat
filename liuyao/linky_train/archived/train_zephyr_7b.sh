 MODEL_PATH=/dfs/comicai/yiming.wang/models_download/zephyr-7b-beta
 DATA_PATH=/dfs/comicai/yao.liu/repo/linky_rp_data/linky_rp_merge/data/pretrain/train.json
# EVAL_DATA_PATH=/dfs/comicai/yao.liu/repo/linky_rp_data/linky_rp_merge/data/pretrain/dev.json
 OUTPUT_DIR=/dfs/comicai/yao.liu/checkpoints/linky_llm_v2/zephyr_7b_pretrian_v0.1


 torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train_mem.py \
     --model_name_or_path $MODEL_PATH \
     --data_path $DATA_PATH \
     --output_dir $OUTPUT_DIR \
     --num_train_epochs 10 \
     --learning_rate 2e-7 \
     --lr_scheduler_type "cosine" \
     --weight_decay 0. \
     --warmup_ratio 0.04 \
     --per_device_train_batch_size 2 \
     --gradient_accumulation_steps 4 \
     --save_strategy "epoch" \
     --save_total_limit 10 \
     --gradient_checkpointing True \
     --model_max_length 2048 \
     --report_to "wandb" \
     --logging_steps 1 \
     --fsdp "full_shard auto_wrap offload" \
     --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
     --bf16 True \
     --lazy_preprocess True \
     --data_format pretrain

# torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train_mem.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_path $DATA_PATH \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs 3 \
#     --learning_rate 2e-6 \
#     --lr_scheduler_type "cosine" \
#     --weight_decay 0. \
#     --warmup_ratio 0.04 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --eval_data_path $EVAL_DATA_PATH \
#     --per_device_eval_batch_size 4 \
#     --evaluation_strategy "steps" \
#     --eval_steps 50 \
#     --save_strategy "epoch" \
#     --save_total_limit 8 \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --report_to "wandb" \
#     --logging_steps 5 \
#     --fsdp "full_shard auto_wrap offload" \
#     --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
#     --bf16 True \
#     --tf32 True \
#     --lazy_preprocess True \
#     --data_format pretrain

# torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train_mem.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_path $DATA_PATH \
#     --eval_data_path $EVAL_DATA_PATH \
#     --output_dir $OUTPUT_DIR \
#     --report_to "wandb" \
#     --bf16 True \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "steps" \
#     --eval_steps 50 \
#     --save_strategy "epoch" \
#     --save_total_limit 8 \
#     --learning_rate 2e-6 \
#     --weight_decay 0. \
#     --warmup_ratio 0.04 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap offload" \
#     --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
#     --tf32 True \
#     --model_max_length 4096 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --data_format pretrain
