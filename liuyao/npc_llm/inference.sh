RUN_ROOT=/maindata/data/user/ai_story/yao.liu/user_model

python3 -m fastchat.serve.cli \
  --model-path $RUN_ROOT/Mixtral_8x7B_Instruct_V0.1_SFT_SEQ4096_LR1e-6_EP4_GBS16x1x1_NEFT5_20240328_v1.1/checkpoints/checkpoint-378 \
  --conv-template custom_blank \
  --num-gpus 2
