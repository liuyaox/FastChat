export FASTCHAT_WORKER_HEART_BEAT_INTERVAL=10
export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/girl_friendly
MODEL_PATH=${RUN_ROOT}/Mixtral_8x7B_Instruct_V0.1_2SFT_LR5e-6_EPOCH3_GBS16_SEQ4096_PROC16_20240323_v-15.1_then_v1merged/checkpoints/checkpoint-496

CUDA_VISIBLE_DEVICES="0,1" \
${DEPT_HOME}/shijund/miniconda3/envs/fastchat/bin/python -m fastchat.serve.vllm_worker \
  --model-name xxx \
  --conv-template mistral \
  --model-path $MODEL_PATH \
  --host 0.0.0.0 \
  --worker-address http://0.0.0.0:21010 \
  --port 21010 \
  --controller http://0.0.0.0:21001 \
  --num-gpus 2
