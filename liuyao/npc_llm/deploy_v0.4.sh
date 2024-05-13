export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.4.1.1-mixtral8x7b-instruct-sft-ckpt61 \
    --model-path $RUN_ROOT/Mixtral_8x7B_Instruct_V0.1_SFT_LR5e-6_EPOCH4_BS1_SEQ4096_PROC32_20240312/checkpoints/checkpoint-61 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21016 \
    --port 21016 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 2

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.4.1.1-mixtral8x7b-instruct-sft-ckpt122 \
    --model-path $RUN_ROOT/Mixtral_8x7B_Instruct_V0.1_SFT_LR5e-6_EPOCH4_BS1_SEQ4096_PROC32_20240312/checkpoints/checkpoint-122 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21017 \
    --port 21017 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 2


export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.4.1.1-mixtral8x7b-instruct-sft-ckpt183 \
    --model-path $RUN_ROOT/Mixtral_8x7B_Instruct_V0.1_SFT_LR5e-6_EPOCH4_BS1_SEQ4096_PROC32_20240312/checkpoints/checkpoint-183 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21016 \
    --port 21016 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 2
