export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.3.1.1-mixtral8x7b-sft-ckpt60 \
    --model-path $RUN_ROOT/Mixtral_8x7B_V0.1_SFT_LR5e-6_EPOCH4_BS1_SEQ4096_PROC32_20240312/checkpoints/checkpoint-60 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21010 \
    --port 21010 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 2

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.3.1.1-mixtral8x7b-sft-ckpt120 \
    --model-path $RUN_ROOT/Mixtral_8x7B_V0.1_SFT_LR5e-6_EPOCH4_BS1_SEQ4096_PROC32_20240312/checkpoints/checkpoint-120 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21012 \
    --port 21012 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 2

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.3.1.1-mixtral8x7b-sft-ckpt180 \
    --model-path $RUN_ROOT/Mixtral_8x7B_V0.1_SFT_LR5e-6_EPOCH4_BS1_SEQ4096_PROC32_20240312/checkpoints/checkpoint-180 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21014 \
    --port 21014 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 2

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.3.1.1-mixtral8x7b-sft-ckpt240 \
    --model-path $RUN_ROOT/Mixtral_8x7B_V0.1_SFT_LR5e-6_EPOCH4_BS1_SEQ4096_PROC32_20240312/checkpoints/checkpoint-240 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21014 \
    --port 21014 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 2

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.3.1.1-mixtral8x7b-sft-ckpt300 \
    --model-path $RUN_ROOT/Mixtral_8x7B_V0.1_SFT_LR5e-6_EPOCH4_BS1_SEQ4096_PROC32_20240312/checkpoints/checkpoint-300 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21016 \
    --port 21016 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 2
