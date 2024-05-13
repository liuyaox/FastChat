export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.2-mistral7b-instruct-cpt-ckpt1000 \
    --model-path $RUN_ROOT/Mistral_7B_Instruct_V0.2_CPT_DP24_VPPNone_ACC32_MBSZ2_GBSZ1536_SEQLEN4096_ITERS8000_20240312/checkpoints/iter_0001000 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21010 \
    --port 21010 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 1

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.2-mistral7b-instruct-cpt-ckpt2000 \
    --model-path $RUN_ROOT/Mistral_7B_Instruct_V0.2_CPT_DP24_VPPNone_ACC32_MBSZ2_GBSZ1536_SEQLEN4096_ITERS8000_20240312/checkpoints/iter_0002000 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21011 \
    --port 21011 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 1

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=2 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.2-mistral7b-instruct-cpt-ckpt3000 \
    --model-path $RUN_ROOT/Mistral_7B_Instruct_V0.2_CPT_DP24_VPPNone_ACC32_MBSZ2_GBSZ1536_SEQLEN4096_ITERS8000_20240312/checkpoints/iter_0003000 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21012 \
    --port 21012 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 1

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=3 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.2-mistral7b-instruct-cpt-ckpt4000 \
    --model-path $RUN_ROOT/Mistral_7B_Instruct_V0.2_CPT_DP24_VPPNone_ACC32_MBSZ2_GBSZ1536_SEQLEN4096_ITERS8000_20240312/checkpoints/iter_0004000 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21013 \
    --port 21013 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 1

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=4 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.2-mistral7b-instruct-cpt-ckpt5000 \
    --model-path $RUN_ROOT/Mistral_7B_Instruct_V0.2_CPT_DP24_VPPNone_ACC32_MBSZ2_GBSZ1536_SEQLEN4096_ITERS8000_20240312/checkpoints/iter_0005000 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21014 \
    --port 21014 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 1

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=5 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.2-mistral7b-instruct-cpt-ckpt6000 \
    --model-path $RUN_ROOT/Mistral_7B_Instruct_V0.2_CPT_DP24_VPPNone_ACC32_MBSZ2_GBSZ1536_SEQLEN4096_ITERS8000_20240312/checkpoints/iter_0006000 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21015 \
    --port 21015 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 1

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=6 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.2-mistral7b-instruct-cpt-ckpt7000 \
    --model-path $RUN_ROOT/Mistral_7B_Instruct_V0.2_CPT_DP24_VPPNone_ACC32_MBSZ2_GBSZ1536_SEQLEN4096_ITERS8000_20240312/checkpoints/iter_0007000 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21016 \
    --port 21016 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 1

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=7 \
python3 -m fastchat.serve.vllm_worker \
    --model-name npcllm-v0.2-mistral7b-instruct-cpt-ckpt8000 \
    --model-path $RUN_ROOT/Mistral_7B_Instruct_V0.2_CPT_DP24_VPPNone_ACC32_MBSZ2_GBSZ1536_SEQLEN4096_ITERS8000_20240312/checkpoints/iter_0008000 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21017 \
    --port 21017 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 1
