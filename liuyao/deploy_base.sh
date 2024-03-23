# Mistral-7B
export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0 \
python3 -m fastchat.serve.vllm_worker \
    --model-name Mistral-7B-v0.1 \
    --model-path $DEPT_HOME/nlp_models/mistralai/Mistral-7B-v0.1 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21010 \
    --port 21010 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 1

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name Mistral-7B-Instruct-v0.2 \
    --model-path $DEPT_HOME/nlp_models/mistralai/Mistral-7B-Instruct-v0.2 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21011 \
    --port 21011 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 1

# Mixtral-8x7B
export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name Mixtral-8x7B-v0.1 \
    --model-path $DEPT_HOME/nlp_models/mistralai/Mixtral-8x7B-v0.1 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21016 \
    --port 21016 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 2

export DEPT_HOME=/maindata/data/user/ai_story
export RUN_ROOT=$DEPT_HOME/yao.liu/npc_llm
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m fastchat.serve.vllm_worker \
    --model-name Mixtral-8x7B-Instruct-v0.1 \
    --model-path $DEPT_HOME/nlp_models/mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21016 \
    --port 21016 \
    --controller http://0.0.0.0:21001  \
    --num-gpus 2

