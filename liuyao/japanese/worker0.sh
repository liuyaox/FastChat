CUDA_VISIBLE_DEVICES=0 \
python3 -m fastchat.serve.vllm_worker \
    --model-name japanese-novel-gpt-j-6b \
    --model-path /dfs/comicai/nlp_models/Japanese/AIBunCho/japanese-novel-gpt-j-6b \
    --host 0.0.0.0 \
    --worker-address http://0.0.0.0:21007 \
    --port 21007 \
    --controller http://0.0.0.0:21001 \
    --num-gpus 1 \
    --max-model-len 4096
