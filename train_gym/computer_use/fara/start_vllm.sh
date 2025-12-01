MODEL_CACHE_PATH=/home/dimweb/vision_genai/train_gym/computer_use/ui_tars/data
VLLM_IMAGE=vllm/vllm-openai:v0.11.0
MODEL_NAME=microsoft/Fara-7B

   
docker run --rm --runtime nvidia \
    -v $MODEL_CACHE_PATH:/root/.cache/huggingface/hub \
    -p 1337:8000 \
    --ipc=host \
    $VLLM_IMAGE \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --model $MODEL_NAME \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --served-model-name gpt-4o-2024-08-06 \
    --max-model-len 44768 \
    # --enable-auto-tool-choice \
    # --tool-call-parser hermes \
    # --reasoning-parser deepseek_r1
    # --tool-call-parser qwen3_xml \

    # weather in LA
