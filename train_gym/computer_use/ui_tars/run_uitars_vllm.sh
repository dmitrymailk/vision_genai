MODEL_CACHE_PATH=./data
VLLM_IMAGE=vllm/vllm-openai:v0.11.0
# VLLM_IMAGE=vllm/vllm-openai:v0.10.2 
# VLLM_IMAGE=vllm/vllm-openai:nightly 
# MODEL_NAME=ByteDance-Seed/UI-TARS-1.5-7B
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct

# docker run --rm -d --runtime nvidia \
    # -v ./vllm_cache/vllm:/vllm_cache \
docker run --rm --runtime nvidia \
    -v $MODEL_CACHE_PATH:/root/.cache/huggingface/hub \
    -v ./vllm_cache/vllm:/usr/local/lib/python3.12/dist-packages/vllm \
    -p 1337:8000 \
    --ipc=host \
    $VLLM_IMAGE \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --model $MODEL_NAME \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-model-len 32000 \
    --enforce-eager \
    --served-model-name gpt-4o \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    # --tool-call-parser qwen3_xml \

    # weather in LA
