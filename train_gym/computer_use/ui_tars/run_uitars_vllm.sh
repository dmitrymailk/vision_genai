MODEL_CACHE_PATH=$PWD/data
# VLLM_IMAGE=vllm/vllm-openai:v0.11.0
# VLLM_IMAGE=vllm/vllm-openai:v0.10.2 
VLLM_IMAGE=vllm/vllm-openai:nightly 
MODEL_NAME=ByteDance-Seed/UI-TARS-1.5-7B

# docker run --rm -d --runtime nvidia \
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
    --max-model-len 32000 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name computer-use-preview
    # --served-model-name gpt-4o
    # --tool-call-parser pythonic
