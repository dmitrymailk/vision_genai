docker run \
	--gpus '"device=0"' \
	--shm-size 20g \
	-v ./data/:/models \
	-p 8000:8000 ghcr.io/ggml-org/llama.cpp:server-cuda-b7045  \
	-m /models/Qwen3VL-8B-Instruct-Q8_0.gguf \
	--port 8000 \
	--host 0.0.0.0 \
	--n-gpu-layers 29 \
	-np 8