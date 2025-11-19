# https://github.com/ollama/ollama/issues/12025#issuecomment-3221932677
# need to delete delete /usr/lib/ollama/cuda_v* in docker
docker run --rm \
  -v ./ollama:/root/.ollama \
  -p 11434:11434 \
  -e OLLAMA_HOST=0.0.0.0 \
  -e OLLAMA_CONTEXT_LENGTH=120000 \
  -e OLLAMA_ORIGINS="*" \
  ollama/ollama:0.12.9