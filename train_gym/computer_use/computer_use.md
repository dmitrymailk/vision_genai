- https://github.com/bytebot-ai/bytebot
- 

### FIX docker in docker

```console
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.49/containers/json": dial unix /var/run/docker.sock: connect: permission denied
```
```bash
sudo chmod 777 /var/run/docker.sock
```

### Ollama pull model
- внутри докера работают почти все функции по запуску докера, однако не работают volumes. он видимо не понимает пути внутри докера. ему либо нужно прописывать настоящие пути из системы, либо запускать снаружи такие скрипты. в основном постоянные волюмы нужны чтобы не скачивать модели после каждого запуска контейнера vllm или ollama.
```bash
curl http://localhost:11434/api/pull -d '{
  "model": "qwen3-vl:8b"
}'
```
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-vl:8b",
  "prompt": "Why is the sky blue?"
}'
```