- https://github.com/bytedance/UI-TARS-desktop
- https://github.com/bytebot-ai/bytebot
- https://github.com/browser-use/browser-use
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

### UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv --python 3.11
```

### Install docker in docker

sudo apt-get install ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

VERSION_STRING=5:24.0.2-1~ubuntu.22.04~jammy

sudo apt-get install docker-ce=$VERSION_STRING docker-ce-cli=$VERSION_STRING containerd.io docker-buildx-plugin docker-compose-plugin

sudo apt-mark hold docker-ce docker-ce-cli

sudo chmod 777 /var/run/docker.sock


sudo docker run hello-world