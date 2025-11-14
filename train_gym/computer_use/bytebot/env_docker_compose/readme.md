
git clone https://github.com/bytebot-ai/bytebot.git && cd bytebot
cd /code/train_gym/computer_use/bytebot/env_docker_compose/bytebot
docker compose -f docker/docker-compose.proxy.yml up
docker compose -f docker/docker-compose.proxy.yml exec bytebot-agent sh
docker compose -f docker/docker-compose.proxy.yml up bytebot-agent --build
docker compose -f docker/docker-compose.proxy.yml up bytebot-llm-proxy --build