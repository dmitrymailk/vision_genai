uv venv --python 3.11

source .venv/bin/activate

uv pip install -e .

sudo .venv/bin/playwright install-deps

<!-- fara-cli --task "whats the weather in new york now" --endpoint_config endpoint_configs/azure_foundry_config.json -->
<!-- fara-cli --task "find latest news in volgograd" --endpoint_config endpoint_configs/azure_foundry_config.json -->
fara-cli --task "how many pages does wikipedia have" --endpoint_config endpoint_configs/azure_foundry_config.json


magentic-ui --port 8081 --config magentic_config.yaml

### Install magentic from source

uv venv --python=3.12 .venv

source .venv/bin/activate

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

uv pip install magentic_ui==0.1.5

magentic-ui --fara --port 8081 --config magentic_config.yaml