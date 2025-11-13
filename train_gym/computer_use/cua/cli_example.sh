export OPENAI_API_KEY="hf_xxx"
export OPENAI_API_BASE="http://localhost:1337/v1"
# trycua/cua-xfce:latest
# uv run --with "cua-agent[cli]" -m agent.cli openai/computer-use-preview --provider docker
uv run --with "cua-agent[cli]" -m agent.cli huggingface-local/ByteDance-Seed/UI-TARS-1.5-7B --provider docker