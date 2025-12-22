### сделать результаты эвалюации только для чтения, защита от изменения или удаления

1. Не очень надежно, vs code в docker легко сделает нужный файл для записи
```bash
chmod -R 555 +i /code/train_gym/computer_use/os_world/OSWorld/result_no_chrome
```
2. Надежно, даже sudo не может случайно удалить данный файл.
```bash
sudo chattr -R +i /code/train_gym/computer_use/os_world/OSWorld/result_no_chrome
```
чтобы разблокировать можно написать
```bash
sudo chattr -R -i /code/train_gym/computer_use/os_world/OSWorld/result_no_chrome
```

### Старт агента вместе с lm studio

```bash
# example of simple env
# python quickstart.py --provider_name docker --os_type Ubuntu

export OPENAI_API_KEY='changeme'
# export OPENAI_BASE_URL='http://172.17.0.1:11434/v1'
# export OPENAI_BASE_URL='http://172.17.0.1:1337/v1'
# export OPENAI_BASE_URL='http://192.168.120.210:1234/v1'
export OPENAI_BASE_URL='http://0.0.0.0:1234/v1'
# export OPENAI_BASE_URL='http://192.168.120.210:30000/v1'
# export vm_ip='172.17.0.1'
# model_name=qwen3-vl:8b
model_name=gpt-4o
# создать клон с новым именем
# ollama cp qwen3-vl:8b gpt-4o

# python run_multienv.py \
#     --provider_name docker \
#     --headless \
#     --observation_type screenshot \
#     --model gpt-4o \
#     --sleep_after_execution 3 \
#     --max_steps 15 \
#     --num_envs 1 \
#     --client_password password
task_path=evaluation_examples/test_nochrome.json
output_path=result/debug
python run_multienv_qwen3vl.py \
    --provider_name docker \
    --model $model_name \
    --test_all_meta_path $task_path \
    --result_dir $output_path
# docker stop $(docker ps -q) && docker rm $(docker ps -a -q)
```