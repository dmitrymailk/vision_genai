# Massive models train



## [Introducing the First AMD 1B Language Models: AMD OLMo](https://www.amd.com/en/developer/resources/technical-articles/introducing-the-first-amd-1b-language-model.html)
- 1.2B params
- Context length 2048, vocab 50280
- 64 AMD Instinct MI250 GPUs
- pre-trained with 1.3 trillion tokens ([Dolma v1.7](https://huggingface.co/datasets/allenai/dolma) dataset) on 16 nodes with 4 [AMD Instinct™ MI250](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250.html) GPUs
- 7.66B tokens in SFT
- результат по метрикам совпадает со всеми моделями из списка, в целом они все совпадают  [TinyLLaMA-v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1) (1.1B), [MobiLLaMA-1B](https://huggingface.co/MBZUAI/MobiLlama-1B) (1.2B), [OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf) (1.2B), [OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf) (1.2B), and [OpenELM-1_1B](https://huggingface.co/apple/OpenELM-1_1B) (1.1B).


## [MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT](https://arxiv.org/pdf/2402.16840)
- 1.2B params
- 20 GPU nodes 8 NVIDIA A100 GPUs with 80 GB memory each and 800 Gbps interconnect for model training. Each GPU is interconnected through 8 NVLink links, complemented by a cross-node connection configuration of 2 port 200 Gb/sec (4×HDR) InfiniBand
- 14k-15k tokens per second on a single GPU
- checkpoints after every 3.3B tokens
- 2048 tokens long max length 
- vocabulary size of 32,000
- pretraining, we use 1.2T tokens from LLM360 Amber dataset
- https://github.com/mbzuai-oryx/MobiLlama

## [TinyLlama-1.1B](https://github.com/jzhang38/TinyLlama)
- wandb loss https://wandb.ai/lance777/lightning_logs/reports/metric-train_loss-23-09-04-23-38-15---Vmlldzo1MzA4MzIw
- отчет о багах по загрузке данных(разные ноды учились на одних и тех же данных) (fixed) https://github.com/jzhang38/TinyLlama/issues/67 https://whimsical-aphid-86d.notion.site/Release-of-TinyLlama-1-5T-Checkpoints-Postponed-01b266998c1c47f78f5ae1520196d194
- баг из-за отсутствия маскирования BOS, который к тому же сильно заполнял датасет при обучении (fixed) https://github.com/jzhang38/TinyLlama/issues/83
- 1.1B params 
- 3 trillion tokens train  (slightly more than 3 epochs/1430k steps)
- 16 A100-40G GPUs
- 90 days training (не правда на самом деле)
- Sequence Length	2048
- Grouped Query Attention
- Layers: 22, Heads: 32, Query Groups: 4, Embedding Size: 2048, Intermediate Size (Swiglu): 5632
- Batch Size	2 million tokens (2048 * 1024)
- Training Data [Slimpajama](https://huggingface.co/datasets/cerebras/slimpajama-627b) & [Starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)
- throughput of 24k tokens per second per A100-40G GPU
- It means you can train a chinchilla-optimal TinyLlama (1.1B param, 22B tokens) in 32 hours with 8 A100.
- 56% model flops utilization (MFU)
- 5*10**11/24_000/16/60/60/24=15.07 days
- 1*10**12/24_000/16/60/60/24=30.14 days
- 3*10**12/24_000/16/60/60/24=90.42 days
- 14 days => 24_000 * 16 * 86_400 * 14=464_486_400_000 ~ 464B tokens

Согласно их таблице эвалюации, улучшения в зависимости от токенов были следующие
- 105B	50k	46.11
- 503B	240K	48.28
- 1T	480k	50.22
- 2T	955k	51.64
- 2.5T	1195k	53.86
- 3T	1431k	52.99
- Таблица метрик в зависимости от чекпоинтов https://github.com/jzhang38/TinyLlama/blob/main/EVAL.md
Согласно их результатам, средний рост по всем метрикам незначительный, зато компьюта в разы больше. 500B=0.48, 1000B=50.22, 1500B=51.29.
500B ждать 15 days, зато 1500B ждать уже 45 days.  

Тут не понятно после какого момента модель достигает MVP, потому что так-то можно и 18Т ждать(как делают qwen), чтобы получить лучшую модель, но если заниматься ресерчем где нужно проверять и сравнивать гипотезы, то какое значение метрик для дефолтного трансформера свидетельствует о том что модель достигла MVP? Дальше в соответствие уже с этим нужно тренить, допустим что 1Т для дефолтного значит следующие архитектуры должны сходится быстрее и достигать более высоких метрик за этот же срок(например такая логика). 

### Optimizations
- multi-gpu and multi-node distributed training with FSDP.
- flash attention 2.
- fused layernorm.
- fused swiglu.
- fused cross entropy loss .
- fused rotary positional embedding.

#### A100 GPU hours taken on 300B tokens
- TinyLlama-1.1B -> 3456
- [Pythia-1.0B](https://huggingface.co/EleutherAI/pythia-1b) -> 4830
- [MPT-1.3B](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b) -> 7920

The Pythia number comes from their [paper](https://arxiv.org/abs/2304.01373). The MPT number comes from [here](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b), in which they say MPT-1.3B " was trained on 440 A100-40GBs for about half a day" on 200B tokens.

## [Introducing Instella: New State-of-the-art Fully Open 3B Language Models](https://rocm.blogs.amd.com/artificial-intelligence/introducing-instella-3B/README.html)

- 3B params
- 128 Instinct MI300X GPUs
- total 4.15T tokens
- Pre-training (Stage 1) 4.065 Trillion
- Pre-training (Stage 2) 57.575 Billion
- SFT 8.902 Billion (x3 epochs)
- DPO 760 Million
- 36 decoder layers
- 4096 max length
- 
- github repo https://github.com/AMD-AIG-AIMA/Instella

## [2 OLMo 2 Furious](https://arxiv.org/pdf/2501.00656)
#### 7B model
- wandb logs https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/OLMo-2-7B-Nov-2024--VmlldzoxMDUzMzE1OA
- wandb complete logs https://wandb.ai/ai2-llm/OLMo-2-1124-7B?nw=nwuserakshitab
- 7B params
- 3.895T for pretraining 
- max_sequence_length: 4096
- Layers 32, Hidden Size 4096, Attention Heads 32
- Batch Size 1024
- 4T for pretraining 
- ~5.6k tokens/per/second/gpu
- 0.68 batches/per/second/gpu

Исходя из этих 2 запусков: [запуск_1](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/runs/awwjyi5w/overview), [запуск_2](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/runs/uwjy7cji/overview?nw=nwuserakshitab). Они длились по 7h 43m 54s и 9h 29m 4s, в сумме 7B претрен модель была получена за    
7*60*60+43*60+54+9*60*60+29*60+4=61978 sec ~ 17.21 hours, всего через нее прошло 3_895_023_632_384 токенов
исходя из пропускной способности 3_895_023_632_384/61978=62845261.7 tokens/sec/total
- 62845261.7/5600=11222.36 карт(что бред.)

- 3_895_023_632_384/512/5600/60/60/24=15.723 days, я не знаю, не может быть чтобы с такими вводными модель натренилась за 17 часов. но в wandb в сумме 17.

- global_train_batch_size:1024, device_train_batch_size:2,device_train_grad_accum:1 => 512 GPU H100 (исходя из их конфига претрен был на 512 gpu)
- 512 * 5600=2867200 tokens/sec/total
- 512 * 5600 * 61978=177_703_321_600 ~ 178B tokens

Имея всего 1 такой кластер на 8 H100, мы пройдем
- 1 days => 86_400 * 1 * 5_600 * 8=3_870_720_000 ~ 3.870B tokens
- 2 days => 3.87 * 2 ~ 7.74B tokens
- 7 days => 3.87 * 7 ~ 27.09B tokens
- 14 days => 3.87 * 7 ~ 54.18B tokens

Анализ динамики обучения 7B модели(https://wandb.ai/ai2-llm/OLMo-2-1124-7B/runs/awwjyi5w)
- (7 days) 27_447_525_376 (6544 steps), **winogrande_acc**=0.534(6000 step),**hellaswag_len_norm**=0.506,**mmlu_other_var_len_norm**=0.381 
- (14 days) 54_777_610_240 (13060 steps), **winogrande_acc**=0.586(13_000 step),**hellaswag_len_norm**=0.605,**mmlu_other_var_len_norm**=0.44, (0.586+0.605+0.44)/3=0.5436

Если судить только по метрикам, за 7 дней претрен с 7B моделью получается хуже чем с 1B. за 14 дней лучше 7B модель но не сильно 0.5436 против 0.533.

#### 1B model
- скорее всего тренировка была на кластере Augusta, у которого максимум 160 node, 8 H100 per node=1280 GPUS
- model train config https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml
- wandb logs https://api.wandb.ai/links/ai2-llm/izdtrtu0
- 1.4B params (больше чем у https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)
- 4T for pretraining 
- ~35k tokens/per/second/gpu
- 4 batches/per/second/gpu
- max_sequence_length: 4096
- vocab_size: 100278
- We trained the 1B model on 128 H100 GPUs, across 16 nodes. The training time itself took just less than 9 days (8 days, ~17 hours) https://github.com/allenai/OLMo/issues/861#issuecomment-3084658291 => 8 * 24=192h => 192+17=209h => 209 * 60 * 60=752_400 sec
- 128 * 35_000=4_480_000 toks/sec => 752_400 * 4_480_000=3_370_752_000_000
- числа немного не сходятся с wandb и тем что они говорят на 4*10**12/35_000/128/60/60/24=10.33, если брать скорость их wandb то им бы потребовалось чуть больше 10 дней

Имея всего 1 такой кластер на 8 H100, мы пройдем
- 1 day => 86_400 * 1 * 35_000 * 8=24_192_000_000 ~ 24.192B tokens
- 2 day => 24.192 * 2 ~ 48.384B tokens
- 7 day => 24.192 * 7 ~ 169.344B tokens
- 8 days, ~17 hours => 752_400 * 35_000 * 8=210_672_000_000  ~ 211B tokens
- 14 days => 86_400 * 14 * 35_000 * 8=338_688_000_000 ~ 340B
- 30 days => 86_400 * 30 * 35_000 * 8=725_760_000_000 ~ 725B tokens

Анализ динамики обучения 1B модели
- (1 day) 24_383_586_304 tokens (11_627 step),**winogrande_acc**=0.524(12_000 step),**hellaswag_len_norm**=0.446,**mmlu_other_var_len_norm**=0.365
- (2 day) 48_020_586_496 tokens (22_898 step),**winogrande_acc**=0.550(23_000 step),**hellaswag_len_norm**=0.497,**mmlu_other_var_len_norm**=0.390
- (7 day) 169_869_312_000 tokens (81_000 step),**winogrande_acc**=0.585(80_000 step),**hellaswag_len_norm**=0.576,**mmlu_other_var_len_norm**=0.407
- (14 day) 341_542_174_720 tokens (162860 step),**winogrande_acc**=0.60 (160_000 step),**hellaswag_len_norm**=0.59 ,**mmlu_other_var_len_norm**=0.409, (0.60+0.59+0.409)/3=0.533

Однако в их статье tokens per second намного выше, ~55k(доказательств кроме статьи не нашел, возможно это когда нет никаких эвалюаций)
#### Дебаг репы для обучения

После попыток запусков изначальной [ссылка на конфиг](https://github.com/dmitrymailk/OLMo/blob/04820704616af5d25cdba4df23aa7b4d9ce86cad/configs/official-0425/OLMo2-1B-stage1.yaml), не получилось запустить обучение 1B модели с контекстом 1024 на batch_size=1 именно для 1B модели, пришлось сократить количество декодер блоков до с n_layers: 16 до 8. И даже так на трейне получается что 23GB памяти задействовано. Это слишком много памяти, надо фиксить. Также есть проблемы с обычным трейно из-за wandb для обычного юзера. [Версия со всеми фиксами](https://github.com/dmitrymailk/OLMo/blob/b690ffcb1b998af7437c740d41419eecd081b406/configs/official-0425/OLMo2-1B-stage1.yaml).

Оказалось что их модель вовсе не 1B параметров, а 1.4B(1_484_916_736), как раз столько дают 16 слоёв. После того как поменял n_layers: 16 до 12, получилось (1_216_448_512) это немного меньше чем у unsloth/Llama-3.2-1B-Instruct (1_235_814_400). C таким конфигом обучается с max_sequence_length: 1024

Обучение с FSPD на одной карте RTX4090 дает ~1500 tokens/sec, DDP вызывает OOM с 2 и 4 батчем, память скачет с 17GB до 22GB. Однако обычное обучение с accelerate(там используется ни DDP, ни FSDP по умолчанию. Просто autocast to_fp32 https://github.com/huggingface/accelerate/blob/v1.10.0/src/accelerate/accelerator.py#L1755) дает 12700 tok/sec (со всеми оптимизациями ~23400 tok/sec) я дебажу на компьютере с 1 картой.  Но мне кажется это очень медленным. Хотя я сократил длину до 1024 и батча 4. 

Также я не понял почему лосс вычисляется следующим образом(наверное это ошибка в усреднении).
1. В hf была такая ошибка, но они ее исправили. Сейчас там сначала считают количество элементов которые не равны -100 https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/trainer.py#L5377
2. Затем данное число передают в лосс чтобы на него поделить, если reduction sum https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/loss/loss_utils.py#L31

В OLMO репозитории 
1. Выполняют .numel() от тензора батча, что возвращает просто количество элементов, без учета тех что мы игнорим. https://github.com/allenai/OLMo/blob/main/olmo/train.py#L785
2. Вычисляет cross entropy с учетом игнорирования индекса https://github.com/allenai/OLMo/blob/main/olmo/train.py#L749
3. Но затем делит на полную сумму https://github.com/allenai/OLMo/blob/main/olmo/train.py#L765

Забавный факт что gemini 2.5 pro нашел эту же ошибку. Я скопировал туда весь файл с трейнером и сказал найди критические ошибки. Сначала он сказал что главная ошибка это вычисление глобального лосса только когда мы логгируем в wandb, типа это создает расхождения. Но затем когда я явно написал найди ошибки в лоссе, он написал ровно то что я дебажил пару часов(но он это нашел за 30 сек мда, Qwen3-235B-A22B-2507 тоже со второй попытки с размышлением).

## [SmolLM3](https://hf.co/blog/smollm3)
- https://github.com/huggingface/smollm/tree/main/text
- wandb train logs https://wandb.ai/huggingface/SmolLM3-training-logs?nw=nwusereliebak
- 3B params
- 384 H100 GPUs for 24 days
- MFU 29%
- 12-13k tokens/sec/gpu
- seq len 4096 pretraining
- batch 2.36 M
- grad accum 1
- micro batch size 3
- precision bf16
- tensor parallel 2
- data perallel 192
- 11.2T tokens total
- 8T tokens pretrain
- 2T tokens Mid-training
- 1.1T - rl finetune
- 4k to 64k with RoPE in Mid-training 

- сначала делали претрен на 4096 токенах, затем растягивали 50B tokens на 32к, потом еще 50B на 64к, yarn добили до 128к.
- нет выложенных графиков, нельзя предсказать динамику обучения.
- нет информации о экспериментах на 1B моделях(странно)

#### Дебаг репы для обучения
Так как версии либ не зафиксированы, нужно время чтобы угадать какая комбинация библиотек позволит репе заработать.

Конфиги для локального обучения tiny_llama не работают, несколько PR request которые фиксят это лежат с прошлого года. Ничего не фиксится и не работает. Конфиги для обучения smollm3 бесполезны на локальном железе, так как даже датасеты которые они выложили запривачены и к ним доступа. Нужно токенизировать и создавать свои датасеты для обучения. Также они намешали в код работу с s3 бд, но мне не понятно как работать с s3 датасетами и как они туда попадают.

Тоже самое происходит и с конфигами smollm2, данные не открыли, как их готовить тоже не предоставили [issues SmolLM2 Pretrain Dataset #35](https://github.com/huggingface/smollm/issues/35)

## 1B-4B text models
- https://github.com/deepseek-ai/Janus
- https://huggingface.co/Menlo/Jan-nano