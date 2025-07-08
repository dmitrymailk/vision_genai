- transformers==4.53.1
- accelerate==1.8.1

### huggingface transfomers
дефолтный колбек который решает когда логгировать
- https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/trainer_callback.py#L583
место где этот колбек вызывается
- https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/trainer.py#L2622
место где происходит отправка лосса в wandb (там мы делим лосс на количество шагов с последней отправки(лосс при этом суммируется))
- https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/trainer.py#L3068
подготовка и оборачивание датасета в accelerate в классе trainer
- https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/trainer.py#L1039
класс который отвечаем за функцию семплинга в dataloader
- https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/trainer.py#L1057
- https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/trainer.py#L992

### accelerate
что происходит с dataloader когда мы вызываем конструкцию accelerator.prepare()
- вызов _prepare_one
- https://github.com/huggingface/accelerate/blob/v1.8.1/src/accelerate/accelerator.py#L1433
- https://github.com/huggingface/accelerate/blob/v1.8.1/src/accelerate/accelerator.py#L2416
- создание нового класса семплера SeedableRandomSampler
- https://github.com/huggingface/accelerate/blob/v1.8.1/src/accelerate/data_loader.py#L1170
- SeedableRandomSampler
- https://github.com/huggingface/accelerate/blob/v1.8.1/src/accelerate/data_loader.py#L73
- но почему trainer от hf по умолчанию решает создать такой семплер? он включает по дефолту параметр use_seedable_sampler, c дополнительными параметрами, которые становятся параметрами dataloader_config
- https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/trainer.py#L5170
- дока на accelerate https://huggingface.co/docs/accelerate/en/package_reference/accelerator#accelerate.Accelerator.dataloader_config

Все это по итогу влечет за собой расхождения лоссов и градиентов при тренировке на одинаковых датасетах. Просто логгируется лосс для разных наборов данных. А из-за того что коллбек отслеживает логгирование лосса на основе глобального шага global_step % logging_steps == 0, лосс также начнет различаться на второй эпохе, если вдруг логирование в своем цикле будет на основе локальных шагов даталоадера. Как же это решить.

Первое.
```python
parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, TrainingArguments)
)
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

dataloader_params = [
    "split_batches",
    "dispatch_batches",
    "even_batches",
    "use_seedable_sampler",
]

dataloader_config = DataLoaderConfiguration(
  **{
      param: training_args.accelerator_config.pop(param)
      for param in dataloader_params
  }
)
accelerator = Accelerator(
    dataloader_config=dataloader_config,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    **accelerator_log_kwargs,
)
```
Или просто своровать отсюда
- https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/trainer.py#L5170

Второе. Делать логгирование на основе глобального шага как это указано тут
- https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/trainer_callback.py#L583
- ps. мне такое не очень нравится, так как получается некоторое сглаживание, которое просто зависит от того как часто я спрашиваю какой лосс на данном шаге.


#### Базовый скрипт на основе accelerate
- [accelerate_example.py](./accelerate_example.py)

#### Базовый скрипт на основе huggingface trainer
- [hf_trainer_example.py](./hf_trainer_example.py)