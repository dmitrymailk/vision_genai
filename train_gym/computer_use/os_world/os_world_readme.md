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
