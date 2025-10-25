### DDP, accelerate
- unsloth/Llama-3.2-1B-Instruct,block_size=2048,max_batch=8,opt_2(liger_kernel),5090 palit - 19280 tok/sec, batch=5 - 17064 tok/sec
- unsloth/Llama-3.2-1B-Instruct,block_size=2048,max_batch=5,opt_2(liger_kernel),4090 aero - 12753 tok/sec

- 5090 в ~1.511 быстрее по максимальной производительности, и в ~1.33 в относительной(при равных батчах и условиях)