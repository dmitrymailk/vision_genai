### ToDO

#### RMT generation with fsdp2 error
```python
def process_input(self, input_ids, memory_state, write_mem, **kwargs):
    seg_kwargs = dict(**kwargs)

    inputs_embeds = kwargs.get("inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
```
```text
[rank0]:     raise RuntimeError(
[rank0]: RuntimeError: aten.embedding.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!
```
