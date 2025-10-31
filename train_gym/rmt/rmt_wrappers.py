import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import TYPE_CHECKING
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from liger_kernel.transformers.model.llama import lce_maybe_trainable_lm_head
import gc

if TYPE_CHECKING:
    from transformers.cache_utils import Cache


class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim = getattr(self.model.config, "n_embd", self.model.config.hidden_size)
        memory_weights = (
            torch.randn(
                (num_mem_tokens, memory_dim),
                device=self.model.device,
                dtype=self.model.dtype,
            )
            * embeddings.weight.data.std()
        )
        self.register_parameter(
            "memory",
            torch.nn.Parameter(
                memory_weights,
                requires_grad=True,
            ),
        )

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(
            input_ids, memory_state, write_mem=True, **kwargs
        )
        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(out, **kwargs)

        return out, new_memory_state

    def generate(self, input_ids, memory_state, attention_mask=None, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(
            input_ids, memory_state, attention_mask=attention_mask, write_mem=False
        )
        out = self.model.generate(
            inputs_embeds=seg_kwargs["inputs_embeds"],
            attention_mask=seg_kwargs["attention_mask"],
            **generate_kwargs,
        )
        return out

    def process_input(self, input_ids, memory_state, write_mem, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if self.num_mem_tokens > 0:
            if write_mem:
                inputs_embeds = torch.cat(
                    [memory_state, inputs_embeds, memory_state], dim=1
                )
            else:
                inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        seg_kwargs["input_ids"] = None
        seg_kwargs["inputs_embeds"] = inputs_embeds
        if kwargs.get("attention_mask") is not None:
            seg_kwargs["attention_mask"] = self.pad_attention_mask(
                kwargs["attention_mask"], inputs_embeds.shape
            )
        seg_kwargs["output_hidden_states"] = True
        return seg_kwargs

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[
                :, self.num_mem_tokens : self.num_mem_tokens + attention_mask.shape[1]
            ] = attention_mask
            return mask

    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens :]
            out["logits"] = model_outputs.logits[
                :, self.num_mem_tokens : -self.num_mem_tokens
            ]

            if kwargs.get("output_hidden_states"):
                out["hidden_states"] = [
                    lh[:, self.num_mem_tokens : -self.num_mem_tokens]
                    for lh in model_outputs.hidden_states
                ]
            if kwargs.get("output_attentions"):
                out["attentions"] = model_outputs["attentions"]
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state


class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
        self.config = self.memory_cell.model.config

    def forward(
        self,
        input_ids,
        labels=None,
        labels_mask=None,
        inputs_embeds=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        memory_state = None
        segmented = self.segment(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        cell_outputs = []
        for seg_num, segment in enumerate(segmented):
            cell_out, memory_state = self.memory_cell(
                **segment, memory_state=memory_state, output_hidden_states=True
            )
            cell_outputs.append(cell_out)
            memory_state = self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(
            cell_outputs,
            labels=labels,
            labels_mask=labels_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return out

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out, memory_state = self.memory_cell(
                **segment, memory_state=memory_state, output_hidden_states=True
            )

        final_segment = segmented[-1]
        out = self.memory_cell.generate(
            **final_segment, memory_state=memory_state, **generate_kwargs
        )
        out = torch.cat([input_ids, out], dim=1)
        return out

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments

    def split_tensor(self, tensor):
        align = self.rmt_config.get("segment_alignment")
        segment_size = self.rmt_config.get("segment_size")
        if align in {"left", None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [
                tensor.shape[1]
            ]
            segments = [
                tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])
            ]
        elif align in {"right", None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [
                tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])
            ]
        elif align == "center":
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple(
            [
                torch.cat(layer_hs, dim=1)
                for layer_hs in zip(*[o.hidden_states for o in cell_outputs])
            ]
        )

        labels = kwargs.get("labels")
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))

            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get("labels_mask")
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]

            out["loss"] = loss_fct(flat_logits, flat_labels)
        else:
            out["loss"] = 0

        out["logits"] = full_logits
        segment_keys = ["loss", "logits"]
        if kwargs.get("output_attentions"):
            segment_keys.append("attentions")
        if kwargs.get("output_hidden_states"):
            segment_keys.append("hidden_states")
            out["hidden_states"] = full_hidden_states

        return out

    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get("k2"), self.rmt_config.get(
            "max_n_segments"
        )
        if seg_num == 0 or k2 in {-1, None} or seg_num + k2 > max_n_segments:
            return memory_state

        memory_state = memory_state.detach()
        return memory_state

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)


class MemoryCellTrain(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim = getattr(self.model.config, "n_embd", self.model.config.hidden_size)
        memory_weights = (
            torch.randn(
                (num_mem_tokens, memory_dim),
                device=self.model.device,
                dtype=self.model.dtype,
            )
            * embeddings.weight.data.std()
        )
        self.register_parameter(
            "memory",
            torch.nn.Parameter(
                memory_weights,
                requires_grad=True,
            ),
        )

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        labels = None
        num_items_in_batch = None
        if "labels" in kwargs:
            labels = kwargs.pop("labels")

        if "num_items_in_batch" in kwargs:
            num_items_in_batch = kwargs.pop("num_items_in_batch")

        seg_kwargs = self.process_input(
            input_ids, memory_state, write_mem=True, **kwargs
        )
        out = self.model(**seg_kwargs)
        kwargs["labels"] = labels
        kwargs["num_items_in_batch"] = num_items_in_batch
        out, new_memory_state = self.process_output(out, **kwargs)

        return out, new_memory_state

    def generate(self, input_ids, memory_state, attention_mask=None, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(
            input_ids,
            memory_state,
            attention_mask=attention_mask,
            write_mem=False,
        )
        out = self.model.generate(
            inputs_embeds=seg_kwargs["inputs_embeds"],
            attention_mask=seg_kwargs["attention_mask"],
            **generate_kwargs,
        )
        return out

    def process_input(self, input_ids, memory_state, write_mem, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if self.num_mem_tokens > 0:
            if write_mem:
                inputs_embeds = torch.cat(
                    [memory_state, inputs_embeds, memory_state], dim=1
                )
            else:
                inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        seg_kwargs["input_ids"] = None
        seg_kwargs["inputs_embeds"] = inputs_embeds
        if kwargs.get("attention_mask") is not None:
            seg_kwargs["attention_mask"] = self.pad_attention_mask(
                kwargs["attention_mask"], inputs_embeds.shape
            )
        seg_kwargs["output_hidden_states"] = True
        return seg_kwargs

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[
                :, self.num_mem_tokens : self.num_mem_tokens + attention_mask.shape[1]
            ] = attention_mask
            return mask

    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens :]
            out["logits"] = model_outputs.logits[
                :, self.num_mem_tokens : -self.num_mem_tokens
            ]

            if kwargs.get("output_hidden_states"):
                out["hidden_states"] = [
                    lh[:, self.num_mem_tokens : -self.num_mem_tokens]
                    for lh in model_outputs.hidden_states
                ]
            if kwargs.get("output_attentions"):
                out["attentions"] = model_outputs["attentions"]

            if not kwargs["labels"] is None:
                loss = self.model.loss_function(
                    logits=out["logits"],
                    labels=kwargs["labels"],
                    vocab_size=self.model.config.vocab_size,
                    num_items_in_batch=kwargs["num_items_in_batch"],
                )

                out["loss"] = loss

            # clean memory while training
            if self.training:
                out["logits"] = None
                out["attentions"] = None
                out["hidden_states"] = None
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state


class RecurrentWrapperTrain(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
        self.accepts_loss_kwargs = True
        self.config = self.memory_cell.model.config

    def forward(
        self,
        input_ids,
        labels=None,
        labels_mask=None,
        inputs_embeds=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        num_items_in_batch=None,
        **kwargs,
    ):
        memory_state = None
        segmented = self.segment(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        cell_outputs = []
        for seg_num, segment in enumerate(segmented):
            cell_out, memory_state = self.memory_cell(
                **segment,
                memory_state=memory_state,
                output_hidden_states=True,
                num_items_in_batch=num_items_in_batch,
            )
            cell_outputs.append(cell_out)
            memory_state = self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(
            cell_outputs,
            labels=labels,
            labels_mask=labels_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return out

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out, memory_state = self.memory_cell(
                **segment, memory_state=memory_state, output_hidden_states=True
            )

        final_segment = segmented[-1]
        out = self.memory_cell.generate(
            **final_segment, memory_state=memory_state, **generate_kwargs
        )
        out = torch.cat([input_ids, out], dim=1)
        return out

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments

    def split_tensor(self, tensor):
        align = self.rmt_config.get("segment_alignment")
        segment_size = self.rmt_config.get("segment_size")
        if align in {"left", None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [
                tensor.shape[1]
            ]
            segments = [
                tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])
            ]
        elif align in {"right", None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [
                tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])
            ]
        elif align == "center":
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        if not self.training:
            full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
            full_hidden_states = tuple(
                [
                    torch.cat(layer_hs, dim=1)
                    for layer_hs in zip(*[o.hidden_states for o in cell_outputs])
                ]
            )

        labels = kwargs.get("labels")
        if labels is not None:
            losses = [o.loss for o in cell_outputs]
            losses = torch.stack(losses, dim=0).sum(dim=0)

            out["loss"] = losses
        else:
            out["loss"] = 0

        if not self.training:
            out["logits"] = full_logits

        if not self.training:
            if kwargs.get("output_hidden_states"):
                out["hidden_states"] = full_hidden_states

        return out

    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get("k2"), self.rmt_config.get(
            "max_n_segments"
        )
        if seg_num == 0 or k2 in {-1, None} or seg_num + k2 > max_n_segments:
            return memory_state

        memory_state = memory_state.detach()
        return memory_state

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)


class MemoryCellTrainLiger(MemoryCellTrain):
    def generate(self, input_ids, memory_state, attention_mask=None, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(
            input_ids,
            memory_state,
            attention_mask=attention_mask,
            write_mem=False,
        )
        old_fwd = self.model.forward

        def new_forward(*args, **kwargs):
            output = old_fwd(*args, **kwargs)
            out = CausalLMOutputWithCrossAttentions()
            out["past_key_values"] = output["past_key_values"]
            logits = self.model.lm_head(output.last_hidden_state)
            out["logits"] = logits

            if kwargs.get("output_hidden_states"):
                out["hidden_states"] = output.hidden_states

            if kwargs.get("output_attentions"):
                out["attentions"] = output["attentions"]

            return out

        self.model.forward = new_forward

        out = self.model.generate(
            inputs_embeds=seg_kwargs["inputs_embeds"],
            attention_mask=seg_kwargs["attention_mask"],
            **generate_kwargs,
        )
        out = torch.cat([input_ids, out], dim=1)
        self.model.forward = old_fwd
        return out

    @torch.compile
    def extract_last_hidden_state(self, model_outputs):
        # return model_outputs.last_hidden_state[
        return model_outputs.hidden_states[-1][
            :, self.num_mem_tokens : -self.num_mem_tokens
        ].contiguous()

    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            # memory_state = model_outputs.last_hidden_state[:, -self.num_mem_tokens :]
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens :]

            last_hidden_state = self.extract_last_hidden_state(model_outputs)
            # last_hidden_state = model_outputs.last_hidden_state
            # fake_labels = (
            #     torch.ones(
            #         (memory_state.shape[0], self.num_mem_tokens),
            #         device=memory_state.device,
            #         dtype=torch.long,
            #     )
            #     * -100
            # )
            # kwargs["labels"] = torch.cat(
            #     [
            #         fake_labels,
            #         kwargs["labels"],
            #         fake_labels,
            #     ],
            #     dim=1,
            # )

            model_outputs.last_hidden_state = None

            if not self.training:
                logits = self.model.lm_head(last_hidden_state)
                out["logits"] = logits

                if kwargs.get("output_hidden_states"):
                    out["hidden_states"] = [
                        lh[:, self.num_mem_tokens : -self.num_mem_tokens]
                        for lh in model_outputs.hidden_states
                    ]

                if kwargs.get("output_attentions"):
                    out["attentions"] = model_outputs["attentions"]

            if not kwargs.get("labels") is None:
                loss = lce_maybe_trainable_lm_head(
                    self.model,
                    hidden_states=last_hidden_state,
                    hidden_size=self.model.config.hidden_size,
                    labels=kwargs["labels"],
                    shift_labels=None,
                    num_items_in_batch=kwargs["num_items_in_batch"],
                )

                out["loss"] = loss

            # clean memory while training
            if self.training:
                out["logits"] = None
                out["attentions"] = None
                out["hidden_states"] = None
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state


def lce_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    # if self.config.pretraining_tp > 1:
    #     raise Exception("Liger Kernel does not support pretraining_tp!!")

    return outputs
