# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    auto_docstring,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
)
from transformers.models.llama.configuration_llama import LlamaConfig
import math

from transformers.integrations import use_kernel_forward_from_hub


logger = logging.get_logger(__name__)

from transformers.configuration_utils import PretrainedConfig


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:  # False
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:  # False
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta  # 500000.0
        partial_rotary_factor = (
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")  # False
            else 1.0
        )  # partial_rotary_factor=1.0
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )  # head_dim=64
        dim = int(head_dim * partial_rotary_factor)  # dim=64

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    # torch.arange(0, dim, 2, dtype=torch.int64)=tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62])
    # tensor([0.0000, 0.0312, 0.0625, 0.0938, 0.1250, 0.1562, 0.1875, 0.2188, 0.2500,0.2812, 0.3125, 0.3438, 0.3750, 0.4062, 0.4375, 0.4688, 0.5000, 0.5312,0.5625, 0.5938, 0.6250, 0.6562, 0.6875, 0.7188, 0.7500, 0.7812, 0.8125,0.8438, 0.8750, 0.9062, 0.9375, 0.9688], dtype=torch.float32)
    # tensor([1.0000e+00, 1.5069e+00, 2.2708e+00, 3.4220e+00, 5.1567e+00, 7.7708e+00, 1.1710e+01, 1.7646e+01, 2.6591e+01, 4.0071e+01, 6.0385e+01, 9.0996e+01, 1.3712e+02, 2.0664e+02, 3.1139e+02, 4.6924e+02, 7.0711e+02, 1.0656e+03, 1.6057e+03, 2.4197e+03, 3.6463e+03, 5.4948e+03, 8.2802e+03, 1.2478e+04, 1.8803e+04, 2.8335e+04, 4.2699e+04, 6.4344e+04, 9.6961e+04, 1.4611e+05, 2.2018e+05, 3.3180e+05], dtype=torch.float32)
    # tensor([1.0000e+00, 6.6360e-01, 4.4037e-01, 2.9223e-01, 1.9392e-01, 1.2869e-01, 8.5397e-02, 5.6670e-02, 3.7606e-02, 2.4955e-02, 1.6560e-02, 1.0990e-02, 7.2927e-03, 4.8394e-03, 3.2114e-03, 2.1311e-03, 1.4142e-03, 9.3847e-04, 6.2277e-04, 4.1327e-04, 2.7425e-04, 1.8199e-04, 1.2077e-04, 8.0143e-05, 5.3183e-05, 3.5292e-05, 2.3420e-05, 1.5542e-05, 1.0313e-05, 6.8440e-06, 4.5417e-06, 3.0139e-06], dtype=torch.float32)
    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: PretrainedConfig,
    device: "torch.device",
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(
        config, device, seq_len, **rope_kwargs
    )
    # inv_freq=tensor([1.0000e+00, 6.6360e-01, 4.4037e-01, 2.9223e-01, 1.9392e-01, 1.2869e-01, 8.5397e-02, 5.6670e-02, 3.7606e-02, 2.4955e-02, 1.6560e-02, 1.0990e-02, 7.2927e-03, 4.8394e-03, 3.2114e-03, 2.1311e-03, 1.4142e-03, 9.3847e-04, 6.2277e-04, 4.1327e-04, 2.7425e-04, 1.8199e-04, 1.2077e-04, 8.0143e-05, 5.3183e-05, 3.5292e-05, 2.3420e-05, 1.5542e-05, 1.0313e-05, 6.8440e-06, 4.5417e-06, 3.0139e-06], dtype=torch.float32)
    # inv_freq.shape=32
    # attention_factor=1.0
    factor = config.rope_scaling["factor"]  # `8` in the original implementation
    # factor=32.0
    low_freq_factor = config.rope_scaling[
        "low_freq_factor"
    ]  # `1` in the original implementation
    # low_freq_factor=1.0
    high_freq_factor = config.rope_scaling[
        "high_freq_factor"
    ]  # `4` in the original implementation
    # high_freq_factor=4.0
    old_context_len = config.rope_scaling[
        "original_max_position_embeddings"
    ]  # `8192` in the original implementation
    # old_context_len=8192

    low_freq_wavelen = old_context_len / low_freq_factor
    # low_freq_wavelen=8192.0/1.0=8192.0
    high_freq_wavelen = old_context_len / high_freq_factor
    # high_freq_wavelen=8192.0/4.0=2048.0

    wavelen = 2 * math.pi / inv_freq
    # wavelen=torch.Size([32])
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
    # inv_freq_llama=torch.Size([32])
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    # smooth_factor=torch.Size([32])
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    # smoothed_inv_freq=torch.Size([32])
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    # is_medium_freq=torch.Size([32])
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    # inv_freq_llama=torch.Size([32])
    # inv_freq_llama=tensor([1.0000e+00, 6.6360e-01, 4.4037e-01, 2.9223e-01, 1.9392e-01, 1.2869e-01, 8.5397e-02, 5.6670e-02, 3.7606e-02, 2.4955e-02, 1.6560e-02, 1.0990e-02, 7.2927e-03, 4.8394e-03, 3.2114e-03, 1.2905e-03, 4.2956e-04, 9.7083e-05, 1.9462e-05, 1.2915e-05, 8.5703e-06, 5.6872e-06, 3.7741e-06, 2.5045e-06, 1.6620e-06, 1.1029e-06, 7.3187e-07, 4.8567e-07, 3.2229e-07, 2.1387e-07, 1.4193e-07, 9.4183e-08], dtype=torch.float32)
    return inv_freq_llama, attention_factor


# @use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        # True for llama3_2
        # config.rope_scaling={'factor': 32.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings  # 131072
        self.original_max_seq_len = config.max_position_embeddings  # 131072

        self.config = config
        # self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        rope_init_funcs = {"llama3": _compute_llama3_parameters}
        self.rope_init_fn = rope_init_funcs[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        # inv_freq.shape=32
        # self.attention_scaling=1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    # бесполезно, не используется в этой модели
    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.attention_interface = ALL_ATTENTION_FUNCTIONS[
            self.config._attn_implementation
        ]
        # flash_attention_2

        self.attn_drop = 0.0 if not self.training else self.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(
        #         key_states, value_states, self.layer_idx, cache_kwargs
        #     )

        attn_output, attn_weights = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attn_drop,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        # self.padding_idx=128004
        self.vocab_size = config.vocab_size
        # self.vocab_size=128256

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        # config.hidden_size=2048
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # config.num_hidden_layers=16
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # config.hidden_size=2048
        # config.rms_norm_eps=1e-05
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
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
        # output_hidden_states=False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # use_cache=True но на train это бесполезно
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )

        if inputs_embeds is None:
            # inputs_embeds = None
            inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds=torch.Size([4, 1024, 2048])
            # input_ids=torch.Size([4, 1024])

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            # past_seen_tokens=0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            # cache_position=torch.Size([1024])
            # cache_position=tensor([   0,    1,    2,  ..., 1021, 1022, 1023], device='cuda:0')

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            # position_ids=torch.Size([1, 1024])
            # position_ids=tensor([[   0,    1,    2,  ..., 1021, 1022, 1023]], device='cuda:0')

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        # causal_mask=None

        hidden_states = inputs_embeds
        # hidden_states=torch.Size([4, 1024, 2048])

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # len(position_embeddings)=2
        # position_embeddings[0].shape=torch.Size([1, 1024, 64])
        # position_embeddings[1].shape=torch.Size([1, 1024, 64])

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # all_hidden_states=None
        # all_self_attns=None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            # if output_hidden_states:
            #     all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        # True
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        # if self.config._attn_implementation == "flex_attention":
        #     if isinstance(attention_mask, torch.Tensor):
        #         attention_mask = make_flex_block_causal_mask(attention_mask)
        #     return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        # past_seen_tokens = (
        #     past_key_values.get_seq_length() if past_key_values is not None else 0
        # )
        # using_compilable_cache = (
        #     past_key_values.is_compileable if past_key_values is not None else False
        # )

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if (
        #     self.config._attn_implementation == "sdpa"
        #     and not using_compilable_cache
        #     and not output_attentions
        # ):
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         is_training=self.training,
        #     ):
        #         return None

        # dtype = input_tensor.dtype
        # sequence_length = input_tensor.shape[1]
        # if using_compilable_cache:
        #     target_length = past_key_values.get_max_cache_shape()
        # else:
        #     target_length = (
        #         attention_mask.shape[-1]
        #         if isinstance(attention_mask, torch.Tensor)
        #         else past_seen_tokens + sequence_length + 1
        #     )

        # # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        # causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
        #     attention_mask,
        #     sequence_length=sequence_length,
        #     target_length=target_length,
        #     dtype=dtype,
        #     cache_position=cache_position,
        #     batch_size=input_tensor.shape[0],
        # )

        # if (
        #     self.config._attn_implementation == "sdpa"
        #     and attention_mask is not None
        #     and attention_mask.device.type in ["cuda", "xpu", "npu"]
        #     and not output_attentions
        # ):
        #     # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        #     # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        #     # Details: https://github.com/pytorch/pytorch/issues/110213
        #     min_dtype = torch.finfo(dtype).min
        #     causal_mask = AttentionMaskConverter._unmask_unattended(
        #         causal_mask, min_dtype
        #     )

        # return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=cache_position.device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=cache_position.device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        print("LlamaForCausalLM_v2")
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #
        #  config = LlamaConfig {
        # "architectures": [
        #     "LlamaForCausalLM"
        # ],
        # "attention_bias": false,
        # "attention_dropout": 0.0,
        # "bos_token_id": 128000,
        # "eos_token_id": 128009,
        # "head_dim": 64,
        # "hidden_act": "silu",
        # "hidden_size": 2048,
        # "initializer_range": 0.02,
        # "intermediate_size": 8192,
        # "max_position_embeddings": 131072,
        # "mlp_bias": false,
        # "model_type": "llama",
        # "num_attention_heads": 32,
        # "num_hidden_layers": 16,
        # "num_key_value_heads": 8,
        # "pad_token_id": 128004,
        # "pretraining_tp": 1,
        # "rms_norm_eps": 1e-05,
        # "rope_scaling": {
        #     "factor": 32.0,
        #     "high_freq_factor": 4.0,
        #     "low_freq_factor": 1.0,
        #     "original_max_position_embeddings": 8192,
        #     "rope_type": "llama3"
        # },
        # "rope_theta": 500000.0,
        # "tie_word_embeddings": true,
        # "torch_dtype": "bfloat16",
        # "transformers_version": "4.53.2",
        # "unsloth_fixed": true,
        # "use_cache": true,
        # "vocab_size": 128256
        # }

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
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

        # input_ids=torch.Size([4, 1024])
        # attention_mask=attention_mask.shape
        # position_ids=None
        # past_key_values=None
        # inputs_embeds=None
        # labels=torch.Size([4, 1024])
        # use_cache=None
        # output_attentions=False
        # output_hidden_states=False
        # cache_position=None
        # logits_to_keep=0
        # kwargs={}

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # hidden_states=torch.Size([4, 1024, 2048])
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # hidden_states[:, slice_indices, :].shape=torch.Size([4, 1024, 2048])
        # logits=torch.Size([4, 1024, 128256])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
