"""
annotated minimal version, with torch compile
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["http_proxy"] = "127.0.0.1:2334"
os.environ["https_proxy"] = "127.0.0.1:2334"
os.environ["TORCH_LOGS"] = "recompiles"

import torch

torch.set_float32_matmul_precision("high")
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from dataclasses import dataclass
import wandb
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from diffusers.configuration_utils import ConfigMixin, register_to_config


from diffusers.models.modeling_utils import ModelMixin

# from diffusers.models.unets.unet_2d_blocks import get_down_block, get_up_block
from diffusers.models.unets.unet_2d import UNet2DOutput

import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils.torch_utils import is_torch_version


from diffusers.models.activations import get_activation

from diffusers.utils import deprecate
import numbers

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import is_torch_version, maybe_allow_in_graph
import inspect
import math
from typing import List, Optional, Tuple, Union

logger = logging.get_logger(__name__)

from diffusers.utils.torch_utils import apply_freeu
from diffusers.configuration_utils import register_to_config
from diffusers.utils import deprecate, logging


from diffusers.models.normalization import (
    RMSNorm,
)


class Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        # self.channels=channels=512
        # self.out_channels=out_channels=512
        # self.use_conv=use_conv=True
        # self.use_conv_transpose=use_conv_transpose=False
        # self.name=name='conv'
        # self.interpolate=interpolate=True
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        # THIS BRANCH
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        # use_conv_transpose=False
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(
                channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=bias,
            )
        # THIS BRANCH
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        # name='conv'
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        assert hidden_states.shape[1] == self.channels
        # self.norm=None
        # hidden_states=torch.Size([4, 512, 2, 2])
        if self.norm is not None:
            # hidden_states=
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )
        # self.use_conv_transpose=False
        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16 until PyTorch 2.1
        # https://github.com/pytorch/pytorch/issues/86679#issuecomment-1783978767
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        # self.interpolate=True
        if self.interpolate:
            # upsample_nearest_nhwc also fails when the number of output elements is large
            # https://github.com/pytorch/pytorch/issues/141831
            # output_size=None
            # output_size=2
            scale_factor = (
                2
                if output_size is None
                else max([f / s for f, s in zip(output_size, hidden_states.shape[-2:])])
            )
            if hidden_states.numel() * scale_factor > pow(2, 31):
                hidden_states = hidden_states.contiguous()
            # hidden_states=torch.Size([4, 512, 2, 2])
            # THIS BRANCH
            if output_size is None:
                # hidden_states=torch.Size([4, 512, 4, 4])
                hidden_states = F.interpolate(
                    hidden_states, scale_factor=2.0, mode="nearest"
                )
            else:
                hidden_states = F.interpolate(
                    hidden_states, size=output_size, mode="nearest"
                )

        # Cast back to original dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            # self.name='conv'
            # THIS BRANCH
            if self.name == "conv":
                # hidden_states=torch.Size([4, 512, 4, 4])
                hidden_states = self.conv(hidden_states)
                # hidden_states=torch.Size([4, 512, 4, 4])
                hidden_states
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        self.norm = None

        # сокращаем размер в 2 раза
        conv = nn.Conv2d(
            self.channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.conv = conv

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class UNetMidBlock2D(nn.Module):
    """
    A 2D UNet mid-block [`UNetMidBlock2D`] with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (`int`): The number of input channels.
        temb_channels (`int`): The number of temporal embedding channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
            The type of normalization to apply to the time embeddings. This can help to improve the performance of the
            model on tasks with long-range temporal dependencies.
        resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
        resnet_pre_norm (`bool`, *optional*, defaults to `True`):
            Whether to use pre-normalization for the resnet blocks.
        add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
        attention_head_dim (`int`, *optional*, defaults to 1):
            Dimension of a single attention head. The number of attention heads is determined based on this value and
            the number of input channels.
        output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

    Returns:
        `torch.Tensor`: The output of the last residual block, which is a tensor of shape `(batch_size, in_channels,
        height, width)`.

    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = (
                resnet_groups if resnet_time_scale_shift == "default" else None
            )

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=(
                            temb_channels
                            if resnet_time_scale_shift == "spatial"
                            else None
                        ),
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        данный слой отличается от downsample и upsample что
        мы никуда не сохраняем отпуты. что наверняка плохо
        так как вероятно все затухает
        """
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if attn is not None:
                    hidden_states = attn(hidden_states, temb=temb)
                hidden_states = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb
                )
            else:
                if attn is not None:
                    hidden_states = attn(hidden_states, temb=temb)
                hidden_states = resnet(hidden_states, temb)

        return hidden_states


class AttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.downsample_type = downsample_type
        # attention_head_dim=8
        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels
        # num_layers=2
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Attention(
                    # out_channels=512
                    # heads=out_channels // attention_head_dim=64
                    # dim_head=attention_head_dim=8
                    # rescale_output_factor=output_scale_factor=1.0
                    # eps=resnet_eps=1e-05
                    # norm_num_groups=resnet_groups=32
                    # residual_connection=True
                    # bias=True
                    # upcast_softmax=True
                    # _from_deprecated_attn_block=True
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if downsample_type == "conv":
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        # out_channels=512,
                        # use_conv=True,
                        # out_channels=out_channels=512
                        # padding=downsample_padding=1
                        # name="op",
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
            )

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb
                )
                hidden_states = attn(hidden_states, **cross_attention_kwargs)
                output_states = output_states + (hidden_states,)
            else:
                # hidden_states=torch.Size([16, 256, 4, 4])
                hidden_states = resnet(hidden_states, temb)
                # hidden_states=torch.Size([16, 512, 4, 4])
                hidden_states = attn(hidden_states, **cross_attention_kwargs)
                # hidden_states=torch.Size([16, 512, 4, 4])
                output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                if self.downsample_type == "resnet":
                    hidden_states = downsampler(hidden_states, temb=temb)
                else:
                    hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []
        # num_layers=2
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                # in_channels=in_channels=128
                # out_channels=out_channels=out_channels
                # temb_channels=temb_channels=512
                # eps=resnet_eps=1e-05
                # groups=resnet_groups=128
                # dropout=dropout=0.0
                # time_embedding_norm=resnet_time_scale_shift='default'
                # non_linearity=resnet_act_fn='silu'
                # output_scale_factor=output_scale_factor=1.0
                # pre_norm=resnet_pre_norm=True
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
                # torch.compile(
                #     ResnetBlock2D(
                #         in_channels=in_channels,
                #         out_channels=out_channels,
                #         temb_channels=temb_channels,
                #         eps=resnet_eps,
                #         groups=resnet_groups,
                #         dropout=dropout,
                #         time_embedding_norm=resnet_time_scale_shift,
                #         non_linearity=resnet_act_fn,
                #         output_scale_factor=output_scale_factor,
                #         pre_norm=resnet_pre_norm,
                #     ),
                #     fullgraph=True,
                #     dynamic=True,
                #     mode="reduce-overhead",
                # )
            )

        self.resnets = nn.ModuleList(resnets)
        # add_downsample=True
        # THIS BRANCH
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    # out_channels=128
                    # use_conv=True
                    # out_channels=out_channels=128
                    # padding=downsample_padding=1
                    # name="op"
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        output_states = ()

        for resnet in self.resnets:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class AttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: int = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        upsample_type: str = "conv",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.upsample_type = upsample_type

        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if upsample_type == "conv":
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        elif upsample_type == "resnet":
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        up=True,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb
                )
                hidden_states = attn(hidden_states)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                if self.upsample_type == "resnet":
                    hidden_states = upsampler(hidden_states, temb=temb)
                else:
                    hidden_states = upsampler(hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                # res_skip_channels=512
                # resnet_in_channels=512
                # in_channels=resnet_in_channels + res_skip_channels=1024
                # out_channels=out_channels=512
                # temb_channels=temb_channels=512
                # eps=resnet_eps=1e-05
                # groups=resnet_groups=32
                # dropout=dropout=0.0
                # time_embedding_norm=resnet_time_scale_shift='default'
                # non_linearity=resnet_act_fn='silu'
                # output_scale_factor=output_scale_factor
                # pre_norm=resnet_pre_norm=1.0
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
                # torch.compile(
                #     ResnetBlock2D(
                #         in_channels=resnet_in_channels + res_skip_channels,
                #         out_channels=out_channels,
                #         temb_channels=temb_channels,
                #         eps=resnet_eps,
                #         groups=resnet_groups,
                #         dropout=dropout,
                #         time_embedding_norm=resnet_time_scale_shift,
                #         non_linearity=resnet_act_fn,
                #         output_scale_factor=output_scale_factor,
                #         pre_norm=resnet_pre_norm,
                #     ),
                #     fullgraph=True,
                #     dynamic=True,
                #     mode="reduce-overhead",
                # )
            )

        self.resnets = nn.ModuleList(resnets)
        # add_upsample=True
        # THIS BRANCH
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        # out_channels=512,
                        # use_conv=True,
                        # out_channels=out_channels=512,
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        # resolution_idx=None
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        # is_freeu_enabled=None=False
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet in self.resnets:
            # pop res hidden states
            # res_hidden_states=torch.Size([4, 512, 2, 2])
            # len(res_hidden_states_tuple)=len(res_hidden_states_tuple)
            res_hidden_states = res_hidden_states_tuple[-1]
            #  it returns all elements [:] except the last one -1.
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )
            # hidden_states=torch.Size([4, 512, 2, 2])
            # res_hidden_states=torch.Size([4, 512, 2, 2])
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)
        # THIS BRANCH
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                # upsample_size=None
                # hidden_states=torch.Size([4, 512, 2, 2])
                hidden_states = upsampler(hidden_states, upsample_size)
                # upsample_size=None
                # hidden_states=torch.Size([4, 256, 8, 8])

        return hidden_states


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    down_block_type = (
        down_block_type[7:]
        if down_block_type.startswith("UNetRes")
        else down_block_type
    )
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnDownBlock2D":
        if add_downsample is False:
            downsample_type = None
        else:
            downsample_type = downsample_type or "conv"  # default to 'conv'
        return AttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            downsample_type=downsample_type,
        )

    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
) -> nn.Module:
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = (
        up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    )
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    elif up_block_type == "AttnUpBlock2D":
        if add_upsample is False:
            upsample_type = None
        else:
            upsample_type = upsample_type or "conv"  # default to 'conv'

        return AttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            upsample_type=upsample_type,
        )

    raise ValueError(f"{up_block_type} does not exist.")


# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
    """
    ok Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


@maybe_allow_in_graph
class Attention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        kv_heads (`int`,  *optional*, defaults to `None`):
            The number of key and value heads to use for multi-head attention. Defaults to `heads`. If
            `kv_heads=heads`, the model will use Multi Head Attention (MHA), if `kv_heads=1` the model will use Multi
            Query Attention (MQA) otherwise GQA is used.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor2_0"] = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = query_dim
        self.out_context_dim = query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal
        self._from_deprecated_attn_block = _from_deprecated_attn_block
        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5
        self.heads = heads
        self.sliceable_head_dim = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        self.group_norm = nn.GroupNorm(
            num_channels=query_dim,
            num_groups=norm_num_groups,
            eps=eps,
            affine=True,
        )

        self.spatial_norm = None

        self.norm_q = None
        self.norm_k = None

        self.norm_cross = None

        self.to_q = nn.Linear(
            query_dim,
            self.inner_dim,
            bias=bias,
        )

        # only relevant for the `AddedKVProcessor` classes
        self.to_k = nn.Linear(
            self.cross_attention_dim,
            self.inner_kv_dim,
            bias=bias,
        )
        self.to_v = nn.Linear(
            self.cross_attention_dim,
            self.inner_kv_dim,
            bias=bias,
        )

        self.added_proj_bias = added_proj_bias

        self.add_q_proj = None
        self.add_k_proj = None
        self.add_v_proj = None

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        self.to_add_out = None

        self.norm_added_q = None
        self.norm_added_k = None

        processor = AttnProcessor2_0()
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        # hidden_states=torch.Size([16, 512, 4, 4])
        # encoder_hidden_states=None
        # attention_mask=None,
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k
            for k, _ in cross_attention_kwargs.items()
            if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        # cross_attention_kwargs={}
        # attn_parameters={'args', 'kwargs', 'attention_mask', 'temb', 'attn', 'hidden_states', 'encoder_hidden_states'}
        cross_attention_kwargs = {
            k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
        }
        # hidden_states=torch.Size([16, 512, 4, 4])
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size
        )
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(
            batch_size, seq_len * extra_dim, head_size, dim // head_size
        )
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(
                batch_size * head_size, seq_len * extra_dim, dim // head_size
            )

        return tensor

    def prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    target_length,
                )
                padding = torch.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(
                    head_size, dim=0, output_size=attention_mask.shape[0] * head_size
                )
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(
                head_size, dim=1, output_size=attention_mask.shape[1] * head_size
            )

        return attention_mask

    def norm_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert (
            self.norm_cross is not None
        ), "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        # hidden_states=torch.Size([16, 512, 4, 4])
        # residual=torch.Size([16, 512, 4, 4])
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            # (16, 512, 4, 4)
            batch_size, channel, height, width = hidden_states.shape
            # hidden_states=torch.Size([16, 16, 512]) (batch_size, size, channel)
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        # 16, 16
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        # attention_mask=None
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )
        # hidden_states=torch.Size([16, 16, 512])
        # query=torch.Size([16, 16, 512])
        query = attn.to_q(hidden_states)

        # THIS BRANCH
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        # key=torch.Size([16, 16, 512])
        # value=torch.Size([16, 16, 512])
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # inner_dim=512, attn.heads=64,
        inner_dim = key.shape[-1]
        # head_dim=8
        head_dim = inner_dim // attn.heads
        # query=torch.Size([16, 64, 16, 8])
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # key=torch.Size([16, 64, 16, 8])
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # value=torch.Size([16, 64, 16, 8])
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        # hidden_states=torch.Size([16, 64, 16, 8])
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        # hidden_states=torch.Size([16, 16, 512])
        hidden_states = hidden_states.to(query.dtype)

        # linear proj, hidden_states=torch.Size([16, 16, 512])
        hidden_states = attn.to_out[0](hidden_states)
        # dropout, hidden_states=torch.Size([16, 16, 512])
        hidden_states = attn.to_out[1](hidden_states)
        # THIS BRANCH
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        # hidden_states=torch.Size([16, 512, 4, 4])
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        # attn.rescale_output_factor=1.0
        # hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class DDPMScheduler(SchedulerMixin, ConfigMixin):
    """
    ok `DDPMScheduler` explores the connections between denoising score matching and Langevin dynamics sampling.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, `squaredcos_cap_v2`, or `sigmoid`.
        trained_betas (`np.ndarray`, *optional*):
            An array of betas to pass directly to the constructor without using `beta_start` and `beta_end`.
        variance_type (`str`, defaults to `"fixed_small"`):
            Clip the variance when adding noise to the denoised sample. Choose from `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
    ):
        # trained_betas=None
        # beta_schedule='linear'
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # THIS BRANCH
        elif beta_schedule == "linear":
            # beta_start=0.0001
            # beta_end=0.02
            # num_train_timesteps=1000
            # self.betas=torch.Size([1000])
            # self.betas[:5]=tensor([1.0000e-04, 1.1992e-04, 1.3984e-04, 1.5976e-04, 1.7968e-04])
            self.betas = torch.linspace(
                beta_start,
                beta_end,
                num_train_timesteps,
                dtype=torch.float32,
            )

        # Rescale for zero SNR
        # rescale_betas_zero_snr=False
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)
        #  self.alphas=torch.Size([1000])
        self.alphas = 1.0 - self.betas
        # self.alphas_cumprod=torch.Size([1000])
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        # self.timesteps[:10]=tensor([999, 998, 997, 996, 995, 994, 993, 992, 991, 990])
        # self.timesteps.shape=torch.Size([1000])
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy()
        )
        # self.variance_type=variance_type='fixed_small'
        self.variance_type = variance_type

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.

        """
        # num_inference_steps=1000
        # device='cuda'
        # timesteps=None
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError(
                "Can only pass one of `num_inference_steps` or `custom_timesteps`."
            )
        # timesteps=None
        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`: {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:  # num_inference_steps > self.config.num_train_timesteps=False
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps
            self.custom_timesteps = False

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            # self.config.timestep_spacing='leading'
            if self.config.timestep_spacing == "linspace":
                timesteps = (
                    np.linspace(
                        0, self.config.num_train_timesteps - 1, num_inference_steps
                    )
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )
            # THIS CONDITION
            elif self.config.timestep_spacing == "leading":
                # step_ratio=1
                step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (
                    (np.arange(0, num_inference_steps) * step_ratio)
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )
                # timesteps.shape=(1000,)
                # self.config.steps_offset=0
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = np.round(
                    np.arange(self.config.num_train_timesteps, 0, -step_ratio)
                ).astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )
        # self.timesteps.shape=(1000,)
        # timesteps[:10]=array([999, 998, 997, 996, 995, 994, 993, 992, 991, 990])
        # и так далее до 0
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        # t=
        # predicted_variance=
        # variance_type=
        # prev_t=
        prev_t = self.previous_timestep(t)

        # self.alphas_cumprod=torch.Size([1000])
        # alpha_prod_t=tensor(4.0358e-05, device='cuda:0')
        # alpha_prod_t_prev=tensor(4.1182e-05, device='cuda:0')
        # current_beta_t=tensor(0.0200, device='cuda:0')
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        # variance=tensor(0.0200, device='cuda:0')
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        # variance_type=None
        # THIS BRANCH
        if variance_type is None:
            variance_type = self.config.variance_type

        # hacks - were probably added for training stability
        # variance_type='fixed_small'
        # THIS BRANCH
        if variance_type == "fixed_small":
            variance = variance
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = (
                sample.float()
            )  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = (
            torch.clamp(sample, -s, s) / s
        )  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
                шум предсказанный моделью, который нам нужно вычесть из текущего
                sample
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        # timestep=tensor(999)
        # t=tensor(999)
        t = timestep
        # prev_t=tensor(998)
        prev_t = self.previous_timestep(t)
        # model_output.shape[1]=3
        # sample.shape[1]=3
        # sample.shape[1]*2=6
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
            # THIS BRANCH
        else:
            # predicted_variance=None
            predicted_variance = None

        # 1. compute alphas, betas
        # self.alphas_cumprod=torch.Size([1000])
        # alpha_prod_t=tensor(4.0358e-05, device='cuda:0')
        # alpha_prod_t_prev=tensor(4.1182e-05, device='cuda:0')
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        # self.config.prediction_type='epsilon'
        # THIS BRANCH
        if self.config.prediction_type == "epsilon":
            # sample=torch.Size([16, 3, 64, 64])
            # beta_prod_t=tensor(1.0000, device='cuda:0')
            # model_output=torch.Size([16, 3, 64, 64])
            # alpha_prod_t=tensor(4.0358e-05, device='cuda:0')
            # pred_original_sample=torch.Size([16, 3, 64, 64])
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        # self.config.thresholding=False
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        # self.config.clip_sample=True
        # THIS BRANCH
        elif self.config.clip_sample:
            # self.config.clip_sample_range=1.0
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        # alpha_prod_t_prev
        # current_beta_t=tensor(0.0200, device='cuda:0')
        # beta_prod_t=tensor(1.0000, device='cuda:0')
        # pred_original_sample_coeff=tensor(0.0001, device='cuda:0')
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        # current_alpha_t=tensor(0.9800, device='cuda:0')
        # beta_prod_t_prev=tensor(1.0000, device='cuda:0')
        # beta_prod_t=tensor(1.0000, device='cuda:0')
        # current_sample_coeff=tensor(0.9899, device='cuda:0')
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        # pred_original_sample_coeff=tensor(0.0001, device='cuda:0')
        # pred_original_sample=torch.Size([16, 3, 64, 64])
        # current_sample_coeff=tensor(0.9899, device='cuda:0')
        # sample=torch.Size([16, 3, 64, 64])
        # pred_prev_sample=torch.Size([16, 3, 64, 64])
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            # variance_noise
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            # self.variance_type='fixed_small'
            if self.variance_type == "fixed_small_log":
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance)
                    * variance_noise
                )
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            # THIS BRANCH
            else:
                # variance_noise=torch.Size([16, 3, 64, 64])
                # predicted_variance=None
                # t=tensor(999)
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance) ** 0.5
                ) * variance_noise
        # pred_prev_sample=torch.Size([16, 3, 64, 64])
        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )
        # prev_sample=pred_prev_sample=torch.Size([16, 3, 64, 64])
        # pred_original_sample=pred_original_sample=torch.Size([16, 3, 64, 64])
        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample,
            pred_original_sample=pred_original_sample,
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        # self.alphas_cumprod=torch.Size([1000])
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # timesteps=torch.Size([16])
        # timesteps=tensor([548, 605, 655, 909,  73, 346, 509, 798, 792, 217, 108, 915, 653,  24, 201, 255], device='cuda:0')
        timesteps = timesteps.to(original_samples.device)
        # sqrt_alpha_prod=torch.Size([16])
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # sqrt_alpha_prod=torch.Size([16])
        # original_samples.shape=torch.Size([16, 3, 64, 64])
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            # sqrt_alpha_prod=torch.Size([16, 1]) и так пока не станет равным семплам
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        # sqrt_alpha_prod=torch.Size([16, 1, 1, 1])
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        # original_samples.shape=torch.Size([16, 3, 64, 64])
        # sqrt_alpha_prod=torch.Size([16, 1, 1, 1])
        # sqrt_one_minus_alpha_prod=torch.Size([16, 1, 1, 1])
        # noise=torch.Size([16, 3, 64, 64])
        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def get_velocity(
        self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps

    def previous_timestep(self, timestep):
        # timestep=tensor(999)
        # self.custom_timesteps=False
        # self.num_inference_steps=1000
        # self.custom_timesteps or self.num_inference_steps=True
        if self.custom_timesteps or self.num_inference_steps:
            # index=0
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            #  self.timesteps.shape[0]-1=999
            # если индекс самый последний, значит мы достигли конца
            # просто заглушка
            # иначе берем следующий шаг из self.timesteps
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            prev_t = timestep - 1
        # prev_t=tensor(998)
        # timestep=tensor(999)
        return prev_t


class UNet2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample. Dimensions must be a multiple of `2 ** (len(block_out_channels) -
            1)`.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`):
            Tuple of downsample block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            Block type for middle of UNet, it can be either `UNetMidBlock2D` or `None`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(224, 448, 672, 896)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        downsample_type (`str`, *optional*, defaults to `conv`):
            The downsample type for downsampling layers. Choose between "conv" and "resnet"
        upsample_type (`str`, *optional*, defaults to `conv`):
            The upsample type for upsampling layers. Choose between "conv" and "resnet"
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for normalization.
        attn_norm_num_groups (`int`, *optional*, defaults to `None`):
            If set to an integer, a group norm layer will be created in the mid block's [`Attention`] layer with the
            given number of groups. If left as `None`, the group norm layer will only be created if
            `resnet_time_scale_shift` is set to `default`, and if created will have `norm_num_groups` groups.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock2D`]). Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, or `"identity"`.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim` when performing class
            conditioning with `class_embed_type` equal to `None`.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2D",
        up_block_types: Tuple[str, ...] = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
    ):
        super().__init__()
        # sample_size=64
        # time_embedding_dim=None
        # block_out_channels=(128, 128, 256, 256, 512, 512)
        self.sample_size = sample_size
        # time_embed_dim=512
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        # Check inputs
        # len(down_block_types)=6 ('DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'DownBlock2D')
        # len(up_block_types)=6 ('UpBlock2D', 'AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D')
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )
        # len(block_out_channels)=6
        # block_out_channels=(128, 128, 256, 256, 512, 512)
        # down_block_types=('DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'DownBlock2D')
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        # in_channels=3
        # block_out_channels[0]=128
        # kernel_size=3
        # padding=(1, 1)
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=(1, 1),
        )

        # time
        # time_embedding_type=positional

        # block_out_channels[0]=128
        # flip_sin_to_cos=True
        # freq_shift=0
        self.time_proj = Timesteps(
            block_out_channels[0],
            flip_sin_to_cos,
            freq_shift,
        )
        # timestep_input_dim=128
        timestep_input_dim = block_out_channels[0]

        # timestep_input_dim=128
        # time_embed_dim=512
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        # class_embed_type=None
        # num_class_embeds=None
        self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        # output_channel=128
        # down_block_types=('DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'DownBlock2D')
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            # first step
            # num_layers=layers_per_block=2
            # in_channels=input_channel=128
            # out_channels=output_channel=128
            # temb_channels=time_embed_dim=512
            # add_downsample=not is_final_block=True
            # resnet_eps=norm_eps=1e-05
            # resnet_act_fn=act_fn='silu'
            # resnet_groups=norm_num_groups=32
            # attention_head_dim=attention_head_dim=8
            # downsample_padding=downsample_padding=1
            # resnet_time_scale_shift=resnet_time_scale_shift='default'
            # downsample_type=downsample_type='conv'
            # dropout=dropout=0.0
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=(
                    attention_head_dim
                    if attention_head_dim is not None
                    else output_channel
                ),
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        # mid_block_type='UNetMidBlock2D'
        if mid_block_type is None:
            self.mid_block = None
        else:
            # in_channels=block_out_channels[-1]=512
            # temb_channels=time_embed_dim=512
            # dropout=dropout=0.0
            # resnet_eps=norm_eps=1e-05
            # resnet_act_fn=act_fn='silu'
            # output_scale_factor=mid_block_scale_factor=1
            # resnet_time_scale_shift=resnet_time_scale_shift=
            # attention_head_dim=attention_head_dim='default'
            # resnet_groups=norm_num_groups=32
            # attn_groups=attn_norm_num_groups=None
            # add_attention=add_attention=True
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=(
                    attention_head_dim
                    if attention_head_dim is not None
                    else block_out_channels[-1]
                ),
                resnet_groups=norm_num_groups,
                attn_groups=attn_norm_num_groups,
                add_attention=add_attention,
            )

        # up
        # block_out_channels=(128, 128, 256, 256, 512, 512)
        # reversed_block_out_channels=[512, 512, 256, 256, 128, 128]
        reversed_block_out_channels = list(reversed(block_out_channels))
        # reversed_block_out_channels[0]=512
        # output_channel=512
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            is_final_block = i == len(block_out_channels) - 1

            # up_block_type='UpBlock2D'
            # num_layers=layers_per_block + 1=3
            # in_channels=input_channel=128
            # out_channels=output_channel=128
            # prev_output_channel=prev_output_channel=128
            # temb_channels=time_embed_dim=512
            # add_upsample=not is_final_block=False
            # resnet_eps=norm_eps=1e-05
            # resnet_act_fn=act_fn='silu'
            # resnet_groups=norm_num_groups=32
            # attention_head_dim=attention_head_dim=8
            # resnet_time_scale_shift=resnet_time_scale_shift='default'
            # upsample_type=upsample_type='conv'
            # dropout=dropout=
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=(
                    attention_head_dim
                    if attention_head_dim is not None
                    else output_channel
                ),
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)

        # out
        # norm_num_groups=32
        # num_groups_out=32
        num_groups_out = (
            norm_num_groups
            if norm_num_groups is not None
            else min(block_out_channels[0] // 4, 32)
        )
        # num_channels=block_out_channels[0]=128
        # num_groups=num_groups_out=32
        # eps=norm_eps=1e-05
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=num_groups_out,
            eps=norm_eps,
        )
        self.conv_act = nn.SiLU()
        # block_out_channels[0]=128,
        # out_channels=3,
        # kernel_size=3,
        # padding=1,
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        # self.config.center_input_sample=False
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        # timestep = tensor([ 15, 333, 894, 690, 691, 808, 653, 645, 120,  44, 289, 868, 955, 688, 662, 969], device='cuda:0')
        # timestep.shape = torch.Size([16])
        timesteps = timestep
        # THIS BRANCH ON TRAIN
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps],
                dtype=torch.long,
                device=sample.device,
            )
        # THIS BRANCH ON INFERENCE
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps=tensor([781, 853, 596, 787, 581, 787, 421,  18, 535, 545, 568, 373, 693, 402, 131, 972], device='cuda:0')
        # sample.shape=torch.Size([16, 3, 64, 64])
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )
        # t_emb.shape=torch.Size([16, 128])
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        # emb.shape=torch.Size([16, 512])
        emb = self.time_embedding(t_emb)

        # self.class_embedding=None
        # class_labels=None

        # 2. pre-process
        # sample.shape=torch.Size([16, 3, 64, 64])
        # skip_sample.shape=torch.Size([16, 3, 64, 64])
        skip_sample = sample
        # sample.shape=torch.Size([16, 128, 64, 64])
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        # self.down_blocks=ModuleList(
        # (0-1): 2 x DownBlock2D(
        #     (resnets): ModuleList(
        #     (0-1): 2 x ResnetBlock2D(
        #         (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
        #         (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=128, bias=True)
        #         (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #     )
        #     )
        #     (downsamplers): ModuleList(
        #     (0): Downsample2D(
        #         (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     )
        #     )
        # )
        # (2): DownBlock2D(
        #     (resnets): ModuleList(
        #     (0): ResnetBlock2D(
        #         (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
        #         (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     (1): ResnetBlock2D(
        #         (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #     )
        #     )
        #     (downsamplers): ModuleList(
        #     (0): Downsample2D(
        #         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     )
        #     )
        # )
        # (3): DownBlock2D(
        #     (resnets): ModuleList(
        #     (0-1): 2 x ResnetBlock2D(
        #         (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #     )
        #     )
        #     (downsamplers): ModuleList(
        #     (0): Downsample2D(
        #         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     )
        #     )
        # )
        # (4): AttnDownBlock2D(
        #     (attentions): ModuleList(
        #     (0-1): 2 x Attention(
        #         (group_norm): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (to_q): Linear(in_features=512, out_features=512, bias=True)
        #         (to_k): Linear(in_features=512, out_features=512, bias=True)
        #         (to_v): Linear(in_features=512, out_features=512, bias=True)
        #         (to_out): ModuleList(
        #         (0): Linear(in_features=512, out_features=512, bias=True)
        #         (1): Dropout(p=0.0, inplace=False)
        #         )
        #     )
        #     )
        #     (resnets): ModuleList(
        #     (0): ResnetBlock2D(
        #         (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
        #         (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     (1): ResnetBlock2D(
        #         (norm1): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
        #         (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #     )
        #     )
        #     (downsamplers): ModuleList(
        #     (0): Downsample2D(
        #         (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     )
        #     )
        # )
        # (5): DownBlock2D(
        #     (resnets): ModuleList(
        #     (0-1): 2 x ResnetBlock2D(
        #         (norm1): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
        #         (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #     )
        #     )
        # )
        # )

        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        # self.mid_block=UNetMidBlock2D(
        #   (attentions): ModuleList(
        #     (0): Attention(
        #       (group_norm): GroupNorm(32, 512, eps=1e-05, affine=True)
        #       (to_q): Linear(in_features=512, out_features=512, bias=True)
        #       (to_k): Linear(in_features=512, out_features=512, bias=True)
        #       (to_v): Linear(in_features=512, out_features=512, bias=True)
        #       (to_out): ModuleList(
        #         (0): Linear(in_features=512, out_features=512, bias=True)
        #         (1): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #   )
        #   (resnets): ModuleList(
        #     (0-1): 2 x ResnetBlock2D(
        #       (norm1): GroupNorm(32, 512, eps=1e-05, affine=True)
        #       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
        #       (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
        #       (dropout): Dropout(p=0.0, inplace=False)
        #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #       (nonlinearity): SiLU()
        #     )
        #   )
        # )
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        # 5. up
        # self.up_blocks=ModuleList(
        # (0): UpBlock2D(
        #     (resnets): ModuleList(
        #     (0-2): 3 x ResnetBlock2D(
        #         (norm1): GroupNorm(32, 1024, eps=1e-05, affine=True)
        #         (conv1): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
        #         (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     )
        #     (upsamplers): ModuleList(
        #     (0): Upsample2D(
        #         (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     )
        # )
        # (1): AttnUpBlock2D(
        #     (attentions): ModuleList(
        #     (0-2): 3 x Attention(
        #         (group_norm): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (to_q): Linear(in_features=512, out_features=512, bias=True)
        #         (to_k): Linear(in_features=512, out_features=512, bias=True)
        #         (to_v): Linear(in_features=512, out_features=512, bias=True)
        #         (to_out): ModuleList(
        #         (0): Linear(in_features=512, out_features=512, bias=True)
        #         (1): Dropout(p=0.0, inplace=False)
        #         )
        #     )
        #     )
        #     (resnets): ModuleList(
        #     (0-1): 2 x ResnetBlock2D(
        #         (norm1): GroupNorm(32, 1024, eps=1e-05, affine=True)
        #         (conv1): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
        #         (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     (2): ResnetBlock2D(
        #         (norm1): GroupNorm(32, 768, eps=1e-05, affine=True)
        #         (conv1): Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
        #         (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     )
        #     (upsamplers): ModuleList(
        #     (0): Upsample2D(
        #         (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     )
        # )
        # (2): UpBlock2D(
        #     (resnets): ModuleList(
        #     (0): ResnetBlock2D(
        #         (norm1): GroupNorm(32, 768, eps=1e-05, affine=True)
        #         (conv1): Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     (1-2): 2 x ResnetBlock2D(
        #         (norm1): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     )
        #     (upsamplers): ModuleList(
        #     (0): Upsample2D(
        #         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     )
        # )
        # (3): UpBlock2D(
        #     (resnets): ModuleList(
        #     (0-1): 2 x ResnetBlock2D(
        #         (norm1): GroupNorm(32, 512, eps=1e-05, affine=True)
        #         (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     (2): ResnetBlock2D(
        #         (norm1): GroupNorm(32, 384, eps=1e-05, affine=True)
        #         (conv1): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     )
        #     (upsamplers): ModuleList(
        #     (0): Upsample2D(
        #         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     )
        # )
        # (4): UpBlock2D(
        #     (resnets): ModuleList(
        #     (0): ResnetBlock2D(
        #         (norm1): GroupNorm(32, 384, eps=1e-05, affine=True)
        #         (conv1): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=128, bias=True)
        #         (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     (1-2): 2 x ResnetBlock2D(
        #         (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=128, bias=True)
        #         (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     )
        #     (upsamplers): ModuleList(
        #     (0): Upsample2D(
        #         (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     )
        # )
        # (5): UpBlock2D(
        #     (resnets): ModuleList(
        #     (0-2): 3 x ResnetBlock2D(
        #         (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (time_emb_proj): Linear(in_features=512, out_features=128, bias=True)
        #         (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
        #         (dropout): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (nonlinearity): SiLU()
        #         (conv_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     )
        # )
        # )
        skip_sample = None
        # len(down_block_res_samples)=18
        # len(upsample_block.resnets)=3
        # прогоняем через upsample блоки
        # поочередно с конца
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            # [1, 2, 3, 4, 5, 6, 7, 8][-3:]==[6, 7, 8]
            # [1, 2, 3, 4, 5, 6, 7, 8][:-3]==[1, 2, 3, 4, 5]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        # sample.shape=torch.Size([16, 128, 64, 64])
        sample = self.conv_norm_out(sample)
        # sample.shape=torch.Size([16, 128, 64, 64])
        sample = self.conv_act(sample)
        # sample.shape=torch.Size([16, 3, 64, 64])
        sample = self.conv_out(sample)

        # skip_sample=None
        if skip_sample is not None:
            sample += skip_sample

        # not return_dict=False
        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)


class DDPMPipeline(DiffusionPipeline):
    r"""
    ok Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: UNet2DModel, scheduler: DDPMScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        # self.unet.config.sample_size=64
        # batch_size=16
        # self.unet.config.in_channels=3
        # self.unet.config.sample_size=64
        # image_shape=(16, 3, 64, 64)
        # if isinstance(self.unet.config.sample_size, int):
        image_shape = (
            batch_size,
            self.unet.config.in_channels,
            self.unet.config.sample_size,
            self.unet.config.sample_size,
        )

        # image=torch.Size([16, 3, 64, 64])
        image = randn_tensor(
            image_shape,
            generator=generator,
            device=self.device,
            dtype=self.unet.dtype,
        )

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        # self.scheduler.timesteps[:10]=tensor([999, 998, 997, 996, 995, 994, 993, 992, 991, 990])
        # image=torch.Size([16, 3, 64, 64])
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            # model_output=torch.Size([16, 3, 64, 64])
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            # t=999
            # image=torch.Size([16, 3, 64, 64])
            # model_output=torch.Size([16, 3, 64, 64])
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample
            # break
        # не знаю почему еще раз изображение делят на 2 и прибавляют 0.5
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class RMSNorm(nn.Module):
    r"""
    ok RMS Norm as introduced in https://arxiv.org/abs/1910.07467 by Zhang et al.

    Args:
        dim (`int`): Number of dimensions to use for `weights`. Only effective when `elementwise_affine` is True.
        eps (`float`): Small value to use when calculating the reciprocal of the square-root.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        bias (`bool`, defaults to False): If also training the `bias` param.
    """

    def __init__(
        self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False
    ):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class ResnetBlock2D(nn.Module):
    r"""ok
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" for a
            stronger conditioning with scale and shift.
        kernel (`torch.Tensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        kernel: Optional[torch.Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        # self.up=False
        # self.down=False
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True
        )

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = torch.nn.GroupNorm(
            num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(
            out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1
        )
        # self.nonlinearity=SiLU()
        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        # self.use_in_shortcut=False
        self.use_in_shortcut = (
            self.in_channels != conv_2d_out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: torch.Tensor,
        temb: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # if len(args) > 0 or kwargs.get("scale", None) is not None:
        #     deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        #     deprecate("scale", "1.0.0", deprecation_message)
        # temb=torch.Size([16, 512])
        hidden_states = input_tensor

        # torch.Size([16, 128, 64, 64])
        hidden_states = self.norm1(hidden_states)
        # torch.Size([16, 128, 64, 64])
        hidden_states = self.nonlinearity(hidden_states)
        # torch.Size([16, 128, 64, 64])
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
        # self.time_embedding_norm="default"

        # temb=torch.Size([16, 128, 1, 1])
        if temb is not None:
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        # torch.Size([16, 128, 64, 64])
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor.contiguous())
        # input_tensor=torch.Size([4, 128, 64, 64])
        # hidden_states=torch.Size([4, 128, 64, 64])
        # self.output_scale_factor=1.0
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    ok This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    """
    ok
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        self.cond_proj = None

        self.act = get_activation(act_fn)

        time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        self.post_act = None

    def forward(self, sample, condition=None):

        sample = self.linear_1(sample)

        sample = self.act(sample)

        sample = self.linear_2(sample)

        return sample


class Timesteps(nn.Module):
    """
    ok
    """

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


@dataclass
class Args:
    resolution: int = 64
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = "linear"
    learning_rate: float = 1e-4
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-6
    adam_epsilon: float = 1e-08
    dataset_name: str = "huggan/flowers-102-categories"
    dataset_config_name: str = None
    cache_dir: str = None
    ema_max_decay: float = 0.9999
    ema_inv_gamma: float = 1.0
    ema_power: float = 3 / 4
    train_batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    num_epochs: int = 1
    center_crop: bool = False
    random_flip: bool = False
    dataloader_num_workers: int = 0
    use_ema: bool = True
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = None
    output_dir: str = "minimal_diffusion/unconditional_diffusion/ddpm-model-64"
    save_images_epochs: int = 10
    save_model_epochs: int = 1
    prediction_type: str = "epsilon"
    logging_dir: str = "logs"
    mixed_precision: str = "no"
    logger: str = "wandb"
    ddpm_num_inference_steps: int = 1000
    project_name: str = "train_unconditional"


args = Args()
##d##d#######################d#d#d#d####################d
##d##d#######################d#d#d#d####################d
##d##d#######################d#d#d#d####################d
##d##d#######################d#d#d#d####################d
##d##d#######################d#d#d#d####################d
model = UNet2DModel(
    sample_size=args.resolution,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
model = model.cuda()


ema_model = EMAModel(
    model.parameters(),
    decay=args.ema_max_decay,
    use_ema_warmup=True,
    inv_gamma=args.ema_inv_gamma,
    power=args.ema_power,
    model_cls=UNet2DModel,
    model_config=model.config,
)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=args.ddpm_num_steps,
    beta_schedule=args.ddpm_beta_schedule,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

dataset = load_dataset(
    args.dataset_name,
    args.dataset_config_name,
    cache_dir=args.cache_dir,
    split="train",
)
# dataset = dataset.train_test_split(
#     test_size=4,
#     seed=42,
# )["test"]

augmentations = transforms.Compose(
    [
        transforms.Resize(
            args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        (
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution)
        ),
        (
            transforms.RandomHorizontalFlip()
            if args.random_flip
            else transforms.Lambda(lambda x: x)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5],
            [0.5],
        ),
    ]
)


def transform_images(examples):
    images = [augmentations(image.convert("RGB")) for image in examples["image"]]
    return {"input": images}


dataset.set_transform(transform_images)
train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=args.dataloader_num_workers,
)

# Initialize the learning rate scheduler
lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)


total_batch_size = args.train_batch_size * args.gradient_accumulation_steps
num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / args.gradient_accumulation_steps
)
max_train_steps = args.num_epochs * num_update_steps_per_epoch


print("***** Running training *****")
print(f"  Num examples = {len(dataset)}")
print(f"  Num Epochs = {args.num_epochs}")
print(f"  Instantaneous batch size per device = {args.train_batch_size}")
print(
    f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
)
print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
print(f"  Total optimization steps = {max_train_steps}")

global_step = 0
first_epoch = 0

weight_dtype = torch.float32
# model.enable_xformers_memory_efficient_attention()
logging_dir = os.path.join(args.output_dir, args.logging_dir)
accelerator_project_config = ProjectConfiguration(
    project_dir=args.output_dir, logging_dir=logging_dir
)
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.logger,
    project_config=accelerator_project_config,
    # kwargs_handlers=[kwargs],
)
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

if accelerator.is_main_process:
    run = os.path.split(args.project_name)[-1].split(".")[0]
    accelerator.init_trackers(run)

# model = torch.compile(model)

# Train!
# with torch.profiler.profile() as prof:
for epoch in range(first_epoch, args.num_epochs):
    model.train()
    progress_bar = tqdm(
        total=num_update_steps_per_epoch,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in enumerate(train_dataloader):
        # Skip steps until we reach the resumed step
        # clean_images=torch.Size([16, 3, 64, 64])
        clean_images = batch["input"].to(weight_dtype)
        # Sample noise that we'll add to the images
        # noise=torch.Size([16, 3, 64, 64])
        noise = torch.randn(
            clean_images.shape, dtype=weight_dtype, device=clean_images.device
        )
        # bsz=16
        bsz = clean_images.shape[0]
        # Sample a random timestep for each image
        # timesteps=torch.Size([16])
        # timesteps=tensor([563, 667, 202, 694, 151, 681, 668, 757, 174, 408, 989, 731, 189, 907, 464, 640], device='cuda:0')
        # noise_scheduler.config.num_train_timesteps=1000
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # noisy_images=torch.Size([16, 3, 64, 64])
        # clean_images=torch.Size([16, 3, 64, 64])
        # noise=torch.Size([16, 3, 64, 64])
        # timesteps=torch.Size([16])
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        with accelerator.accumulate(model):
            # Predict the noise residual
            model_output = model(noisy_images, timesteps).sample

            loss = F.mse_loss(
                model_output.float(),
                noise.float(),
            )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        # prof.step()
        # if step > 2:
        #     break

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if args.use_ema:
                ema_model.step(model.parameters())
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    print(f"Saved state to {save_path}")

        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
        }
        if args.use_ema:
            logs["ema_decay"] = ema_model.cur_decay_value
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
    progress_bar.close()

    accelerator.wait_for_everyone()

    # Generate sample images for visual inspection
    # break
    if accelerator.is_main_process:
        if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
            unet = accelerator.unwrap_model(model)

            if args.use_ema:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())
            # unet=UNet2DModel
            # scheduler=noise_scheduler=<DDPMScheduler, len() = 1000>
            pipeline = DDPMPipeline(
                unet=unet,
                scheduler=noise_scheduler,
            )

            generator = torch.Generator(device=pipeline.device).manual_seed(0)
            # run pipeline in inference (sample random noise and denoise)
            # args.ddpm_num_inference_steps=1000
            images = pipeline(
                generator=generator,
                batch_size=args.eval_batch_size,
                num_inference_steps=args.ddpm_num_inference_steps,
                output_type="np",
            ).images

            if args.use_ema:
                ema_model.restore(unet.parameters())

            # denormalize the images and save to tensorboard
            images_processed = (images * 255).round().astype("uint8")

            if args.logger == "wandb":
                # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                accelerator.get_tracker("wandb").log(
                    {
                        "test_samples": [wandb.Image(img) for img in images_processed],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            # save the model
            unet = accelerator.unwrap_model(model)

            if args.use_ema:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())

            pipeline = DDPMPipeline(
                unet=unet,
                scheduler=noise_scheduler,
            )

            pipeline.save_pretrained(args.output_dir)

            if args.use_ema:
                ema_model.restore(unet.parameters())
        # break


accelerator.end_training()
# prof.export_chrome_trace("trace_accelerate.json")
