# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod


from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle

from torchtitan.models.moe.utils import _permute, _unpermute

try:
    from deep_ep import Buffer
    DEEP_EP_AVAILABLE = True
except ImportError:
    DEEP_EP_AVAILABLE = False
    Buffer = None

# Registry to track modules with DeepEP enabled
# Using WeakSet so modules can be garbage collected
# _DEEP_EP_MODULES: weakref.WeakSet = weakref.WeakSet()

# Shared buffer cache to avoid allocating multiple buffers
# Key: (group_name, hidden_dim) -> Buffer
_DEEP_EP_BUFFER_CACHE: dict = {}

# Simple cache for autograd state (avoids save_for_backward SAC issues)
# Key: id(handle) -> state_tuple
_AUTOGRAD_STATE_CACHE: dict = {}


def clear_autograd_cache():
    """Clear cache at end of training step to prevent leaks."""
    _AUTOGRAD_STATE_CACHE.clear()


@dataclass
class DeepEPDispatchState:
    """
    Context object to store DeepEP dispatch state.
    This replaces the many individual _deep_ep_* attributes on the module.
    """
    # Original routing info (for backward pass), storing results from router for ep hooks to use
    original_x: torch.Tensor = None
    topk_idx: torch.Tensor = None
    topk_weights: torch.Tensor = None

    # Dispatch metadata, received from deepep dispatch function after dispatch,
    handle: any = None
    recv_topk_idx: torch.Tensor = None
    recv_topk_weights: torch.Tensor = None
    num_recv_tokens_per_expert_list: torch.Tensor = None

    # Token reordering metadata, this is stored for combine to use
    local_token_indices: torch.Tensor = None
    local_slot_indices: torch.Tensor = None
    sorted_order: torch.Tensor = None
    expanded_weights: torch.Tensor = None
    num_unique_tokens: int = None

    # Overlap mode state
    recv_x: torch.Tensor = None
    overlap_mode: bool = False


class BaseExpertParallel(ParallelStyle, ABC):
    @abstractmethod
    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        ...

    @abstractmethod
    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        ...


# implementation of Tensor Parallel for the GroupedExperts in MoE
class TensorParallel(ParallelStyle):
    def _prepare_input_fn(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        # NOTE: Currently in MoE TP, experts multiplication runs in plain Tensors.
        #       The grad_placements on inputs is set to Partial so that necessary
        #       reductions are performed during backward.
        routed_input = DTensor.from_local(
            routed_input, device_mesh, (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        return routed_input, num_tokens_per_expert

    def _partition_fn(self, name, module, device_mesh):
        # w1 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        module.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)])),
        )  # Column-wise sharding

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            # pyrefly: ignore [bad-argument-type]
            self._prepare_input_fn,
        )


class ExpertParallel(BaseExpertParallel):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape = None
        self.permuted_indices = None

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        # perform all-to-all
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        # We need to perform another shuffle to get the correct layout, via the _permute function
        # below, which also does padding to make sure the number of tokens each expert gets locally
        # is a multiple of TOKEN_GROUP_ALIGN_SIZE_M.
        # Note that this will create side effects when wrapping the for-loop implementation
        # of GroupedExperts, as it does not need padding.

        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
        ) = _permute(
            routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
        )

        return routed_input, num_tokens_per_expert_group

    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        routed_output = _unpermute(
            routed_output, self.input_shape, self.permuted_indices
        )

        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            # pyrefly: ignore [bad-argument-type]
            input_fn=self._token_dispatch,
            # pyrefly: ignore [bad-argument-type]
            output_fn=self._token_combine,
        )


# This class is for dp2ep with TP (without TP we can just use ExpertParallel)
class ExpertTensorParallel(ExpertParallel):
    def _token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs

        # NOTE: Currently in MoE TP, experts multiplication runs in plain Tensors.
        #       The grad_placements on inputs is set to Partial so that necessary
        #       reductions are performed during backward.
        routed_input = DTensor.from_local(
            routed_input, device_mesh["tp"], (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        inputs = (routed_input, num_tokens_per_expert)

        # token dispatch happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_dispatch(mod, inputs, device_mesh["ep"])

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        # w1 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w1",
            # pyrefly: ignore [bad-argument-type]
            nn.Parameter(distribute_tensor(mod.w1, device_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        mod.register_parameter(
            "w2",
            # pyrefly: ignore [bad-argument-type]
            nn.Parameter(distribute_tensor(mod.w2, device_mesh, [Shard(0), Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w3",
            # pyrefly: ignore [bad-argument-type]
            nn.Parameter(distribute_tensor(mod.w3, device_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

    def _token_combine(self, mod, routed_output, device_mesh):
        # token combine happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_combine(mod, routed_output, device_mesh["ep"])

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            # pyrefly: ignore [bad-argument-type]
            input_fn=self._token_dispatch,
            # pyrefly: ignore [bad-argument-type]
            output_fn=self._token_combine,
        )


# This class is to support Sequence Parallel for ETP=1
# when EP borrows from all TP and part of DP
class ReordererSequenceParallel(ParallelStyle):
    def __init__(self):
        super().__init__()

    def _prepare_inputput_fn(self, mod, inputs, device_mesh):
        # shape (batch_size*seq_len, top_k)
        top_scores, selected_experts_indices = inputs
        num_tokens, _ = top_scores.shape

        # NOTE: If needed, we can pad tokens in case bs*slen is not divisible by TP degree
        # if top_scores.shape[0] % device_mesh.size() != 0:
        #     num_tokens = top_scores.shape[0]
        #     tp_size = device_mesh.size()
        #     n_pad = (num_tokens // tp_size + 1) * tp_size - num_tokens
        #     selected_experts_indices = F.pad(selected_experts_indices, [0, 0, 0, n_pad])
        #     top_scores = F.pad(top_scores, [0, 0, 0, n_pad])

        def _split_along_first_dim(x: torch.Tensor) -> torch.Tensor:
            assert x.is_contiguous()
            if num_tokens % device_mesh.size() != 0:
                raise ValueError(
                    "Uneven split of tokens of is not supported yet. "
                    "Requires EP degree dividing batch size * seq len."
                )
            local_num_tokens = num_tokens // device_mesh.size()
            local_rank = device_mesh.get_local_rank()
            offset = local_rank * local_num_tokens
            output = x[offset : offset + local_num_tokens]

            return output

        top_scores = _split_along_first_dim(top_scores)
        selected_experts_indices = _split_along_first_dim(selected_experts_indices)

        # shape (batch_size * seq_len // ep_degree, top_k)
        return top_scores, selected_experts_indices

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        # shape (batch_size * seq_len * top_k // ep_degree)
        top_scores, token_indices_experts_sorted, num_tokens_per_expert = outputs

        # NOTE: As we shard routed tokens along bs*slen dim across the TP ranks,
        #       the MoE gather and scatter still require global token indices.
        local_rank = device_mesh.get_local_rank()
        token_indices_experts_sorted = (
            token_indices_experts_sorted + top_scores.shape[0] * local_rank
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            # pyrefly: ignore [bad-argument-type]
            input_fn=self._prepare_inputput_fn,
            # pyrefly: ignore [bad-argument-type]
            output_fn=self._prepare_output_fn,
        )


# ============================================================================
# Cross-Layer Overlap Support for DeepEP
# ============================================================================
# These classes and functions enable overlapping DeepEP communication with
# computation from the same or adjacent layers:
#
# 1. Dispatch can overlap with shared_experts computation
# 2. Combine can potentially overlap with next layer's attention
#
# Usage pattern in MoE.forward():
#   ctx = deep_ep_dispatch_async(x, topk_idx, topk_weights, buffer, num_experts)
#   shared_out = shared_experts(x)  # Overlaps with dispatch
#   recv_x, handle = deep_ep_wait_dispatch(ctx)
#   expert_out = expert_compute(recv_x)
#   combined = deep_ep_combine(expert_out, handle, buffer)
# ============================================================================


class DeepEPDispatchContext:
    """Context holder for async DeepEP dispatch operation."""
    def __init__(self):
        self.event = None
        self.recv_x = None
        self.recv_topk_idx = None
        self.recv_topk_weights = None
        self.num_recv_tokens_per_expert_list = None
        self.handle = None
        self.buffer = None
        self.topk_idx = None
        self.topk_weights = None
        self.num_experts = None


class _DispatchContext:
    """Holder for dispatch outputs that aren't tensor results."""
    recv_topk_idx: torch.Tensor = None
    recv_topk_weights: torch.Tensor = None
    num_recv_tokens_per_expert_list: list = None
    handle: tuple = None


def deep_ep_dispatch_async(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buffer: "Buffer",
    num_experts: int,
) -> DeepEPDispatchContext:
    """
    Start an async DeepEP dispatch operation.

    This launches the dispatch on the communication stream and returns immediately,
    allowing compute to proceed on the main stream while communication runs.

    Args:
        x: Input tensor (bs*slen, dim)
        topk_idx: Expert indices (bs*slen, top_k)
        topk_weights: Expert weights (bs*slen, top_k)
        buffer: DeepEP buffer
        num_experts: Total number of experts

    Returns:
        DeepEPDispatchContext: Context to pass to deep_ep_wait_dispatch
    """
    ctx = DeepEPDispatchContext()
    ctx.buffer = buffer
    ctx.topk_idx = topk_idx
    ctx.topk_weights = topk_weights
    ctx.num_experts = num_experts

    # Calculate layout
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert_layout, \
        is_token_in_rank, _ = buffer.get_dispatch_layout(topk_idx, num_experts)

    # Dispatch tokens asynchronously
    # WORKAROUND: async_finish=False to avoid record_stream() incompatibility with PyTorch nightly
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, \
        handle, event = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert_layout,
            async_finish=True,  # Changed from True: PyTorch nightly compatibility
            allocate_on_comm_stream=False,  # Allocate on compute stream for safety
        )

    ctx.event = event
    ctx.recv_x = recv_x
    ctx.recv_topk_idx = recv_topk_idx
    ctx.recv_topk_weights = recv_topk_weights
    ctx.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
    ctx.handle = handle

    return ctx


def deep_ep_wait_dispatch(ctx: DeepEPDispatchContext) -> tuple:
    """
    Wait for async dispatch to complete and return results.

    Args:
        ctx: Context from deep_ep_dispatch_async

    Returns:
        tuple: (recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle)
    """
    if ctx.event is not None:
        ctx.event.current_stream_wait()

    return (
        ctx.recv_x,
        ctx.recv_topk_idx,
        ctx.recv_topk_weights,
        ctx.num_recv_tokens_per_expert_list,
        ctx.handle,
    )


class DeepEPCombineContext:
    """Context holder for async DeepEP combine operation."""
    def __init__(self):
        self.event = None
        self.combined_x = None


def deep_ep_combine_async(
    x: torch.Tensor,
    handle,
    buffer: "Buffer",
    topk_weights: torch.Tensor = None,
) -> DeepEPCombineContext:
    """
    Start an async DeepEP combine operation.

    This launches the combine on the communication stream and returns immediately,
    allowing compute to proceed on the main stream while communication runs.

    Args:
        x: Expert output tensor
        handle: Handle from dispatch
        buffer: DeepEP buffer
        topk_weights: Optional weights for weighted combine

    Returns:
        DeepEPCombineContext: Context to pass to deep_ep_wait_combine
    """
    ctx = DeepEPCombineContext()

    # WORKAROUND: async_finish=False to avoid record_stream() incompatibility with PyTorch nightly
    combined_x, _, event = buffer.combine(
        x,
        handle,
        topk_weights=topk_weights,
        async_finish=True,  # Changed from True: PyTorch nightly compatibility
        allocate_on_comm_stream=False,
    )

    ctx.event = event
    ctx.combined_x = combined_x

    return ctx


def deep_ep_wait_combine(ctx: DeepEPCombineContext) -> torch.Tensor:
    """
    Wait for async combine to complete and return result.

    Args:
        ctx: Context from deep_ep_combine_async

    Returns:
        torch.Tensor: Combined output
    """
    if ctx.event is not None:
        ctx.event.current_stream_wait()

    return ctx.combined_x


def get_or_create_deep_ep_buffer(
    group,
    hidden_dim: int,
    num_sms: int = 24,
) -> "Buffer":
    """
    Get or create a shared DeepEP buffer for the given group and hidden_dim.
    This avoids allocating multiple buffers for different MoE layers.
    """
    # Create a cache key based on group rank and hidden_dim
    cache_key = (id(group), hidden_dim)

    if cache_key in _DEEP_EP_BUFFER_CACHE:
        return _DEEP_EP_BUFFER_CACHE[cache_key]

    hidden_bytes = hidden_dim * 2  # BF16 = 2 bytes

    # Set SM count for DeepEP kernels
    Buffer.set_num_sms(num_sms)

    # Get buffer size hints from DeepEP configs
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
            num_nvl_bytes,
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
            num_rdma_bytes,
        )

    # Increase buffer sizes to handle larger token counts
    # The combine operation needs more buffer space than dispatch
    num_nvl_bytes = int(num_nvl_bytes * 2)
    num_rdma_bytes = int(num_rdma_bytes * 2)

    buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    _DEEP_EP_BUFFER_CACHE[cache_key] = buffer
    return buffer


def _vectorized_padding(
    x: torch.Tensor,
    original_counts: torch.Tensor,
    padded_counts: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized padding: insert zeros after each expert's tokens.

    Args:
        x: Input tensor of shape (total_original, hidden_dim)
        original_counts: Number of tokens per expert (num_experts,)
        padded_counts: Padded number of tokens per expert (num_experts,)

    Returns:
        Padded tensor of shape (total_padded, hidden_dim)
    """
    total_original = x.shape[0]
    total_padded = int(padded_counts.sum().item())

    if total_padded == total_original:
        return x

    device = x.device
    dtype = x.dtype
    hidden_dim = x.shape[1]

    # Compute exclusive cumulative sums for source and destination offsets
    src_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        original_counts.cumsum(0)[:-1].long()
    ])
    dst_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        padded_counts.cumsum(0)[:-1].long()
    ])

    # Create expert_id for each source token using repeat_interleave
    # This avoids CPU-GPU sync by staying on GPU
    expert_ids = torch.repeat_interleave(
        torch.arange(len(original_counts), device=device),
        original_counts.long()
    )

    # Source indices: 0, 1, 2, ..., total_original-1
    src_indices = torch.arange(total_original, device=device)

    # Position within each expert = src_index - src_offset[expert_id]
    positions_in_expert = src_indices - src_offsets[expert_ids]

    # Destination indices = dst_offset[expert_id] + position_in_expert
    dst_indices = dst_offsets[expert_ids] + positions_in_expert

    # Create output tensor and scatter
    padded = torch.zeros(total_padded, hidden_dim, dtype=dtype, device=device)
    padded[dst_indices] = x

    return padded


def _vectorized_unpadding(
    x: torch.Tensor,
    original_counts: torch.Tensor,
    padded_counts: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized unpadding: extract original tokens from padded layout.

    Args:
        x: Padded tensor of shape (total_padded, hidden_dim)
        original_counts: Number of tokens per expert (num_experts,)
        padded_counts: Padded number of tokens per expert (num_experts,)

    Returns:
        Unpadded tensor of shape (total_original, hidden_dim)
    """
    total_original = int(original_counts.sum().item())
    total_padded = x.shape[0]

    if total_padded == total_original:
        return x

    device = x.device

    # Compute exclusive cumulative sums
    src_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        padded_counts.cumsum(0)[:-1].long()
    ])
    dst_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        original_counts.cumsum(0)[:-1].long()
    ])

    # Create expert_id for each destination token
    expert_ids = torch.repeat_interleave(
        torch.arange(len(original_counts), device=device),
        original_counts.long()
    )

    # Destination indices: 0, 1, 2, ..., total_original-1
    dst_indices = torch.arange(total_original, device=device)

    # Position within each expert = dst_index - dst_offset[expert_id]
    positions_in_expert = dst_indices - dst_offsets[expert_ids]

    # Source indices = src_offset[expert_id] + position_in_expert
    src_indices = src_offsets[expert_ids] + positions_in_expert

    # Gather from padded tensor
    unpadded = x[src_indices]

    return unpadded


# ============================================================================
# DeepEP Autograd Functions with Selective Activation Checkpointing Support
# ============================================================================
# With op-based SAC, the layer may be recomputed during backward. To avoid
# expensive communication recomputation, we cache results at the module level.
# ============================================================================


class _DeepEPDispatch(torch.autograd.Function):
    """Autograd function for DeepEP dispatch operation.

    Forward: dispatch (send tokens to experts)
    Backward: combine (aggregate gradients back)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        buffer: "Buffer",
        num_experts: int,
        dispatch_ctx: any,
    ) -> torch.Tensor:
        """
        Forward pass: dispatch tokens to experts.
        Returns only recv_x - other outputs stored in dispatch_ctx for retrieval.

        OPTIMIZATION: Cache communication results to avoid recomputing expensive
        all-to-all when op-based SAC recomputes the layer.
        """
        # Create cache key based on input tensor
        cache_key = (id(x), x.data_ptr())

        # Check if we've already computed this (SAC recomputation case)
        if hasattr(buffer, '_dispatch_cache') and cache_key in buffer._dispatch_cache:
            # Reuse cached results - no communication!
            cached = buffer._dispatch_cache[cache_key]
            recv_x = cached['recv_x']
            dispatch_ctx.recv_topk_idx = cached['recv_topk_idx']
            dispatch_ctx.recv_topk_weights = cached['recv_topk_weights']
            dispatch_ctx.num_recv_tokens_per_expert_list = cached['num_recv_tokens_per_expert_list']
            dispatch_ctx.handle = cached['handle']
            handle = cached['handle']
        else:
            # First time: do the actual communication
            # Calculate layout
            num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert_layout, \
                is_token_in_rank, _ = buffer.get_dispatch_layout(
                    topk_idx, num_experts,
                )

            # Dispatch tokens
            recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, \
                handle, _ = buffer.dispatch(
                    x,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    num_tokens_per_rank=num_tokens_per_rank,
                    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                    is_token_in_rank=is_token_in_rank,
                    num_tokens_per_expert=num_tokens_per_expert_layout,
                )

            # Cache results to avoid recomputation
            if not hasattr(buffer, '_dispatch_cache'):
                buffer._dispatch_cache = {}
            buffer._dispatch_cache[cache_key] = {
                'recv_x': recv_x,
                'recv_topk_idx': recv_topk_idx,
                'recv_topk_weights': recv_topk_weights,
                'num_recv_tokens_per_expert_list': num_recv_tokens_per_expert_list,
                'handle': handle,
            }

            # Store metadata in dispatch_ctx for retrieval by caller
            dispatch_ctx.recv_topk_idx = recv_topk_idx
            dispatch_ctx.recv_topk_weights = recv_topk_weights
            dispatch_ctx.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
            dispatch_ctx.handle = handle

        # Store for backward - use simple global cache
        cache_key = id(handle)
        _AUTOGRAD_STATE_CACHE[cache_key] = (buffer, handle)
        ctx.cache_key = cache_key

        return recv_x

    @staticmethod
    def backward(ctx, grad_recv_x):
        """
        Backward pass: combine gradients back.
        The backward of dispatch is combine.
        """
        # Pop from cache (removes entry)
        buffer, handle = _AUTOGRAD_STATE_CACHE.pop(ctx.cache_key)

        # Combine gradients back
        combined_grad_x, _, _ = buffer.combine(
            grad_recv_x.contiguous(),
            handle,
        )

        return combined_grad_x, None, None, None, None, None


class _DeepEPCombine(torch.autograd.Function):
    """Autograd function for DeepEP combine operation.

    Forward: combine (aggregate outputs from experts)
    Backward: dispatch (send gradients to experts using saved routing info)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        handle,
        buffer: "Buffer",
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
    ):
        """
        Forward pass: combine tokens from experts.
        """
        x = x.contiguous()
        combined_x, _, _ = buffer.combine(x, handle)

        # Store for backward - use simple global cache
        cache_key = id(handle)
        _AUTOGRAD_STATE_CACHE[cache_key] = (handle, buffer, num_experts, topk_idx, topk_weights)
        ctx.cache_key = cache_key

        return combined_x

    @staticmethod
    def backward(ctx, grad_combined_x):
        """
        Backward pass: dispatch gradients back to experts.
        """
        # Pop from cache (removes entry)
        handle, buffer, num_experts, topk_idx, topk_weights = _AUTOGRAD_STATE_CACHE.pop(ctx.cache_key)

        # Calculate layout for backward dispatch
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert_layout, \
            is_token_in_rank, _ = buffer.get_dispatch_layout(topk_idx, num_experts)

        # Dispatch gradients back to experts
        grad_x, _, _, _, _, _ = buffer.dispatch(
            grad_combined_x.contiguous(),
            topk_idx=topk_idx.contiguous(),
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert_layout,
        )

        return grad_x, None, None, None, None, None


class DeepEPExpertParallel(ParallelStyle):
    """
    Expert Parallel using DeepEP for high-throughput all-to-all communication.

    This class replaces the NCCL-based ExpertParallel with DeepEP's optimized
    dispatch and combine kernels, providing:
    - High-throughput NVLink communication for intranode
    - High-throughput RDMA communication for internode
    - Optional FP8 support for dispatch
    - SM control for compute-communication overlap

    Args:
        num_experts: Total number of experts
        hidden_dim: Hidden dimension of the model
        num_sms: Number of SMs to use for DeepEP kernels (default: 24)
        enable_token_padding: Whether to pad tokens to TOKEN_GROUP_ALIGN_SIZE_M (default: False)
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        num_sms: int = 24,
        enable_token_padding: bool = False,
    ):
        super().__init__()
        if not DEEP_EP_AVAILABLE:
            raise RuntimeError(
                "DeepEP is not available. Please install it from "
                "https://github.com/deepseek-ai/DeepEP"
            )

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.num_sms = num_sms
        self.enable_token_padding = enable_token_padding
        self.buffer = None
        self.handle = None
        # For padding/unpadding compatibility with grouped_mm
        self.original_counts = None
        self.padded_counts = None
        self.original_total = None

    def _init_buffer(self, device_mesh: DeviceMesh):
        """Initialize DeepEP buffer lazily on first use, using shared buffer cache."""
        if self.buffer is not None:
            return

        group = device_mesh.get_group()
        # Use shared buffer to reduce memory usage across MoE layers
        self.buffer = get_or_create_deep_ep_buffer(
            group, self.hidden_dim, self.num_sms
        )

    def _token_dispatch(self, mod, inputs, device_mesh):
        """
        DeepEP-based token dispatch.

        This replaces the NCCL all-to-all with DeepEP's dispatch kernel.

        IMPORTANT: DeepEP expects the original input tensor x (shape: bs*slen, dim)
        not the pre-reordered routed_input (shape: bs*slen*top_k, dim).
        The original x should be stored on the module as _deep_ep_original_x.

        In overlap mode (_deep_ep_overlap_mode=True), the dispatch has already
        been performed asynchronously, and the results are available in
        _deep_ep_recv_x, _deep_ep_recv_topk_idx, etc.
        """
        routed_input, num_tokens_per_expert = inputs
        self._init_buffer(device_mesh)

        ep_degree = device_mesh.shape[0]
        num_local_experts = self.num_experts // ep_degree

        # Create or retrieve the context object
        # if not hasattr(mod, '_deep_ep_state'):
        #     mod._deep_ep_state = DeepEPDispatchState()
        state = mod._deep_ep_state

        # Check if we're in overlap mode (dispatch already done)
        # Only DeepEPMoE sets this attribute in overlap mode
        overlap_mode = state.overlap_mode

        if overlap_mode:
            # In overlap mode, use the pre-dispatched data
            recv_x = state.recv_x
            recv_topk_idx = state.recv_topk_idx
            recv_topk_weights = state.recv_topk_weights
            self.handle = state.handle

            if recv_x is None or self.handle is None:
                raise RuntimeError(
                    "Overlap mode requires recv_x and handle to be set. "
                    "Make sure dispatch was called before entering overlap mode."
                )
        else:
            # Standard mode: perform dispatch now
            # Get the original input, topk_idx and topk_weights from the state
            original_x = state.original_x
            topk_idx = state.topk_idx
            topk_weights = state.topk_weights

            # if original_x is None:
            #     raise RuntimeError(
            #         "DeepEP requires original_x to be set on the module state. "
            #         "Make sure to set mod._deep_ep_state.original_x before the forward pass."
            #     )
            # if topk_idx is None:
            #     raise RuntimeError(
            #         "DeepEP requires topk_idx to be set on the module state. "
            #         "Make sure to set mod._deep_ep_state.topk_idx before the forward pass."
            #     )

            # Use autograd-aware dispatch
            # dispatch_ctx = _DispatchContext()
            recv_x = _DeepEPDispatch.apply(
                original_x,
                topk_idx,
                topk_weights if topk_weights is not None else torch.ones(topk_idx.shape, dtype=original_x.dtype, device=original_x.device),
                self.buffer,
                self.num_experts,
                state,
            )

            # clean up buffer
            state.original_x = None
            # state.topk_idx = None
            # state.topk_weights = None

            # Retrieve metadata from dispatch context and store in state
            recv_topk_idx = state.recv_topk_idx
            recv_topk_weights = state.recv_topk_weights
            self.handle = state.handle

            # Store in state for combine
            # state.handle = self.handle
            # state.recv_topk_weights = recv_topk_weights
            # state.recv_topk_idx = recv_topk_idx

        # Rest of the processing is the same for both modes
        # DeepEP dispatch returns:
        # - recv_x: (num_unique_tokens, hidden_dim) - UNIQUE tokens received by this rank
        # - recv_topk_idx: (num_unique_tokens, top_k) - expert indices for each token
        # - recv_topk_weights: (num_unique_tokens, top_k) - weights for each token-expert pair
        # - num_recv_tokens_per_expert_list: counts of token-expert pairs per LOCAL expert
        #
        # For grouped_mm, we need to EXPAND recv_x so each token appears once per
        # local expert it's routed to. DeepEP guarantees tokens are strictly ordered by expert.

        # Get which experts are local to this rank
        local_rank = device_mesh.get_local_rank()
        local_expert_start = local_rank * num_local_experts
        local_expert_end = local_expert_start + num_local_experts

        # recv_topk_idx has shape (num_tokens, top_k) with global expert indices
        # Create a mask for which (token, expert) pairs are local
        is_local = (recv_topk_idx >= local_expert_start) & (recv_topk_idx < local_expert_end)

        # Get the indices of local (token, expert) pairs
        # local_token_indices: which token each pair comes from
        # local_slot_indices: which column (0 to top_k-1) in recv_topk_idx
        local_token_indices, local_slot_indices = torch.where(is_local)

        # Get the actual expert indices for local pairs (convert to local: 0 to num_local_experts-1)
        local_expert_idx = recv_topk_idx[local_token_indices, local_slot_indices] - local_expert_start

        # Expand recv_x: gather tokens based on local_token_indices
        # This creates one copy of each token per local expert it's routed to
        expanded_x = recv_x[local_token_indices]

        # Also expand the weights for combine later
        if recv_topk_weights is not None:
            expanded_weights = recv_topk_weights[local_token_indices, local_slot_indices]
        else:
            expanded_weights = None

        # Sort expanded tokens by local expert index
        sorted_order = torch.argsort(local_expert_idx)
        expanded_x_sorted = expanded_x[sorted_order]
        local_expert_idx_sorted = local_expert_idx[sorted_order]

        # Store info in state for combine
        state.local_token_indices = local_token_indices
        state.local_slot_indices = local_slot_indices
        state.sorted_order = sorted_order
        state.expanded_weights = expanded_weights[sorted_order] if expanded_weights is not None else None
        state.num_unique_tokens = recv_x.shape[0]

        # Count tokens per local expert
        num_tokens_per_expert_local = torch.bincount(
            local_expert_idx_sorted.long(), minlength=num_local_experts
        ).to(torch.int32)

        # Apply padding only if enabled
        if self.enable_token_padding:
            # Pad token counts for grouped_mm alignment (vectorized, no CPU-GPU sync)
            from torchtitan.models.moe.utils import TOKEN_GROUP_ALIGN_SIZE_M

            # Vectorized round-up: ((x + align - 1) // align) * align
            num_tokens_per_expert_padded = (
                (num_tokens_per_expert_local + TOKEN_GROUP_ALIGN_SIZE_M - 1)
                // TOKEN_GROUP_ALIGN_SIZE_M
            ) * TOKEN_GROUP_ALIGN_SIZE_M

            # Use vectorized padding (no Python loops, fully GPU-based)
            recv_x_padded = _vectorized_padding(
                expanded_x_sorted,
                num_tokens_per_expert_local,
                num_tokens_per_expert_padded,
            )

            # Store original counts for unpermute during combine
            self.original_counts = num_tokens_per_expert_local
            self.padded_counts = num_tokens_per_expert_padded
            self.original_total = expanded_x_sorted.shape[0]

            return recv_x_padded, num_tokens_per_expert_padded
        else:
            # No padding - use counts as-is
            return expanded_x_sorted, num_tokens_per_expert_local

    def _token_combine(self, mod, routed_output, device_mesh):
        """
        DeepEP-based token combine.

        This replaces the NCCL all-to-all with DeepEP's combine kernel.
        The combine operation returns output weighted by topk_weights.
        """
        # Retrieve state
        # if not hasattr(mod, '_deep_ep_state'):
        #     raise RuntimeError("DeepEP state not found. Dispatch must be called before combine.")

        state = mod._deep_ep_state

        if state.handle is None:
            raise RuntimeError(
                "DeepEP handle not found. Make sure dispatch was called before combine."
            )

        # Undo the padding if it was applied during dispatch
        if self.enable_token_padding:
            routed_output = _vectorized_unpadding(
                routed_output,
                self.original_counts,
                self.padded_counts,
            )

        # Unsort: reverse the sort order from dispatch
        if state.sorted_order is not None:
            # Create inverse permutation
            inv_sorted_order = torch.argsort(state.sorted_order)
            routed_output = routed_output[inv_sorted_order]

        # Apply weights and aggregate outputs back to unique tokens
        # During dispatch, we expanded: each unique token was copied for each local expert
        # Now we need to weight and sum back to unique tokens
        if state.local_token_indices is not None and state.num_unique_tokens is not None:
            # Apply weights to outputs
            if state.expanded_weights is not None:
                # Unsort the weights too (they were sorted with sorted_order)
                if state.sorted_order is not None:
                    expanded_weights = state.expanded_weights[inv_sorted_order]
                else:
                    expanded_weights = state.expanded_weights
                # Apply weights - keep in bfloat16 to save memory
                # (float32 conversion causes OOM with activation checkpointing)
                routed_output = routed_output * expanded_weights.unsqueeze(-1)

            # Aggregate: sum outputs for each unique token
            aggregated_output = torch.zeros(
                state.num_unique_tokens, routed_output.shape[1],
                dtype=routed_output.dtype, device=routed_output.device
            )
            aggregated_output.scatter_add_(
                0,
                state.local_token_indices.unsqueeze(-1).expand_as(routed_output),
                routed_output
            )
            # Ensure bfloat16 and contiguous for DeepEP combine (it only supports bf16/fp16 and requires contiguous tensors)
            routed_output = aggregated_output.to(torch.bfloat16).contiguous()
        else:
            # If no aggregation needed, still ensure correct dtype and layout for DeepEP
            routed_output = routed_output.to(torch.bfloat16).contiguous()

        # Combine tokens using DeepEP with autograd support
        # Note: weights were already applied above, so pass None for topk_weights
        # We need topk_idx and topk_weights for the backward pass (to dispatch gradients)
        # Ensure topk_weights is a proper tensor for save_for_backward
        if state.topk_weights is not None:
            topk_weights_for_backward = state.topk_weights
        else:
            topk_weights_for_backward = torch.ones(
                state.topk_idx.shape,
                dtype=routed_output.dtype,
                device=routed_output.device
            )

        combined_x = _DeepEPCombine.apply(
            routed_output,
            state.handle,
            self.buffer,
            state.topk_idx,
            topk_weights_for_backward,
            self.num_experts,
        )

        # Clean up the state object
        mod._deep_ep_state = None

        # Clean up dispatch cache to prevent memory leaks
        # The cache is only needed within a single forward/backward pass
        if hasattr(self.buffer, '_dispatch_cache'):
            self.buffer._dispatch_cache.clear()

        return combined_x

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        """Shard expert weights on the expert dimension (dim 0)."""
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            mod.register_parameter(name, dist_param)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self._init_buffer(device_mesh)
        module._deep_ep_buffer_ref = self.buffer

        return distribute_module(
            module,
            device_mesh,
            partition_fn=DeepEPExpertParallel._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
