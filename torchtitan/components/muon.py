# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard


__all__ = [
    "AllToAllMuon",
    "MuonMatrixAssignment",
    "MuonMatrixSpec",
    "assign_muon_matrix_owners",
]


@dataclass(frozen=True, slots=True)
class MuonMatrixSpec:
    """Identify one logical Muon matrix within a parameter.

    ``param_offset`` is the element offset of a contiguous matrix within the
    parameter. This allows a future lowering to assign packed experts or
    per-head matrices independently. The current DTensor lowering uses one
    matrix per parameter with offset zero.
    """

    fqn: str
    shape: torch.Size
    param_offset: int = 0

    def __post_init__(self) -> None:
        if len(self.shape) != 2:
            raise ValueError(
                f"Muon matrix {self.fqn} must be 2D, got shape {tuple(self.shape)}"
            )
        if self.param_offset < 0:
            raise ValueError("Muon matrix param_offset must be non-negative")


@dataclass(frozen=True, slots=True)
class MuonMatrixAssignment:
    """Assign one logical Muon matrix to a rank within its storage mesh."""

    matrix: MuonMatrixSpec
    owner_rank: int

    def __post_init__(self) -> None:
        if self.owner_rank < 0:
            raise ValueError("Muon matrix owner_rank must be non-negative")


def assign_muon_matrix_owners(
    matrices: Sequence[MuonMatrixSpec], *, num_owner_ranks: int
) -> tuple[MuonMatrixAssignment, ...]:
    """Assign logical matrices to mesh-local owner ranks in canonical order."""
    if num_owner_ranks <= 0:
        raise ValueError(f"num_owner_ranks must be positive, got {num_owner_ranks}")
    ordered_matrices = sorted(
        matrices,
        key=lambda matrix: (
            matrix.fqn,
            matrix.param_offset,
            tuple(matrix.shape),
        ),
    )
    if len(set(ordered_matrices)) != len(ordered_matrices):
        raise ValueError("Muon matrix specifications must be unique")
    return tuple(
        MuonMatrixAssignment(matrix=matrix, owner_rank=index % num_owner_ranks)
        for index, matrix in enumerate(ordered_matrices)
    )


@dataclass(eq=False, slots=True)
class _MuonComputeStorageBinding:
    """Bind a logical Muon compute assignment to its DTensor storage."""

    param: DTensor
    assignment: MuonMatrixAssignment
    optimizer_group_index: int
    local_shape: torch.Size

    @property
    def name(self) -> str:
        return self.assignment.matrix.fqn

    @property
    def full_shape(self) -> torch.Size:
        return self.assignment.matrix.shape

    @property
    def local_numel(self) -> int:
        return self.local_shape.numel()


@dataclass(slots=True)
class _ScratchMuon:
    param: torch.nn.Parameter
    optimizer: torch.optim.Muon


class AllToAllMuon(torch.optim.Muon):
    """Run full-matrix Muon from uniformly row-sharded FSDP2 gradients.

    Momentum remains sharded with the FSDP2 parameter. For each group of at
    most ``fsdp_world_size`` matrices, one rank owns each full Newton-Schulz
    computation. Two all-to-all collectives gather the post-momentum inputs and
    scatter the resulting update shards.

    This initial implementation supports only a one-dimensional FSDP mesh with
    uniform ``Shard(0)`` DTensors.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        self._bindings: list[_MuonComputeStorageBinding] = []
        self._scratch_muons: dict[
            tuple[tuple[int, ...], torch.dtype, torch.device], _ScratchMuon
        ] = {}
        self._build_dtensor_plan()

    def _build_dtensor_plan(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("AllToAllMuon requires an initialized process group")

        process_group = None
        process_group_ranks = None
        dtype = None
        device = None
        binding_inputs: list[tuple[DTensor, MuonMatrixSpec, int, torch.Size]] = []

        for group_index, group in enumerate(self.param_groups):
            if group.get("fused", False):
                raise ValueError("AllToAllMuon does not support fused=True")
            if group.get("foreach", False):
                raise ValueError("AllToAllMuon does not support foreach=True")

            params = group["params"]
            param_names = group.get("param_names")
            if param_names is None or len(param_names) != len(params):
                raise ValueError(
                    "AllToAllMuon parameter groups require param_names aligned with "
                    "params"
                )

            for name, param in zip(param_names, params, strict=True):
                if not isinstance(param, DTensor):
                    raise ValueError(f"AllToAllMuon parameter {name} must be a DTensor")
                if param.ndim != 2:
                    raise ValueError(
                        f"AllToAllMuon parameter {name} must be 2D, got {param.ndim}D"
                    )
                if param.device_mesh.ndim != 1:
                    raise ValueError(
                        f"AllToAllMuon parameter {name} must use a 1D FSDP mesh"
                    )
                if len(param.placements) != 1 or type(param.placements[0]) is not Shard:
                    raise ValueError(
                        f"AllToAllMuon parameter {name} must have exactly one "
                        "Shard placement"
                    )
                placement = param.placements[0]
                if placement.dim != 0:
                    raise ValueError(
                        f"AllToAllMuon parameter {name} must use Shard(0), "
                        f"got {placement}"
                    )

                param_process_group = param.device_mesh.get_group()
                param_process_group_ranks = tuple(
                    dist.get_process_group_ranks(param_process_group)
                )
                if process_group is None:
                    process_group = param_process_group
                    process_group_ranks = param_process_group_ranks
                elif param_process_group_ranks != process_group_ranks:
                    raise ValueError(
                        "All parameters handled by AllToAllMuon must use the same "
                        "FSDP process group"
                    )

                local_param = param.to_local()
                local_shape = torch.Size(local_param.shape)
                world_size = dist.get_world_size(param_process_group)
                if (
                    param.shape[0] != local_shape[0] * world_size
                    or param.shape[1] != local_shape[1]
                ):
                    raise ValueError(
                        f"AllToAllMuon parameter {name} must be uniformly row-sharded: "
                        f"global shape={tuple(param.shape)}, "
                        f"local shape={tuple(local_shape)}, world_size={world_size}"
                    )

                if dtype is None:
                    dtype = local_param.dtype
                    device = local_param.device
                elif local_param.dtype != dtype or local_param.device != device:
                    raise ValueError(
                        "All parameters handled by AllToAllMuon must have the same "
                        "dtype and device"
                    )

                binding_inputs.append(
                    (
                        param,
                        MuonMatrixSpec(fqn=name, shape=torch.Size(param.shape)),
                        group_index,
                        local_shape,
                    )
                )

        if not binding_inputs:
            raise ValueError("AllToAllMuon requires at least one parameter")
        assert process_group is not None
        assert dtype is not None
        assert device is not None
        self._process_group = process_group
        self._group_rank = dist.get_rank(process_group)
        self._world_size = dist.get_world_size(process_group)
        self._dtype = dtype
        self._device = device
        assignments = assign_muon_matrix_owners(
            [binding_input[1] for binding_input in binding_inputs],
            num_owner_ranks=self._world_size,
        )
        binding_input_by_matrix = {
            matrix: (param, optimizer_group_index, local_shape)
            for param, matrix, optimizer_group_index, local_shape in binding_inputs
        }
        assert len(binding_input_by_matrix) == len(binding_inputs)
        for assignment in assignments:
            param, optimizer_group_index, local_shape = binding_input_by_matrix[
                assignment.matrix
            ]
            self._bindings.append(
                _MuonComputeStorageBinding(
                    param=param,
                    assignment=assignment,
                    optimizer_group_index=optimizer_group_index,
                    local_shape=local_shape,
                )
            )
        self._execution_groups = self._build_execution_groups()
        self._validate_plan_across_ranks()

    def _build_execution_groups(
        self,
    ) -> list[tuple[_MuonComputeStorageBinding | None, ...]]:
        execution_groups = []
        for start in range(0, len(self._bindings), self._world_size):
            bindings_by_owner: list[_MuonComputeStorageBinding | None] = [
                None
            ] * self._world_size
            for binding in self._bindings[start : start + self._world_size]:
                owner_rank = binding.assignment.owner_rank
                assert bindings_by_owner[owner_rank] is None
                bindings_by_owner[owner_rank] = binding
            execution_groups.append(tuple(bindings_by_owner))
        return execution_groups

    def _validate_plan_across_ranks(self) -> None:
        plan = [
            (
                binding.name,
                binding.assignment.matrix.param_offset,
                tuple(binding.full_shape),
                tuple(binding.local_shape),
                binding.optimizer_group_index,
                binding.assignment.owner_rank,
            )
            for binding in self._bindings
        ]
        digest = hashlib.sha256(repr(plan).encode("utf-8")).digest()
        plan_hash = int.from_bytes(digest[:7], byteorder="little")
        local_hash = torch.tensor(plan_hash, dtype=torch.int64, device=self._device)
        gathered_hashes = [
            torch.empty_like(local_hash) for _ in range(self._world_size)
        ]
        dist.all_gather(
            gathered_hashes,
            local_hash,
            group=self._process_group,
        )
        if any(value.item() != plan_hash for value in gathered_hashes):
            raise RuntimeError("AllToAllMuon parameter plans differ across FSDP ranks")

    def _validate_gradients(self) -> None:
        local_errors = []
        for binding in self._bindings:
            grad = binding.param.grad
            if grad is None:
                local_errors.append(f"missing gradient for {binding.name}")
                continue
            if not isinstance(grad, DTensor):
                local_errors.append(f"gradient for {binding.name} is not a DTensor")
                continue
            if (
                torch.Size(grad.shape) != binding.full_shape
                or grad.device_mesh is not binding.param.device_mesh
                or grad.placements != binding.param.placements
                or torch.Size(grad.to_local().shape) != binding.local_shape
                or grad.to_local().dtype != self._dtype
            ):
                local_errors.append(f"gradient layout for {binding.name} changed")

        error_flag = torch.tensor(
            int(bool(local_errors)), dtype=torch.int32, device=self._device
        )
        dist.all_reduce(error_flag, op=dist.ReduceOp.MAX, group=self._process_group)
        if error_flag.item():
            detail = (
                local_errors[0] if local_errors else "error reported by another rank"
            )
            raise RuntimeError(f"Invalid AllToAllMuon gradients: {detail}")

    def _update_local_momentum(
        self, binding: _MuonComputeStorageBinding
    ) -> torch.Tensor:
        group = self.param_groups[binding.optimizer_group_index]
        grad = binding.param.grad
        assert isinstance(grad, DTensor)
        state = self.state[binding.param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        momentum_buffer = state["momentum_buffer"]
        assert isinstance(momentum_buffer, DTensor)

        local_grad = grad.to_local()
        local_buffer = momentum_buffer.to_local()
        momentum = group["momentum"]
        local_buffer.lerp_(local_grad, 1 - momentum)
        if group["nesterov"]:
            return local_grad.lerp(local_buffer, momentum)
        return local_buffer

    def _gather_to_owners(
        self,
        execution_group: tuple[_MuonComputeStorageBinding | None, ...],
        local_inputs: list[torch.Tensor],
    ) -> torch.Tensor | None:
        assert len(local_inputs) == self._world_size
        input_split_sizes = [local_input.numel() for local_input in local_inputs]
        send_buffer = torch.cat(local_inputs)

        owner_binding = execution_group[self._group_rank]
        local_numel = owner_binding.local_numel if owner_binding is not None else 0
        output_split_sizes = [local_numel] * self._world_size
        recv_buffer = torch.empty(
            sum(output_split_sizes), dtype=self._dtype, device=self._device
        )
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self._process_group,
        )
        if owner_binding is None:
            return None
        return recv_buffer.view(owner_binding.full_shape)

    def _get_scratch_muon(
        self, binding: _MuonComputeStorageBinding, full_input: torch.Tensor
    ) -> _ScratchMuon:
        key = (tuple(binding.full_shape), full_input.dtype, full_input.device)
        if key not in self._scratch_muons:
            group = self.param_groups[binding.optimizer_group_index]
            scratch_param = torch.nn.Parameter(torch.zeros_like(full_input))
            scratch_optimizer = torch.optim.Muon(
                [scratch_param],
                lr=group["lr"],
                weight_decay=0.0,
                momentum=0.0,
                nesterov=False,
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
                ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"],
            )
            self._scratch_muons[key] = _ScratchMuon(
                param=scratch_param,
                optimizer=scratch_optimizer,
            )
        return self._scratch_muons[key]

    def _compute_full_delta(
        self, binding: _MuonComputeStorageBinding, full_input: torch.Tensor
    ) -> torch.Tensor:
        scratch = self._get_scratch_muon(binding, full_input)
        source_group = self.param_groups[binding.optimizer_group_index]
        scratch_group = scratch.optimizer.param_groups[0]
        for key in ("lr", "ns_coefficients", "eps", "ns_steps", "adjust_lr_fn"):
            scratch_group[key] = source_group[key]

        scratch.param.zero_()
        scratch.param.grad = full_input
        scratch.optimizer.step()
        scratch.param.grad = None
        return scratch.param.detach()

    def _scatter_from_owners(
        self,
        execution_group: tuple[_MuonComputeStorageBinding | None, ...],
        full_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        owner_binding = execution_group[self._group_rank]
        if owner_binding is None:
            assert full_delta is None
            send_buffer = torch.empty(0, dtype=self._dtype, device=self._device)
            input_split_sizes = [0] * self._world_size
        else:
            assert full_delta is not None
            shards = [
                shard.contiguous().view(-1)
                for shard in torch.chunk(full_delta, self._world_size, dim=0)
            ]
            send_buffer = torch.cat(shards)
            input_split_sizes = [owner_binding.local_numel] * self._world_size

        output_split_sizes = [
            binding.local_numel if binding is not None else 0
            for binding in execution_group
        ]
        recv_buffer = torch.empty(
            sum(output_split_sizes), dtype=self._dtype, device=self._device
        )
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self._process_group,
        )
        return recv_buffer

    def _apply_local_updates(
        self,
        execution_group: tuple[_MuonComputeStorageBinding | None, ...],
        local_deltas: torch.Tensor,
    ) -> None:
        offset = 0
        for binding in execution_group:
            if binding is None:
                continue
            next_offset = offset + binding.local_numel
            local_delta = local_deltas[offset:next_offset].view(binding.local_shape)
            offset = next_offset
            delta = DTensor.from_local(
                local_delta,
                device_mesh=binding.param.device_mesh,
                placements=binding.param.placements,
                run_check=False,
                shape=binding.param.shape,
                stride=binding.param.stride(),
            )
            group = self.param_groups[binding.optimizer_group_index]
            binding.param.mul_(1 - group["lr"] * group["weight_decay"])
            binding.param.add_(delta)
        assert offset == local_deltas.numel()

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one distributed full-matrix Muon step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._validate_gradients()
        empty = torch.empty(0, dtype=self._dtype, device=self._device)
        for execution_group in self._execution_groups:
            local_inputs = [
                empty
                if binding is None
                else self._update_local_momentum(binding).contiguous().view(-1)
                for binding in execution_group
            ]
            full_input = self._gather_to_owners(execution_group, local_inputs)
            owner_binding = execution_group[self._group_rank]
            full_delta = (
                self._compute_full_delta(
                    owner_binding,
                    full_input,
                )
                if owner_binding is not None and full_input is not None
                else None
            )
            local_deltas = self._scatter_from_owners(execution_group, full_delta)
            self._apply_local_updates(execution_group, local_deltas)
        return loss
