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
    """Greedily balance logical matrices and return them in canonical order.

    Largest matrices are assigned first to the least-loaded owner. This keeps
    the policy independent of any storage backend while making the transient
    full-matrix compute footprint approximately even across ranks.
    """
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
    owner_loads = [0] * num_owner_ranks
    owner_by_matrix: dict[MuonMatrixSpec, int] = {}
    for matrix in sorted(
        ordered_matrices,
        key=lambda candidate: (
            -candidate.shape.numel(),
            candidate.fqn,
            candidate.param_offset,
            tuple(candidate.shape),
        ),
    ):
        owner_rank = min(
            range(num_owner_ranks),
            key=lambda rank: (owner_loads[rank], rank),
        )
        owner_by_matrix[matrix] = owner_rank
        owner_loads[owner_rank] += matrix.shape.numel()
    return tuple(
        MuonMatrixAssignment(matrix=matrix, owner_rank=owner_by_matrix[matrix])
        for matrix in ordered_matrices
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
class _FlatAllToAllPlan:
    bindings_by_owner: tuple[tuple[_MuonComputeStorageBinding, ...], ...]
    send_offsets: dict[_MuonComputeStorageBinding, int]
    owner_offsets: dict[_MuonComputeStorageBinding, int]
    input_split_sizes: list[int]
    owned_local_numel: int
    local_buffer: torch.Tensor
    owner_buffer: torch.Tensor


@dataclass(slots=True)
class _ShapeGroupedAllToAllPlan:
    bindings_by_owner: tuple[tuple[_MuonComputeStorageBinding, ...], ...]
    slots: dict[_MuonComputeStorageBinding, int]
    num_slots_per_owner: int
    local_numel: int
    local_buffer: torch.Tensor
    owner_buffer: torch.Tensor


@dataclass(slots=True)
class _ScratchMuon:
    param: torch.nn.Parameter
    optimizer: torch.optim.Muon


class AllToAllMuon(torch.optim.Muon):
    """Run full-matrix Muon from uniformly row-sharded FSDP2 gradients.

    Momentum remains sharded with the FSDP2 parameter. One rank temporarily
    owns each full Newton-Schulz computation. The ``flat`` strategy exchanges
    every matrix through one variable-split all-to-all in each direction. The
    ``shape_grouped`` strategy uses one regular, padded exchange in each
    direction per unique matrix shape.

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
        all_to_all_strategy: str = "flat",
    ) -> None:
        if all_to_all_strategy not in {"flat", "shape_grouped"}:
            raise ValueError(
                "all_to_all_strategy must be 'flat' or 'shape_grouped', got "
                f"{all_to_all_strategy!r}"
            )
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
        self._all_to_all_strategy = all_to_all_strategy
        self._bindings: list[_MuonComputeStorageBinding] = []
        self._scratch_muons: dict[
            tuple[tuple[int, ...], torch.dtype, torch.device], _ScratchMuon
        ] = {}
        self._full_input_buffers: dict[
            tuple[tuple[int, ...], torch.dtype, torch.device], torch.Tensor
        ] = {}
        self._flat_plan: _FlatAllToAllPlan | None = None
        self._shape_grouped_plans: list[_ShapeGroupedAllToAllPlan] = []
        self._process_group: dist.ProcessGroup
        self._group_rank: int
        self._world_size: int
        self._dtype: torch.dtype
        self._tensor_device: torch.device
        self._build_dtensor_plan()

    def _build_dtensor_plan(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("AllToAllMuon requires an initialized process group")

        process_group: dist.ProcessGroup | None = None
        process_group_ranks: tuple[int, ...] | None = None
        dtype: torch.dtype | None = None
        device: torch.device | None = None
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
        self._tensor_device = device  # pyrefly: ignore [read-only]
        matrices = [binding_input[1] for binding_input in binding_inputs]
        assignments = self._assign_matrix_owners(matrices)
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
        self._validate_plan_across_ranks()
        if self._all_to_all_strategy == "flat":
            self._flat_plan = self._build_flat_plan()
        else:
            self._shape_grouped_plans = self._build_shape_grouped_plans()

    def _assign_matrix_owners(
        self, matrices: list[MuonMatrixSpec]
    ) -> tuple[MuonMatrixAssignment, ...]:
        if self._all_to_all_strategy == "flat":
            return assign_muon_matrix_owners(
                matrices,
                num_owner_ranks=self._world_size,
            )

        matrices_by_shape: dict[tuple[int, ...], list[MuonMatrixSpec]] = {}
        for matrix in matrices:
            matrices_by_shape.setdefault(tuple(matrix.shape), []).append(matrix)
        return tuple(
            assignment
            for shape in sorted(matrices_by_shape)
            for assignment in assign_muon_matrix_owners(
                matrices_by_shape[shape],
                num_owner_ranks=self._world_size,
            )
        )

    def _group_bindings_by_owner(
        self,
        bindings: list[_MuonComputeStorageBinding],
    ) -> tuple[tuple[_MuonComputeStorageBinding, ...], ...]:
        bindings_by_owner: list[list[_MuonComputeStorageBinding]] = [
            [] for _ in range(self._world_size)
        ]
        for binding in bindings:
            bindings_by_owner[binding.assignment.owner_rank].append(binding)
        return tuple(tuple(owner_bindings) for owner_bindings in bindings_by_owner)

    def _build_flat_plan(self) -> _FlatAllToAllPlan:
        bindings_by_owner = self._group_bindings_by_owner(self._bindings)
        send_offsets = {}
        owner_offsets = {}
        input_split_sizes = []
        send_offset = 0
        for owner_bindings in bindings_by_owner:
            owner_offset = 0
            for binding in owner_bindings:
                send_offsets[binding] = send_offset
                owner_offsets[binding] = owner_offset
                send_offset += binding.local_numel
                owner_offset += binding.local_numel
            input_split_sizes.append(owner_offset)

        owned_local_numel = input_split_sizes[self._group_rank]
        return _FlatAllToAllPlan(
            bindings_by_owner=bindings_by_owner,
            send_offsets=send_offsets,
            owner_offsets=owner_offsets,
            input_split_sizes=input_split_sizes,
            owned_local_numel=owned_local_numel,
            local_buffer=torch.empty(
                send_offset,
                dtype=self._dtype,
                device=self._tensor_device,
            ),
            owner_buffer=torch.empty(
                owned_local_numel * self._world_size,
                dtype=self._dtype,
                device=self._tensor_device,
            ),
        )

    def _build_shape_grouped_plans(self) -> list[_ShapeGroupedAllToAllPlan]:
        bindings_by_shape: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            list[_MuonComputeStorageBinding],
        ] = {}
        for binding in self._bindings:
            key = (tuple(binding.full_shape), tuple(binding.local_shape))
            bindings_by_shape.setdefault(key, []).append(binding)

        plans = []
        for shape_key in sorted(bindings_by_shape):
            bindings_by_owner = self._group_bindings_by_owner(
                bindings_by_shape[shape_key]
            )
            num_slots_per_owner = max(map(len, bindings_by_owner))
            first_binding = next(
                binding
                for owner_bindings in bindings_by_owner
                for binding in owner_bindings
            )
            local_numel = first_binding.local_numel
            slots = {
                binding: slot
                for owner_bindings in bindings_by_owner
                for slot, binding in enumerate(owner_bindings)
            }
            buffer_numel = self._world_size * num_slots_per_owner * local_numel
            plans.append(
                _ShapeGroupedAllToAllPlan(
                    bindings_by_owner=bindings_by_owner,
                    slots=slots,
                    num_slots_per_owner=num_slots_per_owner,
                    local_numel=local_numel,
                    local_buffer=torch.zeros(
                        buffer_numel,
                        dtype=self._dtype,
                        device=self._tensor_device,
                    ),
                    owner_buffer=torch.empty(
                        buffer_numel,
                        dtype=self._dtype,
                        device=self._tensor_device,
                    ),
                )
            )
        return plans

    def _validate_plan_across_ranks(self) -> None:
        plan = (
            self._all_to_all_strategy,
            [
                (
                    binding.name,
                    binding.assignment.matrix.param_offset,
                    tuple(binding.full_shape),
                    tuple(binding.local_shape),
                    binding.optimizer_group_index,
                    binding.assignment.owner_rank,
                )
                for binding in self._bindings
            ],
        )
        digest = hashlib.sha256(repr(plan).encode("utf-8")).digest()
        plan_hash = int.from_bytes(digest[:7], byteorder="little")
        local_hash = torch.tensor(
            plan_hash, dtype=torch.int64, device=self._tensor_device
        )
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
            int(bool(local_errors)), dtype=torch.int32, device=self._tensor_device
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

    def _get_full_input_buffer(
        self, binding: _MuonComputeStorageBinding
    ) -> torch.Tensor:
        key = (tuple(binding.full_shape), self._dtype, self._tensor_device)
        if key not in self._full_input_buffers:
            self._full_input_buffers[key] = torch.empty(
                binding.full_shape,
                dtype=self._dtype,
                device=self._tensor_device,
            )
        return self._full_input_buffers[key]

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

    def _apply_local_update(
        self,
        binding: _MuonComputeStorageBinding,
        local_delta: torch.Tensor,
    ) -> None:
        delta = DTensor.from_local(
            local_delta.view(binding.local_shape),
            device_mesh=binding.param.device_mesh,
            placements=binding.param.placements,
            run_check=False,
            shape=binding.param.shape,
            stride=binding.param.stride(),
        )
        group = self.param_groups[binding.optimizer_group_index]
        binding.param.mul_(1 - group["lr"] * group["weight_decay"])
        binding.param.add_(delta)

    def _step_flat(self) -> None:
        plan = self._flat_plan
        assert plan is not None

        for owner_bindings in plan.bindings_by_owner:
            for binding in owner_bindings:
                send_offset = plan.send_offsets[binding]
                plan.local_buffer[
                    send_offset : send_offset + binding.local_numel
                ].copy_(self._update_local_momentum(binding).contiguous().view(-1))

        owner_split_sizes = [plan.owned_local_numel] * self._world_size
        dist.all_to_all_single(
            plan.owner_buffer,
            plan.local_buffer,
            output_split_sizes=owner_split_sizes,
            input_split_sizes=plan.input_split_sizes,
            group=self._process_group,
        )

        for binding in plan.bindings_by_owner[self._group_rank]:
            owner_offset = plan.owner_offsets[binding]
            owner_shards = plan.owner_buffer.as_strided(
                (self._world_size, binding.local_numel),
                (plan.owned_local_numel, 1),
                owner_offset,
            )
            full_input = self._get_full_input_buffer(binding)
            full_input.view(self._world_size, binding.local_numel).copy_(owner_shards)
            full_delta = self._compute_full_delta(binding, full_input)
            owner_shards.copy_(full_delta.view(self._world_size, binding.local_numel))

        dist.all_to_all_single(
            plan.local_buffer,
            plan.owner_buffer,
            output_split_sizes=plan.input_split_sizes,
            input_split_sizes=owner_split_sizes,
            group=self._process_group,
        )

        for owner_bindings in plan.bindings_by_owner:
            for binding in owner_bindings:
                send_offset = plan.send_offsets[binding]
                self._apply_local_update(
                    binding,
                    plan.local_buffer[send_offset : send_offset + binding.local_numel],
                )

    def _step_shape_grouped(self) -> None:
        for plan in self._shape_grouped_plans:
            local_slots = plan.local_buffer.view(
                self._world_size,
                plan.num_slots_per_owner,
                plan.local_numel,
            )
            for owner_rank, owner_bindings in enumerate(plan.bindings_by_owner):
                for binding in owner_bindings:
                    local_slots[owner_rank, plan.slots[binding]].copy_(
                        self._update_local_momentum(binding).contiguous().view(-1)
                    )

            dist.all_to_all_single(
                plan.owner_buffer,
                plan.local_buffer,
                group=self._process_group,
            )

            owner_slots = plan.owner_buffer.view(
                self._world_size,
                plan.num_slots_per_owner,
                plan.local_numel,
            )
            for binding in plan.bindings_by_owner[self._group_rank]:
                slot = plan.slots[binding]
                full_input = self._get_full_input_buffer(binding)
                full_input.view(self._world_size, plan.local_numel).copy_(
                    owner_slots[:, slot]
                )
                full_delta = self._compute_full_delta(binding, full_input)
                owner_slots[:, slot].copy_(
                    full_delta.view(self._world_size, plan.local_numel)
                )

            dist.all_to_all_single(
                plan.local_buffer,
                plan.owner_buffer,
                group=self._process_group,
            )

            for owner_rank, owner_bindings in enumerate(plan.bindings_by_owner):
                for binding in owner_bindings:
                    self._apply_local_update(
                        binding,
                        local_slots[owner_rank, plan.slots[binding]],
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one distributed full-matrix Muon step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._validate_gradients()
        if self._all_to_all_strategy == "flat":
            self._step_flat()
        else:
            self._step_shape_grouped()
        return loss
