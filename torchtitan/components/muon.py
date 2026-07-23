# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard


__all__ = ["FSDPMuon"]


@dataclass(eq=False, slots=True)
class _FSDPMuonTask:
    param: DTensor
    name: str
    group_index: int
    host_rank: int
    full_shape: torch.Size
    local_shape: torch.Size

    @property
    def local_numel(self) -> int:
        return self.local_shape.numel()


@dataclass(slots=True)
class _ScratchMuon:
    param: torch.nn.Parameter
    optimizer: torch.optim.Muon


class FSDPMuon(torch.optim.Muon):
    """Run full-matrix Muon from uniformly row-sharded FSDP2 gradients.

    Momentum remains sharded with the FSDP2 parameter. For each group of at
    most ``fsdp_world_size`` matrices, one rank hosts each full Newton-Schulz
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
        self._tasks: list[_FSDPMuonTask] = []
        self._scratch_muons: dict[
            tuple[tuple[int, ...], torch.dtype, torch.device], _ScratchMuon
        ] = {}
        self._build_fsdp_plan()

    def _build_fsdp_plan(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("FSDPMuon requires an initialized process group")

        process_group = None
        process_group_ranks = None
        dtype = None
        device = None

        for group_index, group in enumerate(self.param_groups):
            if group.get("fused", False):
                raise ValueError("FSDPMuon does not support fused=True")
            if group.get("foreach", False):
                raise ValueError("FSDPMuon does not support foreach=True")

            params = group["params"]
            param_names = group.get("param_names")
            if param_names is None or len(param_names) != len(params):
                raise ValueError(
                    "FSDPMuon parameter groups require param_names aligned with params"
                )

            for name, param in zip(param_names, params, strict=True):
                if not isinstance(param, DTensor):
                    raise ValueError(f"FSDPMuon parameter {name} must be a DTensor")
                if param.ndim != 2:
                    raise ValueError(
                        f"FSDPMuon parameter {name} must be 2D, got {param.ndim}D"
                    )
                if param.device_mesh.ndim != 1:
                    raise ValueError(
                        f"FSDPMuon parameter {name} must use a 1D FSDP mesh"
                    )
                if len(param.placements) != 1 or type(param.placements[0]) is not Shard:
                    raise ValueError(
                        f"FSDPMuon parameter {name} must have exactly one Shard placement"
                    )
                placement = param.placements[0]
                if placement.dim != 0:
                    raise ValueError(
                        f"FSDPMuon parameter {name} must use Shard(0), got {placement}"
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
                        "All FSDPMuon parameters must use the same FSDP process group"
                    )

                local_param = param.to_local()
                local_shape = torch.Size(local_param.shape)
                world_size = dist.get_world_size(param_process_group)
                if (
                    param.shape[0] != local_shape[0] * world_size
                    or param.shape[1] != local_shape[1]
                ):
                    raise ValueError(
                        f"FSDPMuon parameter {name} must be uniformly row-sharded: "
                        f"global shape={tuple(param.shape)}, "
                        f"local shape={tuple(local_shape)}, world_size={world_size}"
                    )

                if dtype is None:
                    dtype = local_param.dtype
                    device = local_param.device
                elif local_param.dtype != dtype or local_param.device != device:
                    raise ValueError(
                        "All FSDPMuon parameters must have the same dtype and device"
                    )

                host_rank = len(self._tasks) % world_size
                self._tasks.append(
                    _FSDPMuonTask(
                        param=param,
                        name=name,
                        group_index=group_index,
                        host_rank=host_rank,
                        full_shape=torch.Size(param.shape),
                        local_shape=local_shape,
                    )
                )

        if not self._tasks:
            raise ValueError("FSDPMuon requires at least one parameter")
        assert process_group is not None
        assert dtype is not None
        assert device is not None
        self._process_group = process_group
        self._process_group_ranks = process_group_ranks
        self._group_rank = dist.get_rank(process_group)
        self._world_size = dist.get_world_size(process_group)
        self._dtype = dtype
        self._device = device
        self._micro_groups = [
            self._tasks[start : start + self._world_size]
            for start in range(0, len(self._tasks), self._world_size)
        ]
        self._validate_plan_across_ranks()

    def _validate_plan_across_ranks(self) -> None:
        plan = [
            (
                task.name,
                tuple(task.full_shape),
                tuple(task.local_shape),
                task.group_index,
                task.host_rank,
            )
            for task in self._tasks
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
            raise RuntimeError("FSDPMuon parameter plans differ across FSDP ranks")

    def _validate_gradients(self) -> None:
        local_errors = []
        for task in self._tasks:
            grad = task.param.grad
            if grad is None:
                local_errors.append(f"missing gradient for {task.name}")
                continue
            if not isinstance(grad, DTensor):
                local_errors.append(f"gradient for {task.name} is not a DTensor")
                continue
            if (
                torch.Size(grad.shape) != task.full_shape
                or grad.device_mesh is not task.param.device_mesh
                or grad.placements != task.param.placements
                or torch.Size(grad.to_local().shape) != task.local_shape
                or grad.to_local().dtype != self._dtype
            ):
                local_errors.append(f"gradient layout for {task.name} changed")

        error_flag = torch.tensor(
            int(bool(local_errors)), dtype=torch.int32, device=self._device
        )
        dist.all_reduce(error_flag, op=dist.ReduceOp.MAX, group=self._process_group)
        if error_flag.item():
            detail = (
                local_errors[0] if local_errors else "error reported by another rank"
            )
            raise RuntimeError(f"Invalid FSDPMuon gradients: {detail}")

    def _update_local_momentum(self, task: _FSDPMuonTask) -> torch.Tensor:
        group = self.param_groups[task.group_index]
        grad = task.param.grad
        assert isinstance(grad, DTensor)
        state = self.state[task.param]
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

    def _gather_to_hosts(
        self,
        micro_group: list[_FSDPMuonTask],
        local_inputs: list[torch.Tensor],
    ) -> torch.Tensor | None:
        empty = torch.empty(0, dtype=self._dtype, device=self._device)
        send_chunks = local_inputs + [empty] * (self._world_size - len(local_inputs))
        input_split_sizes = [chunk.numel() for chunk in send_chunks]
        send_buffer = torch.cat(send_chunks)

        if self._group_rank < len(micro_group):
            local_numel = micro_group[self._group_rank].local_numel
        else:
            local_numel = 0
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
        if self._group_rank >= len(micro_group):
            return None
        return recv_buffer.view(micro_group[self._group_rank].full_shape)

    def _get_scratch_muon(
        self, task: _FSDPMuonTask, full_input: torch.Tensor
    ) -> _ScratchMuon:
        key = (tuple(task.full_shape), full_input.dtype, full_input.device)
        if key not in self._scratch_muons:
            group = self.param_groups[task.group_index]
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
        self, task: _FSDPMuonTask, full_input: torch.Tensor
    ) -> torch.Tensor:
        scratch = self._get_scratch_muon(task, full_input)
        source_group = self.param_groups[task.group_index]
        scratch_group = scratch.optimizer.param_groups[0]
        for key in ("lr", "ns_coefficients", "eps", "ns_steps", "adjust_lr_fn"):
            scratch_group[key] = source_group[key]

        scratch.param.zero_()
        scratch.param.grad = full_input
        scratch.optimizer.step()
        scratch.param.grad = None
        return scratch.param.detach()

    def _scatter_from_hosts(
        self,
        micro_group: list[_FSDPMuonTask],
        full_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        if full_delta is None:
            send_buffer = torch.empty(0, dtype=self._dtype, device=self._device)
            input_split_sizes = [0] * self._world_size
        else:
            task = micro_group[self._group_rank]
            shards = [
                shard.contiguous().view(-1)
                for shard in torch.chunk(full_delta, self._world_size, dim=0)
            ]
            send_buffer = torch.cat(shards)
            input_split_sizes = [task.local_numel] * self._world_size

        output_split_sizes = [task.local_numel for task in micro_group]
        output_split_sizes.extend([0] * (self._world_size - len(micro_group)))
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
        micro_group: list[_FSDPMuonTask],
        local_deltas: torch.Tensor,
    ) -> None:
        offset = 0
        for task in micro_group:
            next_offset = offset + task.local_numel
            local_delta = local_deltas[offset:next_offset].view(task.local_shape)
            offset = next_offset
            delta = DTensor.from_local(
                local_delta,
                device_mesh=task.param.device_mesh,
                placements=task.param.placements,
                run_check=False,
                shape=task.param.shape,
                stride=task.param.stride(),
            )
            group = self.param_groups[task.group_index]
            task.param.mul_(1 - group["lr"] * group["weight_decay"])
            task.param.add_(delta)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one distributed full-matrix Muon step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._validate_gradients()
        for micro_group in self._micro_groups:
            local_inputs = [
                self._update_local_momentum(task).contiguous().view(-1)
                for task in micro_group
            ]
            full_input = self._gather_to_hosts(micro_group, local_inputs)
            full_delta = (
                self._compute_full_delta(
                    micro_group[self._group_rank],
                    full_input,
                )
                if full_input is not None
                else None
            )
            local_deltas = self._scatter_from_hosts(micro_group, full_delta)
            self._apply_local_updates(micro_group, local_deltas)
        return loss
