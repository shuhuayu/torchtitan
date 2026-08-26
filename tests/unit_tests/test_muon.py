# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.components.muon import (
    AllToAllMuon,
    assign_muon_matrix_owners,
    MuonMatrixSpec,
)


def _run_distributed_muon_parity(
    rank: int,
    world_size: int,
    store_path: str,
    all_to_all_strategy: str,
    num_layers_per_bucket: int,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        torch.manual_seed(42)
        initial_values = [
            torch.randn(8, 5),
            torch.randn(4, 3),
            torch.randn(4, 3),
            torch.randn(6, 2),
        ]
        param_names = [
            "layers.0.large.weight",
            "layers.1.same_a.weight",
            "layers.2.same_b.weight",
            "layers.3.small.weight",
        ]

        def to_dtensor(tensor: torch.Tensor) -> DTensor:
            local_tensor = tensor.chunk(world_size, dim=0)[rank].contiguous()
            return DTensor.from_local(
                local_tensor,
                device_mesh=mesh,
                placements=(Shard(0),),
                run_check=False,
                shape=tensor.shape,
                stride=tensor.stride(),
            )

        sharded_params = [
            torch.nn.Parameter(to_dtensor(value.clone())) for value in initial_values
        ]
        reference_params = [
            torch.nn.Parameter(value.clone()) for value in initial_values
        ]
        optimizer_kwargs = {
            "lr": 2e-2,
            "weight_decay": 0.1,
            "momentum": 0.9,
            "nesterov": True,
            "ns_steps": 3,
            "adjust_lr_fn": "match_rms_adamw",
        }
        sharded_optimizer = AllToAllMuon(
            [{"params": sharded_params, "param_names": param_names}],
            all_to_all_strategy=all_to_all_strategy,
            num_layers_per_bucket=num_layers_per_bucket,
            **optimizer_kwargs,
        )
        reference_optimizer = torch.optim.Muon(
            reference_params,
            **optimizer_kwargs,
        )

        for _ in range(3):
            grads = [torch.randn_like(value) for value in initial_values]
            for sharded_param, reference_param, grad in zip(
                sharded_params, reference_params, grads, strict=True
            ):
                sharded_param.grad = to_dtensor(grad.clone())
                reference_param.grad = grad.clone()

            sharded_optimizer.step()
            reference_optimizer.step()

            for sharded_param, reference_param in zip(
                sharded_params, reference_params, strict=True
            ):
                assert isinstance(sharded_param, DTensor)
                reference_shard = reference_param.chunk(world_size, dim=0)[rank]
                torch.testing.assert_close(
                    sharded_param.to_local(),
                    reference_shard,
                    rtol=1e-6,
                    atol=1e-7,
                )
                sharded_momentum = sharded_optimizer.state[sharded_param][
                    "momentum_buffer"
                ]
                reference_momentum = reference_optimizer.state[reference_param][
                    "momentum_buffer"
                ].chunk(world_size, dim=0)[rank]
                torch.testing.assert_close(
                    sharded_momentum.to_local(),
                    reference_momentum,
                    rtol=0,
                    atol=0,
                )
    finally:
        dist.destroy_process_group()


class TestMuonOwnerPlan(unittest.TestCase):
    def test_assigns_matrix_owners_round_robin(self) -> None:
        canonical_matrices = [
            MuonMatrixSpec(
                fqn=f"layers.{index}.weight",
                shape=torch.Size((8, 4)),
            )
            for index in range(5)
        ]
        matrices = [canonical_matrices[index] for index in (2, 0, 4, 1, 3)]

        assignments = assign_muon_matrix_owners(matrices, num_owner_ranks=2)

        self.assertEqual(
            [assignment.matrix for assignment in assignments], canonical_matrices
        )
        self.assertEqual(
            [assignment.owner_rank for assignment in assignments],
            [0, 1, 0, 1, 0],
        )

    def test_balances_matrices_by_element_count(self) -> None:
        matrices = [
            MuonMatrixSpec(fqn="a.weight", shape=torch.Size((16, 4))),
            MuonMatrixSpec(fqn="b.weight", shape=torch.Size((12, 4))),
            MuonMatrixSpec(fqn="c.weight", shape=torch.Size((8, 4))),
            MuonMatrixSpec(fqn="d.weight", shape=torch.Size((4, 4))),
        ]

        assignments = assign_muon_matrix_owners(matrices, num_owner_ranks=2)

        self.assertEqual(
            [assignment.owner_rank for assignment in assignments],
            [0, 1, 1, 0],
        )


class TestAllToAllMuon(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._store_dir = tempfile.TemporaryDirectory()
        store_path = os.path.join(cls._store_dir.name, "store")
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{store_path}",
            rank=0,
            world_size=1,
        )
        cls._mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()
        cls._store_dir.cleanup()

    def _to_dtensor(self, tensor: torch.Tensor) -> DTensor:
        return DTensor.from_local(
            tensor,
            device_mesh=self._mesh,
            placements=(Shard(0),),
            run_check=False,
            shape=tensor.shape,
            stride=tensor.stride(),
        )

    def _assert_step_matches_upstream_muon(
        self,
        all_to_all_strategy: str,
        num_layers_per_bucket: int = 1,
    ) -> None:
        torch.manual_seed(42)
        initial_values = [torch.randn(4, 3), torch.randn(6, 2)]
        sharded_params = [
            torch.nn.Parameter(self._to_dtensor(value.clone()))
            for value in initial_values
        ]
        reference_params = [
            torch.nn.Parameter(value.clone()) for value in initial_values
        ]
        optimizer_kwargs = {
            "lr": 2e-2,
            "weight_decay": 0.1,
            "momentum": 0.9,
            "nesterov": True,
            "ns_steps": 3,
            "adjust_lr_fn": "match_rms_adamw",
        }
        sharded_optimizer = AllToAllMuon(
            [
                {
                    "params": sharded_params,
                    "param_names": [
                        "layers.0.first.weight",
                        "layers.1.second.weight",
                    ],
                }
            ],
            all_to_all_strategy=all_to_all_strategy,
            num_layers_per_bucket=num_layers_per_bucket,
            **optimizer_kwargs,
        )
        reference_optimizer = torch.optim.Muon(
            reference_params,
            **optimizer_kwargs,
        )

        for _ in range(3):
            grads = [torch.randn_like(value) for value in initial_values]
            for sharded_param, reference_param, grad in zip(
                sharded_params, reference_params, grads, strict=True
            ):
                sharded_param.grad = self._to_dtensor(grad.clone())
                reference_param.grad = grad.clone()

            sharded_optimizer.step()
            reference_optimizer.step()

            for sharded_param, reference_param in zip(
                sharded_params, reference_params, strict=True
            ):
                assert isinstance(sharded_param, DTensor)
                torch.testing.assert_close(
                    sharded_param.to_local(), reference_param, rtol=0, atol=0
                )
                sharded_momentum = sharded_optimizer.state[sharded_param][
                    "momentum_buffer"
                ]
                reference_momentum = reference_optimizer.state[reference_param][
                    "momentum_buffer"
                ]
                self.assertIsInstance(sharded_momentum, DTensor)
                torch.testing.assert_close(
                    sharded_momentum.to_local(),
                    reference_momentum,
                    rtol=0,
                    atol=0,
                )

    def test_step_matches_upstream_muon(self) -> None:
        for all_to_all_strategy, num_layers_per_bucket in (
            ("flat", 1),
            ("layer_pipelined", 1),
            ("layer_pipelined", 2),
            ("shape_grouped", 1),
        ):
            with self.subTest(
                all_to_all_strategy=all_to_all_strategy,
                num_layers_per_bucket=num_layers_per_bucket,
            ):
                self._assert_step_matches_upstream_muon(
                    all_to_all_strategy,
                    num_layers_per_bucket,
                )

    def test_requires_dtensor_parameters(self) -> None:
        param = torch.nn.Parameter(torch.randn(4, 3))
        with self.assertRaisesRegex(ValueError, "must be a DTensor"):
            AllToAllMuon(
                [{"params": [param], "param_names": ["weight"]}],
            )

    def test_rejects_unknown_all_to_all_strategy(self) -> None:
        param = torch.nn.Parameter(torch.randn(4, 3))
        with self.assertRaisesRegex(ValueError, "all_to_all_strategy"):
            AllToAllMuon(
                [{"params": [param], "param_names": ["weight"]}],
                all_to_all_strategy="unknown",
            )

    def test_layer_pipelined_requires_layer_fqns(self) -> None:
        param = torch.nn.Parameter(self._to_dtensor(torch.randn(4, 3)))
        with self.assertRaisesRegex(ValueError, r"layers\.<index>"):
            AllToAllMuon(
                [{"params": [param], "param_names": ["weight"]}],
                all_to_all_strategy="layer_pipelined",
            )

    def test_layer_pipelined_uses_numeric_layer_order(self) -> None:
        params = [
            torch.nn.Parameter(self._to_dtensor(torch.randn(4, 3))) for _ in range(4)
        ]
        optimizer = AllToAllMuon(
            [
                {
                    "params": params,
                    "param_names": [
                        "layers.10.weight",
                        "layers.2.weight",
                        "layers.11.weight",
                        "layers.3.weight",
                    ],
                }
            ],
            all_to_all_strategy="layer_pipelined",
            num_layers_per_bucket=2,
        )

        self.assertEqual(
            [plan.layer_fqns for plan in optimizer._layer_pipelined_plans],
            [
                ("layers.2", "layers.3"),
                ("layers.10", "layers.11"),
            ],
        )

    def test_rejects_invalid_num_layers_per_bucket(self) -> None:
        param = torch.nn.Parameter(self._to_dtensor(torch.randn(4, 3)))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            AllToAllMuon(
                [{"params": [param], "param_names": ["layers.0.weight"]}],
                all_to_all_strategy="layer_pipelined",
                num_layers_per_bucket=0,
            )
        with self.assertRaisesRegex(ValueError, "only configurable"):
            AllToAllMuon(
                [{"params": [param], "param_names": ["layers.0.weight"]}],
                all_to_all_strategy="flat",
                num_layers_per_bucket=2,
            )

    def test_collective_count_matches_strategy(self) -> None:
        initial_values = [torch.randn(4, 3), torch.randn(4, 3), torch.randn(6, 2)]
        for all_to_all_strategy, num_layers_per_bucket, expected_calls in (
            ("flat", 1, 2),
            ("layer_pipelined", 1, 4),
            ("layer_pipelined", 2, 2),
            ("shape_grouped", 1, 4),
        ):
            with self.subTest(
                all_to_all_strategy=all_to_all_strategy,
                num_layers_per_bucket=num_layers_per_bucket,
            ):
                params = [
                    torch.nn.Parameter(self._to_dtensor(value.clone()))
                    for value in initial_values
                ]
                optimizer = AllToAllMuon(
                    [
                        {
                            "params": params,
                            "param_names": [
                                "layers.0.same_a.weight",
                                "layers.0.same_b.weight",
                                "layers.1.other.weight",
                            ],
                            "all_to_all_strategy": all_to_all_strategy,
                            "num_layers_per_bucket": num_layers_per_bucket,
                        }
                    ],
                    ns_steps=1,
                )
                for param, value in zip(params, initial_values, strict=True):
                    param.grad = self._to_dtensor(torch.ones_like(value))

                with patch.object(
                    dist,
                    "all_to_all_single",
                    wraps=dist.all_to_all_single,
                ) as all_to_all:
                    optimizer.step()

                self.assertEqual(all_to_all.call_count, expected_calls)

    def test_rejects_conflicting_group_strategy(self) -> None:
        param = torch.nn.Parameter(self._to_dtensor(torch.randn(4, 3)))
        with self.assertRaisesRegex(ValueError, "must agree"):
            AllToAllMuon(
                [
                    {
                        "params": [param],
                        "param_names": ["layers.0.weight"],
                        "all_to_all_strategy": "shape_grouped",
                    }
                ],
                all_to_all_strategy="flat",
            )

    def test_layer_pipelined_launches_next_gather_before_current_compute(
        self,
    ) -> None:
        initial_values = [torch.randn(4, 3), torch.randn(6, 2)]
        params = [
            torch.nn.Parameter(self._to_dtensor(value.clone()))
            for value in initial_values
        ]
        optimizer = AllToAllMuon(
            [
                {
                    "params": params,
                    "param_names": [
                        "layers.0.first.weight",
                        "layers.1.second.weight",
                    ],
                }
            ],
            all_to_all_strategy="layer_pipelined",
            ns_steps=1,
        )
        for param, value in zip(params, initial_values, strict=True):
            param.grad = self._to_dtensor(torch.ones_like(value))

        events: list[tuple[str, bool | str]] = []
        original_all_to_all = dist.all_to_all_single
        original_compute = optimizer._compute_flat_owner_deltas

        def record_all_to_all(*args, **kwargs):
            events.append(("all_to_all", kwargs.get("async_op", False)))
            return original_all_to_all(*args, **kwargs)

        def record_compute(plan) -> None:
            layer_fqn = next(
                optimizer._layer_fqn(binding.name)
                for owner_bindings in plan.bindings_by_owner
                for binding in owner_bindings
            )
            events.append(("compute", layer_fqn))
            original_compute(plan)

        with (
            patch.object(dist, "all_to_all_single", side_effect=record_all_to_all),
            patch.object(
                optimizer,
                "_compute_flat_owner_deltas",
                side_effect=record_compute,
            ),
        ):
            optimizer.step()

        self.assertEqual(
            events,
            [
                ("all_to_all", True),
                ("all_to_all", True),
                ("compute", "layers.0"),
                ("all_to_all", True),
                ("compute", "layers.1"),
                ("all_to_all", True),
            ],
        )


class TestDistributedAllToAllMuon(unittest.TestCase):
    def test_strategies_match_upstream_muon(self) -> None:
        for all_to_all_strategy, num_layers_per_bucket in (
            ("flat", 1),
            ("layer_pipelined", 1),
            ("layer_pipelined", 2),
            ("shape_grouped", 1),
        ):
            with self.subTest(
                all_to_all_strategy=all_to_all_strategy,
                num_layers_per_bucket=num_layers_per_bucket,
            ):
                with tempfile.TemporaryDirectory() as store_dir:
                    mp.spawn(
                        _run_distributed_muon_parity,
                        args=(
                            2,
                            os.path.join(store_dir, "store"),
                            all_to_all_strategy,
                            num_layers_per_bucket,
                        ),
                        nprocs=2,
                        join=True,
                    )


if __name__ == "__main__":
    unittest.main()
