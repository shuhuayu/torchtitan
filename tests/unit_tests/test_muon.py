# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.components.muon import FSDPMuon


class TestFSDPMuon(unittest.TestCase):
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

    def test_step_matches_upstream_muon(self) -> None:
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
        sharded_optimizer = FSDPMuon(
            [
                {
                    "params": sharded_params,
                    "param_names": ["first.weight", "second.weight"],
                }
            ],
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

    def test_requires_dtensor_parameters(self) -> None:
        param = torch.nn.Parameter(torch.randn(4, 3))
        with self.assertRaisesRegex(ValueError, "must be a DTensor"):
            FSDPMuon(
                [{"params": [param], "param_names": ["weight"]}],
            )


if __name__ == "__main__":
    unittest.main()
