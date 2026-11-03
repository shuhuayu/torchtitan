#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Same-weight component, block, vision and full-logit Flash-Next parity.

Install transformers at 494470d677d05d44153d235d795e0d01040307ab, then run:
python -m scripts.checkpoint_conversion.numerical_tests_qwen3_8_flash_next
The reduced fixture retains every backbone feature. Components and gradients
use FP32; the recurrent backbone uses BF16, as required by its shared kernel.
"""

import argparse
import copy
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.state_dict_adapter import (
    Qwen38FlashNextStateDictAdapter,
)

REFERENCE_REVISION = "494470d677d05d44153d235d795e0d01040307ab"
CHECKPOINT_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"

# T = packed tokens, D = model dimension, C = residual streams * D,
# E = experts, K = selected experts, V = vocabulary.
BF16_EPS = torch.finfo(torch.bfloat16).eps
FIXTURE_SEED = 44
VISION_SEED = 1729
INDEXER_SEEDS = {3: 10001, 7: 20008}
# Qualified against both CPU GDN normalization baselines; CUDA is qualified
# independently. These are ordinary Normal(0, 0.02) CPU-generated projections.
MM_ROUTER_SEEDS = {
    0: 30003,
    1: 31002,
    2: 32004,
    3: 33011,
    4: 34001,
    5: 35007,
    6: 36023,
    7: 37036,
}


class ReferenceVisionAttention(InnerAttention):
    """Differentiable CPU oracle for the vision encoder's packed mask."""

    def forward(self, q_THK, k_THK, v_THV, *, attention_metadata, **kwargs):
        indices_T = torch.arange(q_THK.shape[0], device=q_THK.device)
        mask_TT = attention_metadata.mask_mod(
            0, 0, indices_T[:, None], indices_T[None, :]
        )
        with sdpa_kernel(SDPBackend.MATH):
            output_1HTV = torch.nn.functional.scaled_dot_product_attention(
                q_THK.transpose(0, 1)[None],
                k_THK.transpose(0, 1)[None],
                v_THV.transpose(0, 1)[None],
                attn_mask=mask_TT,
                scale=kwargs.get("scale"),
                enable_gqa=kwargs.get("enable_gqa", False),
            )
        return output_1HTV[0].transpose(0, 1)


def reference_text_config() -> Any:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        hidden_size=256,
        vocab_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        hc_count=4,
        hc_lowrank=32,
        ple_layer_ids=[2],
        ple_embed_dim=256,
        heads_per_ngram=2,
        ngram_vocab_size_base=257,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=16,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
        use_cache=False,
    )


def run(*, device: str, backbone_dtype: str, full_token_count: int) -> dict[str, Any]:
    with ExitStack() as reference_stack:
        reference_stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        return _run(
            device=device,
            backbone_dtype=backbone_dtype,
            full_token_count=full_token_count,
            reference_stack=reference_stack,
        )


def _run(
    *,
    device: str,
    backbone_dtype: str,
    full_token_count: int,
    reference_stack: ExitStack,
) -> dict[str, Any]:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpConfig,
        Qwen4ExpVisionConfig,
    )
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
        Qwen4ExpVisionModel,
    )

    selected_dtype = getattr(torch, backbone_dtype)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    config = build_model_config(
        "debugmodel_mm", seq_len=64, parallelism=ParallelismConfig()
    )
    for layer in config.layers:
        layer.feed_forward.router.aux_loss = None
    vision_config = config.vision_encoder
    assert vision_config is not None
    config.vision_encoder = None
    config.vocab_size = (
        config.tok_embeddings.num_embeddings
    ) = config.lm_head.out_features = 64
    config.layers[1].ple.ple_embedding.vocab_size = 64
    config.layers[1].ple.ple_embedding.eos_token_id = 63
    hf_config = reference_text_config()
    hf_config._attn_implementation = "sdpa"
    hf = Qwen4ExpForCausalLM(hf_config).to(device).eval()
    native = config.build().to(device).eval()
    adapter = Qwen38FlashNextStateDictAdapter(config, None)
    native.init_states()
    native.load_state_dict(adapter.from_hf(hf.state_dict()), strict=True)
    if torch.device(device).type == "cpu":
        cpu_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        reference_stack.callback(torch.set_num_threads, cpu_threads)
        # The AArch64 nightly's oneDNN Conv3d input backward fails a direct
        # Linear-equivalence check. Native CPU convolution preserves that
        # equivalence and keeps the reference's image gradients meaningful.
        reference_stack.enter_context(torch.backends.mkldnn.flags(enabled=False))
        from tests.unit_tests.gpu.models.qwen3_5.test_qwen3_5_deltanet import (
            _reference_causal_conv1d_varlen,
            ReferenceGatedDeltaKernel,
        )
        from torchtitan.models.common.attention.gdn import GDN

        for layer in native.layers.values():
            if isinstance(layer.attention, GDN):
                layer.attention.inner_attention.kernel = ReferenceGatedDeltaKernel()
        reference_stack.enter_context(
            patch(
                "torchtitan.models.common.attention.gdn._causal_conv1d_varlen",
                _reference_causal_conv1d_varlen,
            )
        )
    if selected_dtype == torch.float32:
        from torchtitan.models.common.moe.moe import RoutedExperts

        def fp32_experts(
            self, x_TD, topk_scores_TK, topk_expert_ids_TK, routing_map_TE, **kwargs
        ):
            # The shared routed path uses BF16 operands. This independent FP32
            # equation oracle checks architecture gradients without quantizing.
            output_TD = torch.zeros_like(x_TD)
            for expert_id in range(self.w13.weight.shape[0]):
                tokens_R, slots_R = torch.where(topk_expert_ids_TK == expert_id)
                gate_up_R2F = torch.nn.functional.linear(
                    x_TD[tokens_R], self.w13.weight[expert_id].flatten(0, 1)
                ).unflatten(-1, self.w13.weight.shape[1:3])
                hidden_RF = self.activation_fn(gate_up_R2F)
                expert_RD = torch.nn.functional.linear(
                    hidden_RF, self.w2.weight[expert_id]
                )
                output_TD = output_TD.index_add(
                    0, tokens_R, expert_RD * topk_scores_TK[tokens_R, slots_R, None]
                )
            return output_TD

        reference_stack.enter_context(
            patch.object(RoutedExperts, "forward", fp32_experts)
        )
    fixture_generator = torch.Generator(device="cpu").manual_seed(FIXTURE_SEED)

    def fixture_randn(shape, *, requires_grad=False):
        return (
            torch.randn(shape, generator=fixture_generator, device="cpu")
            .to(device)
            .requires_grad_(requires_grad)
        )

    positions = torch.arange(39, device=device)
    tokens = torch.randint(0, 63, (39,), generator=fixture_generator, device="cpu").to(
        device
    )
    tokens[7] = 63
    hidden = fixture_randn((39, 1024), requires_grad=True)
    results: dict[str, Any] = {}

    def compare(
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        atol: float = 2e-5,
        rtol: float = 2e-5,
    ) -> None:
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"{name}: {message}",
        )
        difference = actual.detach().double() - expected.detach().double()
        results[name] = {
            "max_abs": float(difference.abs().max()),
            "rms": float(difference.square().mean().sqrt()),
            "relative_l2": float(difference.norm() / expected.detach().double().norm()),
            "atol": atol,
            "rtol": rtol,
        }

    def compare_gradients(
        name: str,
        actual_loss: torch.Tensor,
        expected_loss: torch.Tensor,
        input_TC: torch.Tensor,
        actual_module: torch.nn.Module,
        expected_module: torch.nn.Module,
    ) -> None:
        actual_parameters = dict(actual_module.named_parameters())
        expected_parameters = dict(expected_module.named_parameters())
        assert actual_parameters.keys() == expected_parameters.keys()
        actual_grads = torch.autograd.grad(
            actual_loss, (input_TC, *actual_parameters.values())
        )
        expected_grads = torch.autograd.grad(
            expected_loss, (input_TC, *expected_parameters.values())
        )
        for field, actual_grad, expected_grad in zip(
            ("input", *actual_parameters), actual_grads, expected_grads
        ):
            compare(
                f"{name}_gradient/{field}",
                actual_grad,
                expected_grad,
                atol=3e-6,
                rtol=3e-5,
            )

    def compare_gradient(name, actual, expected, *, atol, rtol, compute_dtype=None):
        if (
            compute_dtype != torch.bfloat16
            and actual.dtype != torch.bfloat16
            and expected.dtype != torch.bfloat16
        ):
            compare(name, actual, expected, atol=atol, rtol=rtol)
            return
        # Packed native projections and separate HF projections round BF16
        # input-gradient sums differently. Bound both RMS and peak error; an
        # elementwise relative check is ill-conditioned near cancellation.
        reference = expected.detach().double()
        difference = actual.detach().double() - reference
        error_rms = difference.square().mean().sqrt()
        reference_rms = reference.square().mean().sqrt()
        assert error_rms <= atol + rtol * reference_rms, name
        assert difference.abs().max() <= atol + rtol * reference.abs().max(), name
        results[name] = {
            "max_abs": float(difference.abs().max()),
            "rms": float(error_rms),
            "relative_l2": float(difference.norm() / reference.norm()),
            "atol": atol,
            "rtol": rtol,
            "comparison": "RMS and infinity norm",
        }

    def compare_mapped_parameter_gradients(
        name,
        actual_module,
        expected_module,
        mapping,
        *,
        actual_prefix="",
        expected_prefix="",
        required_lm_head_gradient=None,
        compute_dtype=None,
        atol=3e-6,
        rtol=3e-5,
    ):
        expected_source = {
            expected_prefix + key: value.grad
            for key, value in expected_module.named_parameters()
            if value.grad is not None
        }
        if required_lm_head_gradient is not None:
            # The reused full-model adapter requires an LM-head entry, even
            # when checking vision alone. Its gradient was already checked.
            expected_source["lm_head.weight"] = required_lm_head_gradient
        expected_gradients = mapping.from_hf(expected_source)
        if required_lm_head_gradient is not None:
            expected_gradients.pop("lm_head.weight")
        actual_norm_squared = 0.0
        expected_norm_squared = 0.0
        gradient_dot = 0.0
        count = 0
        for key, parameter in actual_module.named_parameters():
            actual_grad = parameter.grad
            expected_grad = expected_gradients.pop(actual_prefix + key, None)
            assert (actual_grad is None) == (
                expected_grad is None
            ), f"{name}/{key}: gradient presence differs"
            if actual_grad is None:
                continue
            compare_gradient(
                f"{name}/{key}",
                actual_grad,
                expected_grad,
                atol=atol,
                rtol=rtol,
                compute_dtype=compute_dtype,
            )
            actual_norm = actual_grad.double().norm()
            expected_norm = expected_grad.double().norm()
            entry = results[f"{name}/{key}"]
            entry["norm_ratio"] = (
                float(actual_norm / expected_norm) if expected_norm else None
            )
            entry["cosine"] = (
                float(
                    torch.dot(
                        actual_grad.double().flatten(), expected_grad.double().flatten()
                    )
                    / (actual_norm * expected_norm)
                )
                if actual_norm and expected_norm
                else None
            )
            actual_norm_squared += float(actual_norm.square())
            expected_norm_squared += float(expected_norm.square())
            gradient_dot += float(
                torch.dot(
                    actual_grad.double().flatten(), expected_grad.double().flatten()
                )
            )
            count += 1
        assert (
            not expected_gradients
        ), f"{name}: unmapped gradients: {list(expected_gradients)}"
        actual_global_norm = actual_norm_squared**0.5
        reference_global_norm = expected_norm_squared**0.5
        torch.testing.assert_close(
            actual_global_norm, reference_global_norm, atol=atol, rtol=rtol
        )
        global_cosine = gradient_dot / (actual_global_norm * reference_global_norm)
        assert global_cosine >= 1 - rtol**2, name
        results[name + "/global"] = {
            "parameter_tensors": count,
            "actual_grad_norm": actual_norm_squared**0.5,
            "reference_grad_norm": expected_norm_squared**0.5,
            "norm_ratio": (actual_norm_squared / expected_norm_squared) ** 0.5,
            "cosine": global_cosine,
        }

    # Norms and mixing gradients are checked before the recurrent stack.
    actual, _, write = native.layers["0"].attn_hyper_connection(hidden)
    expected, _, hf_write = hf.model.layers[0].attn_hyper_connection(hidden[None])
    compare("residual_read", actual, expected[0])
    compare("residual_write", write, hf_write[0])
    compare_gradients(
        "residual",
        actual.square().mean() + write.square().mean(),
        expected.square().mean() + hf_write.square().mean(),
        hidden,
        native.layers["0"].attn_hyper_connection,
        hf.model.layers[0].attn_hyper_connection,
    )
    ple = native.layers["1"].ple
    hf_ple = hf.model.layers[1].ple
    # Exercise convolution backwards with nonzero weights; released initialization is zero.
    with torch.no_grad():
        hf_ple.conv1d.weight.copy_(
            torch.empty(hf_ple.conv1d.weight.shape, device="cpu")
            .uniform_(-0.02, 0.02, generator=fixture_generator)
            .to(hf_ple.conv1d.weight)
        )
        ple.conv1d.weight.copy_(hf_ple.conv1d.weight)
    hidden = hidden.detach().requires_grad_()
    actual = ple(hidden, input_ids_T=tokens, document_positions_T=positions)
    expected = hf_ple(hidden[None], tokens[None], None)[0]
    compare("ple", actual, expected)
    compare_gradients(
        "ple", actual.square().mean(), expected.square().mean(), hidden, ple, hf_ple
    )
    x = fixture_randn((39, 256))
    cos, sin = hf.model.rotary_emb(x[None], positions[None, None].expand(3, 1, -1))
    causal = torch.ones(39, 39, device=device, dtype=torch.bool).tril()[None, None]
    hf_mask = hf.model.layers[3].self_attn.indexer(x[None], (cos, sin), causal, None)[
        0, 0
    ]
    selected, _ = native.layers["3"].attention.indexer(
        x, positions=positions, document_positions_T=positions
    )
    mask = torch.zeros(39, 40, device=device, dtype=torch.bool).scatter(
        1, selected.long().masked_fill(selected < 0, 39), True
    )[:, :39]
    assert torch.equal(mask, hf_mask), "Indexer selection differs"
    results["indexer_selection"] = "exact"
    with sdpa_kernel(SDPBackend.MATH):
        compare(
            "qsa",
            native.layers["3"].attention(
                x, None, positions, document_positions_T=positions
            ),
            hf.model.layers[3].self_attn(x[None], (cos, sin), causal, None)[0][0],
        )
        # The router is tested independently from grouped expert computation.
        router_input = fixture_randn((39, 256))
        hf_router_logits, hf_router_scores, hf_router_indices = hf.model.layers[
            3
        ].mlp.gate(router_input)
        native_logits = native.layers["3"].feed_forward.router.gate(router_input)
        hf_logits = torch.nn.functional.linear(
            router_input.float(), hf.model.layers[3].mlp.gate.weight.float()
        )
        compare("router_logits", native_logits, hf_logits)
        assert torch.equal(
            native_logits.topk(2, -1).indices.sort(-1).values,
            hf_router_indices.sort(-1).values,
        )
        native_router_scores, native_router_indices, _ = native.layers[
            "3"
        ].feed_forward.router(router_input)
        assert torch.equal(native_router_indices, hf_router_indices)
        compare("router_mixing_weights", native_router_scores, hf_router_scores)
        # The shared fused GDN kernel accepts only FP16/BF16 QKV. Check its
        # actual training dtype after the new components' FP32 comparisons.
        rotary = copy.deepcopy(hf.model.rotary_emb)
        native = native.to(selected_dtype)
        hf = hf.to(selected_dtype)
        forward_atol = BF16_EPS / 2 if selected_dtype == torch.bfloat16 else 2e-5
        forward_rtol = BF16_EPS if selected_dtype == torch.bfloat16 else 2e-5
        gradient_atol = 1e-4 if selected_dtype == torch.bfloat16 else 1e-7
        gradient_rtol = 4 * BF16_EPS if selected_dtype == torch.bfloat16 else 3e-5
        hf.model.rotary_emb = rotary
        if selected_dtype == torch.bfloat16:
            # Use a rank-one gate with distinct row scales for the full training
            # fixture. Its selection margin isolates BF16 backbone computation
            # from near-tied expert changes; random gates are tested separately.
            with torch.no_grad():
                for layer_id in range(8):
                    weight = hf.model.layers[layer_id].mlp.gate.weight
                    weight.zero_()
                    weight[:, 0] = torch.arange(weight.shape[0], device=device) / 4
                    native.layers[str(layer_id)].feed_forward.router.gate.weight.copy_(
                        weight
                    )
        hidden = hidden.to(selected_dtype)
        cos, sin = hf.model.rotary_emb(
            x.to(selected_dtype)[None], positions[None, None].expand(3, 1, -1)
        )
        for layer_id in (0, 1, 3):
            hf_layer = hf.model.layers[layer_id]
            actual = native.layers[str(layer_id)](
                hidden.detach(),
                None,
                positions,
                ple_input_ids_T=tokens,
                document_positions_T=positions,
            )
            expected = hf_layer(
                hidden.detach()[None],
                position_embeddings=(cos, sin),
                attention_mask=causal if layer_id == 3 else None,
                ple_input_ids=tokens[None],
                past_key_values=None,
            )[0]
            compare(
                f"block_{layer_id}",
                actual,
                expected,
                atol=forward_atol,
                rtol=forward_rtol,
            )
        # A separately seeded, same-weight indexer fixture has positive pruning
        # margins. It compares full backward without imposing block selections.
        with torch.no_grad():
            for layer_id in (3, 7):
                generator = torch.Generator(device="cpu").manual_seed(
                    INDEXER_SEEDS[layer_id]
                )
                weight = hf.model.layers[
                    layer_id
                ].self_attn.indexer.index_qk_proj.weight
                fixture_weight = (
                    torch.empty(weight.shape)
                    .normal_(0, 0.02, generator=generator)
                    .to(device=device, dtype=selected_dtype)
                )
                weight.copy_(fixture_weight)
                native.layers[
                    str(layer_id)
                ].attention.indexer.index_qk_proj.weight.copy_(fixture_weight)
        full_tokens = tokens[:full_token_count]
        full_positions = positions[:full_token_count]
        with torch.no_grad():
            native_states = {}
            hf_states = {}
            native_routes = {}
            hf_routes = {}
            handles = []
            for layer_id in range(8):
                handles.append(
                    native.layers[str(layer_id)].register_forward_hook(
                        lambda module, inputs, output, i=layer_id: native_states.__setitem__(
                            i, output.detach()
                        )
                    )
                )
                handles.append(
                    hf.model.layers[layer_id].register_forward_hook(
                        lambda module, inputs, output, i=layer_id: hf_states.__setitem__(
                            i, output[0].detach()
                        )
                    )
                )
                handles.append(
                    native.layers[
                        str(layer_id)
                    ].feed_forward.router.register_forward_hook(
                        lambda module, inputs, output, i=layer_id: native_routes.__setitem__(
                            i, tuple(value.detach() for value in output[:2])
                        )
                    )
                )
                handles.append(
                    hf.model.layers[layer_id].mlp.gate.register_forward_hook(
                        lambda module, inputs, output, i=layer_id: hf_routes.__setitem__(
                            i, tuple(value.detach() for value in output)
                        )
                    )
                )
            logits = native(
                full_tokens,
                positions=full_positions,
                document_positions=full_positions,
                ple_input_ids=full_tokens,
            )
            hf_logits = hf(input_ids=full_tokens[None], use_cache=False).logits[0]
            for handle in handles:
                handle.remove()
            for layer_id in range(8):
                compare(
                    f"stack_block_{layer_id}",
                    native_states[layer_id],
                    hf_states[layer_id],
                    atol=forward_atol,
                    rtol=forward_rtol,
                )
                native_scores_TK, native_ids_TK = native_routes[layer_id]
                hf_logits_TE, hf_scores_TK, hf_ids_TK = hf_routes[layer_id]
                assert torch.equal(
                    native_ids_TK.sort(-1).values, hf_ids_TK.sort(-1).values
                ), f"Layer {layer_id} router IDs differ"
                # Compare scores in the same expert order, including exact ties.
                native_order_TK = native_ids_TK.argsort(-1)
                hf_order_TK = hf_ids_TK.argsort(-1)
                compare(
                    f"stack_router_{layer_id}",
                    native_scores_TK.gather(-1, native_order_TK),
                    hf_scores_TK.gather(-1, hf_order_TK),
                    atol=forward_atol,
                    rtol=forward_rtol,
                )
                sorted_logits_TE = hf_logits_TE.float().sort(-1, descending=True).values
                results[f"stack_router_margin_{layer_id}"] = float(
                    (sorted_logits_TE[:, 1] - sorted_logits_TE[:, 2]).min()
                )
            compare("logits", logits, hf_logits, atol=forward_atol, rtol=forward_rtol)
            native_logprobs_TV = logits.double().log_softmax(-1)
            hf_logprobs_TV = hf_logits.double().log_softmax(-1)
            hf_probs_TV = hf_logprobs_TV.exp()
            native_top_TK = logits.topk(5, -1).indices
            hf_top_TK = hf_logits.topk(5, -1).indices
            results["logit_probability_metrics"] = {
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        logits.double().flatten(), hf_logits.double().flatten(), dim=0
                    )
                ),
                "mean_kl_reference_to_native": float(
                    (hf_probs_TV * (hf_logprobs_TV - native_logprobs_TV)).sum(-1).mean()
                ),
                "top1_agreement": float(
                    (native_top_TK[:, 0] == hf_top_TK[:, 0]).double().mean()
                ),
                "top5_set_overlap": float(
                    (native_top_TK.unsqueeze(-1) == hf_top_TK.unsqueeze(-2))
                    .any(-1)
                    .double()
                    .mean()
                ),
            }
    # Evaluate and train the entire decoder with the same normalized next-token CE.
    # Auxiliary objectives are disabled here and checked by equation-level tests.
    for training in (False, True):
        native.train(training)
        hf.train(training)
        native.zero_grad(set_to_none=True)
        hf.zero_grad(set_to_none=True)
        embedded_inputs = []

        def retain_embedding_gradient(module, inputs, output):
            output.retain_grad()
            embedded_inputs.append(output)

        handles = [
            native.tok_embeddings.register_forward_hook(retain_embedding_gradient),
            hf.model.embed_tokens.register_forward_hook(retain_embedding_gradient),
        ]
        actual_logits = native(
            full_tokens,
            positions=full_positions,
            document_positions=full_positions,
            ple_input_ids=full_tokens,
        )
        expected_output = hf(
            input_ids=full_tokens[None], labels=full_tokens[None], use_cache=False
        )
        for handle in handles:
            handle.remove()
        mode = "train" if training else "eval"
        compare(
            f"{mode}/logits",
            actual_logits,
            expected_output.logits[0],
            atol=forward_atol,
            rtol=forward_rtol,
        )
        actual_loss = torch.nn.functional.cross_entropy(
            actual_logits[:-1].float(), full_tokens[1:]
        )
        expected_loss = expected_output.loss
        compare(f"{mode}/ce_loss", actual_loss, expected_loss, atol=2e-4, rtol=2e-4)
        actual_loss.backward()
        expected_loss.backward()
        compare_gradient(
            f"{mode}/embedding_input_gradient",
            embedded_inputs[0].grad,
            embedded_inputs[1].grad[0],
            atol=gradient_atol,
            rtol=gradient_rtol,
        )
        compare_mapped_parameter_gradients(
            f"{mode}/parameter_gradient",
            native,
            hf,
            adapter,
            atol=gradient_atol,
            rtol=gradient_rtol,
        )
    native.eval()
    hf.eval()

    # Packed boundaries use the same weights and are compared with isolated documents.
    # Keep the full-model document check below the eviction boundary. Pruned
    # selections, padding and boundaries are checked independently above and
    # in the CPU indexer oracle; equal-score topk ties can choose different
    # blocks when the candidate dimension changes between packed calls.
    packed_tokens = tokens[:30]
    packed_positions = torch.arange(15, device=device).repeat(2)
    metadata = native._get_attention_metadata(positions=packed_positions)
    with torch.no_grad():
        routes = {layer_id: [] for layer_id in range(8)}
        handles = [
            native.layers[str(layer_id)].feed_forward.router.register_forward_hook(
                lambda module, inputs, output, i=layer_id: routes[i].append(
                    output[1].detach()
                )
            )
            for layer_id in range(8)
        ]
        packed = native(
            packed_tokens,
            positions=packed_positions,
            document_positions=packed_positions,
            ple_input_ids=packed_tokens,
            attention_metadata=metadata,
        )
        isolated = torch.cat(
            [
                native(
                    packed_tokens[:15],
                    positions=positions[:15],
                    document_positions=positions[:15],
                    ple_input_ids=packed_tokens[:15],
                ),
                native(
                    packed_tokens[15:],
                    positions=positions[:15],
                    document_positions=positions[:15],
                    ple_input_ids=packed_tokens[15:],
                ),
            ]
        )
        for handle in handles:
            handle.remove()
        isolated_routes = {
            layer_id: torch.cat(values[1:]) for layer_id, values in routes.items()
        }
        changed_tokens = {
            layer_id: int(
                (values[0].sort(-1).values != isolated_routes[layer_id].sort(-1).values)
                .any(-1)
                .sum()
            )
            for layer_id, values in routes.items()
        }
        difference = packed.float() - isolated.float()
        results["packed_documents_unimposed"] = {
            "max_abs": float(difference.abs().max()),
            "rms": float(difference.square().mean().sqrt()),
            "changed_router_tokens_by_layer": changed_tokens,
        }
        # Match the independent eager convolution's BF16 materialization
        # point for this boundary diagnostic. The production varlen path fuses
        # SiLU before rounding; isolated calls materialize BF16 before SiLU.
        # Keep the original native drift report above and all primary HF checks.
        from tests.unit_tests.gpu.models.qwen3_5.test_qwen3_5_deltanet import (
            _reference_causal_conv1d_varlen,
        )

        with patch(
            "torchtitan.models.common.attention.gdn._causal_conv1d_varlen",
            _reference_causal_conv1d_varlen,
        ):
            matched_packed = native(
                packed_tokens,
                positions=packed_positions,
                document_positions=packed_positions,
                ple_input_ids=packed_tokens,
                attention_metadata=metadata,
            )
        compare(
            "packed_documents_matched_convolution",
            matched_packed,
            isolated,
            atol=BF16_EPS / 2,
            rtol=BF16_EPS,
        )
    hf_vision_cfg = Qwen4ExpVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        depth=2,
        num_heads=4,
        out_hidden_size=256,
        num_position_embeddings=1024,
    )
    hf_vision_cfg._attn_implementation = "sdpa"
    torch.manual_seed(VISION_SEED)
    hf_vision = Qwen4ExpVisionModel(hf_vision_cfg).to(device).eval()
    vision = vision_config.build().to(device).eval()
    if torch.device(device).type == "cpu":
        for block in vision.layers.values():
            block.attn.flex_attention = ReferenceVisionAttention()
    vision_model_config = copy.deepcopy(config)
    vision_model_config.vision_encoder = vision_config
    vision_adapter = Qwen38FlashNextStateDictAdapter(vision_model_config, None)
    vision_state = {
        "model.visual." + key: value for key, value in hf_vision.state_dict().items()
    }
    vision_state.update(
        {
            "lm_head.weight": hf.lm_head.weight,
            "model.language_model.embed_tokens.weight": hf.model.embed_tokens.weight,
        }
    )
    mapped = vision_adapter.from_hf(vision_state)
    vision.load_state_dict(
        {
            key.removeprefix("vision_encoder."): value
            for key, value in mapped.items()
            if key.startswith("vision_encoder.")
        },
        strict=True,
    )
    grid = torch.tensor([[1, 4, 4], [2, 2, 2]], device=device)
    patches = fixture_randn((24, 3 * 2 * 16 * 16))
    with torch.no_grad():
        compare(
            "vision",
            vision(patches, grid_thw=grid),
            hf_vision(patches, grid_thw=grid).pooler_output,
            atol=2e-5,
            rtol=2e-5,
        )
    for training in (False, True):
        vision.train(training)
        hf_vision.train(training)
        vision.zero_grad(set_to_none=True)
        hf_vision.zero_grad(set_to_none=True)
        native_patches = patches.detach().requires_grad_()
        hf_patches = patches.detach().clone().requires_grad_()
        actual_vision = vision(native_patches, grid_thw=grid)
        expected_vision = hf_vision(hf_patches, grid_thw=grid).pooler_output
        mode = "train" if training else "eval"
        compare(f"vision/{mode}/output", actual_vision, expected_vision)
        actual_vision.square().mean().backward()
        expected_vision.square().mean().backward()
        assert native_patches.grad is not None and hf_patches.grad is not None
        compare(
            f"vision/{mode}/input_gradient",
            native_patches.grad,
            hf_patches.grad,
            atol=3e-6,
            rtol=3e-5,
        )
        compare_mapped_parameter_gradients(
            f"vision/{mode}/parameter_gradient",
            vision,
            hf_vision,
            vision_adapter,
            actual_prefix="vision_encoder.",
            expected_prefix="model.visual.",
            required_lm_head_gradient=hf.lm_head.weight.grad,
        )
    # Full image/video fusion keeps document positions separate from MRoPE.
    native.vision_encoder = vision
    mm_config = Qwen4ExpConfig(
        text_config=hf.config,
        vision_config=hf_vision.config,
        image_token_id=61,
        video_token_id=62,
        vision_start_token_id=60,
        vision_end_token_id=59,
    )
    torch.manual_seed(VISION_SEED)
    hf_mm = Qwen4ExpForConditionalGeneration(mm_config)
    hf_mm.model.language_model = hf.model
    hf_mm.model.visual = hf_vision
    hf_mm.lm_head = hf.lm_head
    if selected_dtype == torch.bfloat16:
        # Ordinary generated router weights have positive multimodal cutoff gaps.
        # CPU generation keeps the fixture identical across CPU and accelerator runs.
        with torch.no_grad():
            for layer_id, seed in MM_ROUTER_SEEDS.items():
                gate_weight_ED = hf_mm.model.language_model.layers[
                    layer_id
                ].mlp.gate.weight
                generator = torch.Generator(device="cpu").manual_seed(seed)
                fixture_gate_ED = (
                    torch.empty(gate_weight_ED.shape, device="cpu")
                    .normal_(0, 0.02, generator=generator)
                    .to(gate_weight_ED)
                )
                gate_weight_ED.copy_(fixture_gate_ED)
                native.layers[str(layer_id)].feed_forward.router.gate.weight.copy_(
                    fixture_gate_ED
                )
    mapped_mm_state = vision_adapter.from_hf(hf_mm.state_dict())
    native_mm_state = native.state_dict()
    assert native_mm_state.keys() == mapped_mm_state.keys()
    assert all(
        torch.equal(native_mm_state[key], mapped_mm_state[key])
        for key in native_mm_state
    )
    mm_tokens_T = torch.tensor(
        [3, 60, 61, 61, 61, 61, 59, 4, 60, 62, 59, 5, 6, 7, 8, 9],
        device=device,
    )
    image_grid_NC = torch.tensor([[1, 4, 4]], device=device)
    video_grid_NC = torch.tensor([[1, 2, 2]], device=device)
    mm_types_T = torch.zeros_like(mm_tokens_T)
    mm_types_T[mm_tokens_T == 61] = 1
    mm_types_T[mm_tokens_T == 62] = 2
    rope_positions_3BT, _ = hf_mm.model.get_rope_index(
        mm_tokens_T[None], mm_types_T[None], image_grid_NC, video_grid_NC
    )
    rope_positions_TC = rope_positions_3BT[:, 0].transpose(0, 1).contiguous()
    document_positions_T = torch.arange(mm_tokens_T.numel(), device=device)
    assert not torch.equal(rope_positions_TC[:, 0], document_positions_T)
    mm_labels_T = mm_tokens_T.clone()
    mm_labels_T[(mm_tokens_T == 61) | (mm_tokens_T == 62)] = -100
    mm_pixel_generator = torch.Generator(device="cpu").manual_seed(45)
    image_patches_PF = torch.randn(
        16, 1536, generator=mm_pixel_generator, device="cpu"
    ).to(device)
    video_patches_PF = torch.randn(
        4, 1536, generator=mm_pixel_generator, device="cpu"
    ).to(device)
    for training in (False, True):
        native.train(training)
        hf_mm.train(training)
        native.zero_grad(set_to_none=True)
        hf_mm.zero_grad(set_to_none=True)
        native_image_PF = image_patches_PF.clone().requires_grad_()
        hf_image_PF = image_patches_PF.clone().requires_grad_()
        native_video_PF = video_patches_PF.clone().requires_grad_()
        hf_video_PF = video_patches_PF.clone().requires_grad_()
        embedding_gradients: list[torch.Tensor | None] = [None, None]
        fused_inputs = []
        native_mm_routes = {}
        native_mm_gate_logits = {}
        hf_mm_routes = {}
        with ExitStack() as mm_hooks:
            if selected_dtype == torch.bfloat16:
                for layer_id in range(8):
                    native_router = native.layers[str(layer_id)].feed_forward.router
                    hf_router = hf_mm.model.language_model.layers[layer_id].mlp.gate
                    mm_hooks.callback(
                        native_router.register_forward_hook(
                            lambda module, inputs, output, i=layer_id: native_mm_routes.__setitem__(
                                i, output[1].detach()
                            )
                        ).remove
                    )
                    mm_hooks.callback(
                        native_router.gate.register_forward_hook(
                            lambda module, inputs, output, i=layer_id: native_mm_gate_logits.__setitem__(
                                i, output.detach()
                            )
                        ).remove
                    )
                    mm_hooks.callback(
                        hf_router.register_forward_hook(
                            lambda module, inputs, output, i=layer_id: hf_mm_routes.__setitem__(
                                i, tuple(value.detach() for value in output)
                            )
                        ).remove
                    )
            for slot, embedding in enumerate(
                (native.tok_embeddings, hf_mm.model.language_model.embed_tokens)
            ):

                def retain_original_embedding_gradient(module, inputs, output, i=slot):
                    # Native scatter mutates this tensor in place. A hook before
                    # that mutation sees the original embedding node's gradient.
                    def save_gradient(gradient):
                        embedding_gradients[i] = gradient.detach()

                    output.register_hook(save_gradient)

                mm_hooks.callback(
                    embedding.register_forward_hook(
                        retain_original_embedding_gradient
                    ).remove
                )

            def capture_fused_input(module, inputs):
                fused_inputs.append(inputs[0].detach())

            for first_block in (
                native.layers["0"],
                hf_mm.model.language_model.layers[0],
            ):
                mm_hooks.callback(
                    first_block.register_forward_pre_hook(capture_fused_input).remove
                )
            actual_mm_logits_TV = native(
                mm_tokens_T,
                positions=rope_positions_TC,
                document_positions=document_positions_T,
                ple_input_ids=mm_tokens_T,
                pixel_values=native_image_PF,
                pixel_values_videos=native_video_PF,
                grid_thw=image_grid_NC,
                grid_thw_videos=video_grid_NC,
                special_tokens={"image_id": 61, "video_id": 62},
            )
            expected_mm_output = hf_mm(
                input_ids=mm_tokens_T[None],
                labels=mm_labels_T[None],
                position_ids=rope_positions_3BT,
                pixel_values=hf_image_PF,
                pixel_values_videos=hf_video_PF,
                image_grid_thw=image_grid_NC,
                video_grid_thw=video_grid_NC,
                mm_token_type_ids=mm_types_T[None],
                use_cache=False,
            )
            actual_mm_loss = torch.nn.functional.cross_entropy(
                actual_mm_logits_TV[:-1].float(), mm_labels_T[1:], ignore_index=-100
            )
            mode = "train" if training else "eval"
            compare(
                f"multimodal/{mode}/fused_input",
                fused_inputs[0],
                fused_inputs[1][0],
                atol=forward_atol,
                rtol=forward_rtol,
            )
            if selected_dtype == torch.bfloat16:
                for layer_id in range(8):
                    native_logits_TE = native_mm_gate_logits[layer_id].float()
                    hf_logits_TE, _, hf_ids_TK = hf_mm_routes[layer_id]
                    hf_logits_TE = hf_logits_TE.float()
                    assert torch.equal(
                        native_mm_routes[layer_id].sort(-1).values,
                        hf_ids_TK.sort(-1).values,
                    ), f"Multimodal layer {layer_id} router IDs differ"
                    native_sorted_TE = native_logits_TE.sort(-1, descending=True).values
                    hf_sorted_TE = hf_logits_TE.sort(-1, descending=True).values
                    gap_T = torch.minimum(
                        native_sorted_TE[:, 1] - native_sorted_TE[:, 2],
                        hf_sorted_TE[:, 1] - hf_sorted_TE[:, 2],
                    )
                    drift_T = (native_logits_TE - hf_logits_TE).abs().amax(-1)
                    scale_T = torch.maximum(
                        native_logits_TE.abs().amax(-1), hf_logits_TE.abs().amax(-1)
                    )
                    # A gap > 2 * drift preserves top-k sets; 4 * drift gives
                    # headroom, with four dtype epsilons and a 0.002 gap floor.
                    margin_limit_T = torch.maximum(
                        4 * drift_T, 4 * torch.finfo(selected_dtype).eps * scale_T
                    ).clamp_min(0.002)
                    assert bool(
                        (gap_T > margin_limit_T).all()
                    ), f"Multimodal layer {layer_id} fixture has a weak router margin"
                    results[f"multimodal/{mode}/router_margin_{layer_id}"] = {
                        "minimum_cutoff_gap": float(gap_T.min()),
                        "maximum_logit_drift": float(drift_T.max()),
                        "minimum_gap_over_limit": float((gap_T / margin_limit_T).min()),
                    }
            compare(
                f"multimodal/{mode}/logits",
                actual_mm_logits_TV,
                expected_mm_output.logits[0],
                atol=forward_atol,
                rtol=forward_rtol,
            )
            compare(
                f"multimodal/{mode}/ce_loss",
                actual_mm_loss,
                expected_mm_output.loss,
                atol=2e-4,
                rtol=2e-4,
            )
            actual_mm_loss.backward()
            expected_mm_output.loss.backward()
        native_embedding_gradient_TD, hf_embedding_gradient_BTD = embedding_gradients
        assert native_embedding_gradient_TD is not None
        assert hf_embedding_gradient_BTD is not None
        assert native_image_PF.grad is not None and hf_image_PF.grad is not None
        assert native_video_PF.grad is not None and hf_video_PF.grad is not None
        assert native_image_PF.grad.norm() > 0 and native_video_PF.grad.norm() > 0
        for name, actual_gradient, expected_gradient in (
            ("image_patch_gradient", native_image_PF.grad, hf_image_PF.grad),
            ("video_patch_gradient", native_video_PF.grad, hf_video_PF.grad),
            (
                "embedding_output_gradient",
                native_embedding_gradient_TD,
                hf_embedding_gradient_BTD[0],
            ),
        ):
            compare_gradient(
                f"multimodal/{mode}/{name}",
                actual_gradient,
                expected_gradient,
                atol=gradient_atol,
                rtol=gradient_rtol,
                compute_dtype=selected_dtype,
            )
        placeholder_T = (mm_tokens_T == 61) | (mm_tokens_T == 62)
        assert torch.count_nonzero(native_embedding_gradient_TD[placeholder_T]) == 0
        assert torch.count_nonzero(hf_embedding_gradient_BTD[0, placeholder_T]) == 0
        compare_mapped_parameter_gradients(
            f"multimodal/{mode}/parameter_gradient",
            native,
            hf_mm,
            vision_adapter,
            atol=gradient_atol,
            rtol=gradient_rtol,
            compute_dtype=selected_dtype,
        )
        results[f"multimodal/{mode}/loss_values"] = {
            "native_ce": float(actual_mm_loss.detach()),
            "reference_ce": float(expected_mm_output.loss.detach()),
            "native_image_gradient_norm": float(native_image_PF.grad.double().norm()),
            "native_video_gradient_norm": float(native_video_PF.grad.double().norm()),
        }
    import transformers

    return {
        "reference_revision": REFERENCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "device": device,
        "torch": torch.__version__,
        "component_dtype": "float32",
        "torch_revision": torch.version.git_version,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(device)
        if torch.device(device).type == "cuda"
        else None,
        "cuda": torch.version.cuda,
        "backbone_dtype": str(selected_dtype),
        "full_token_count": full_token_count,
        "full_model_indexer_seeds": INDEXER_SEEDS,
        "input_fixture_seed": FIXTURE_SEED,
        "vision_weight_seed": VISION_SEED,
        "multimodal_router_seeds": MM_ROUTER_SEEDS
        if selected_dtype == torch.bfloat16
        else None,
        "expert_compute": "FP32 per-expert equation oracle"
        if selected_dtype == torch.float32
        else "shared BF16 grouped GEMM",
        "seed": 42,
        "attention_backend": "SDPA MATH",
        "gdn_path": "fused GPU kernel"
        if torch.device(device).type == "cuda"
        else "independent PyTorch recurrence (shared Qwen3.5 test oracle)",
        "tolerance_rationale": (
            "FP32 operation-order error; BF16 half-epsilon absolute plus one epsilon relative. "
            "Packed convolution matching is diagnostic only; native routing and drift "
            "are reported separately."
        ),
        "tokens": tokens.tolist(),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone-dtype", choices=["float32", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--full-token-count", type=int, choices=[16, 39], default=16)
    parser.add_argument(
        "--output", default="outputs/qwen3_8_flash_next_reference_parity.json"
    )
    args = parser.parse_args()
    report = run(
        device=args.device,
        backbone_dtype=args.backbone_dtype,
        full_token_count=args.full_token_count,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
