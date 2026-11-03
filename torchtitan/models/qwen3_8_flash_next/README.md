# Qwen3.8-Flash-Next

Flash-Next reuses Qwen3.5's Gated DeltaNet, vision encoder, FSDP and optimizer
lifecycle. Its model package implements four-stream gated residuals, sparse
QSA attention, lexical embeddings, and the checkpoint's router rounding.

## Reference and checkpoint

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
  checkpoint revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Transformers `Qwen4ExpForCausalLM`, `Qwen4ExpVisionModel`, and
  `Qwen4ExpForConditionalGeneration` at revision
  `494470d677d05d44153d235d795e0d01040307ab`.
- The adapter inventories 1,627 released decoder/vision keys on meta tensors.
  It packs the 128 numeric lexical shards in checkpoint order. The 31 MTP keys
  are outside this training model and are skipped with a warning.

## Architecture

| Path | Released shape or configuration |
| --- | --- |
| Decoder | 48 layers, model width 2,560, vocabulary 248,320 |
| Gated residual | Four streams, grouped offset RMSNorm, gate rank 320 |
| Attention | Three shared GDN layers followed by one QSA layer |
| QSA | 24 query / 2 KV heads, head width 256, partial MRoPE |
| Indexer | Four heads of width 128, four-token blocks, 2,048-token budget plus causal tail |
| MoE | 512 experts, top 10, routed/shared width 640, sigmoid shared gate |
| PLE | Layer 2, 16 bigram/trigram heads of width 160, gated lexical injection |
| Lexical table | 320,001,536 rows by 160 columns, 51.2B parameters |
| Vision | 27 layers, width 1,152, 16-pixel patches, merger output width 2,560 |

The residual state is `[T, 4 * D]`. Each sublayer reads `[T, D]` and writes its
update into the four streams. PLE uses original token IDs and one-dimensional
document positions, independently of multimodal rotary coordinates.

QSA caches document/block/tail metadata once per input. Equal ReLU scores select
earlier blocks first, so masked future tokens cannot change a past selection.
HF uses an unspecified cutoff tie policy; reference comparisons use positive
selection margins and report natural router/block choices separately.

The architecture supplies no indexer training coefficient. `indexer_loss=None`
disables the objective; test recipes explicitly select coefficient 1.0. The
attention-teacher KL uses detached teacher and backbone inputs, so it trains
only the indexer. MoE auxiliary counts exclude padded tokens.

## Build and train

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config

parallelism = ParallelismConfig(data_parallel_shard_degree=2)
model = build_model_config("debugmodel", parallelism=parallelism, seq_len=64)
```

Flavors are `debugmodel`, `debugmodel_mm`, and `125B-A6B`. Reduced fixtures retain
both attention types, PLE, four residual streams, and routed/shared experts.
This change supports FSDP. TP, EP, PP and CP are rejected explicitly.

```bash
NGPU=1 MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
  CONFIG=qwen38_flash_next_debugmodel ./run_train.sh
```

## Numerical coverage

| Path | Modes and checks | Reference |
| --- | --- | --- |
| Gated residual and norms | FP32/BF16 forward, input and parameter gradients | Independent equations and pinned HF |
| Lexical embedding | Fresh hash initialization, EOS/packed resets, padding, lookup and convolution gradients | Independent hash/convolution equations |
| QSA | Pruned block selection, causal tails, padding, exact ties, selected-attention backward | Chronological block oracle and selected-token SDPA |
| Indexer teacher | Detached KL, indexer-only gradients, global token denominator | Independent differentiable KL |
| Router | Natural expert sets, rounded-logit softmax, mixing weights, eager/compiled backward | FP32 softmax over BF16 logits |
| Decoder | Train/eval logits, normalized CE, embedding input and all 184 mapped parameter gradients | Same-weight HF; 39 tokens, eight-layer fixture |
| Vision | Packed images/video, merger, pixel and 41 parameter gradients | Same-weight HF vision |
| Full multimodal model | Train/eval image/video fusion, MRoPE, masked CE, pixel and all 225 parameter gradients | Same-weight HF conditional generation |
| Adapter | Strict roundtrip and all released key/shape mappings | Published checkpoint headers |
| Training state | Padding exclusion, auxiliary accumulation/reset across microbatches and remat | Public model and optimizer hooks |

```bash
pytest tests/unit_tests/cpu/models/test_qwen3_8_flash_next.py -q
python -m scripts.checkpoint_conversion.numerical_tests_qwen3_8_flash_next \
  --device cpu --backbone-dtype float32 --full-token-count 39
python -m scripts.checkpoint_conversion.numerical_tests_qwen3_8_flash_next \
  --device cuda --backbone-dtype bfloat16 --full-token-count 39
```

Fixtures use CPU-generated inputs (seed 44), indexer seeds 10001/20008, and
vision seed 1729. CPU FP32 uses independent recurrent GDN and per-expert equations;
CUDA uses the real fused GDN and shared grouped-expert kernels. All HF attention
comparisons run under the same SDPA MATH backend. BF16 gradients use unchanged
RMS and infinity norm bounds (`atol=1e-4`, `rtol=0.03125`); FP32 checks use
`atol=1e-7`, `rtol=3e-5`. Full-model metrics include gradient norms and cosine.

The BF16 multimodal fixture uses ordinary CPU-generated router projections
with recorded per-layer seeds. Natural expert sets must agree, and each top-k
cutoff gap must exceed four times both measured logit drift and dtype epsilon
times logit scale, with a 0.002 floor. FP32 uses the original multimodal fixture.
The primary text and standalone vision fixtures keep their original weights.

Packed/isolated production GDN paths round on different sides of fused SiLU.
The harness reports native drift and natural routing, then matches only that
convolution arithmetic to check document isolation with unchanged bounds.

## Limits

- Dynamic QSA selections require `training.disable_accelerator_graphs=True`.
- Selected-token gathers and SDPA MATH are bounded by query chunks. Full-scale
  sparse-attention throughput and memory have not been benchmarked.
- The meta inventory has 177.4B decoder/vision parameters including the lexical
  table. Full-checkpoint materialization, distributed conversion of that table,
  and full-size training are not validated here.
- Inference KV caching and MTP are outside this training port.
