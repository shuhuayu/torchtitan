# Lightweight FSDP2 Muon Implementation and Study

Status: implemented on the `fsdpmuon` branch

This implementation enables exact full-matrix Muon for TorchTitan's pure FSDP2
path. It is deliberately small: one component module, one optimizer
registration, focused tests, and three communication strategies. It does not
attempt to implement the complete Canzona paper or support TorchTitan's full
parallelism matrix.

## Scope

The first version supports:

- `torch.distributed.fsdp.fully_shard` parameters and gradients.
- A single one-dimensional FSDP shard mesh.
- Two-dimensional weights sharded uniformly with `Shard(0)`.
- Muon for selected matrix weights and AdamW for all remaining parameters.
- Flat, shape-grouped, and layer-pipelined owner redistribution with cached
  communication buffers.
- Configurable consecutive-layer buckets and asynchronous overlap of bucket
  `i + 1`'s gather with bucket `i`'s Newton-Schulz work in the
  layer-pipelined strategy.
- The existing TorchTitan LR scheduler and distributed checkpoint path.

The first version does not support or validate:

- TP, HSDP, EP, and multi-axis DTensor placements.
- `Shard(1)`, uneven shards, and non-2D Muon parameters.
- CPU offload, optimizer compilation, and CUDA graph capture.
- Missing gradients for selected Muon parameters.
- SOAP, fused projection splitting, per-head lowering, and arbitrary user-defined
  bucket specifications.

The supported training configuration is therefore FSDP only:

```text
dp_shard > 1
dp_replicate = cp = tp = pp = ep = 1
spmd_backend = "default"
```

## Files

The implementation changes only these code paths:

```text
torchtitan/components/muon.py                      # new implementation
torchtitan/components/optimizer.py                 # import + registry entry
torchtitan/models/llama3/config_registry.py        # debug and benchmark configs
tests/unit_tests/test_muon.py                      # local algorithm tests
tests/integration_tests/h100.py                    # 2-GPU FSDP2 test
```

`muon.py` owns layout validation, the small static execution plan,
communication, and the Muon step. No experiment package, model changes, FSDP
hooks, trainer changes, or custom checkpoint manager are needed.

Register the component in `OptimizersContainer._resolve_optimizer_cls`:

```python
optimizer_classes = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "Muon": AllToAllMuon,
}
```

The user-facing name describes the algorithm. Core TorchTitan currently lowers
`Muon` to `AllToAllMuon`; a placement-aware container can select a different
lowering without changing parameter-group configuration.

## High-Level Structure

```text
                         AllToAllMuon
                    optimizer orchestrator
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
   PLANNING METADATA                       TENSOR EXECUTION
   built at initialization                 performed every step
          |                                       |
          v                                       v
 MuonMatrixSpec                    _MuonComputeStorageBinding
 (supports future per-head specs)                 |
          |                                       |
          v                                       v
 assign_muon_matrix_owners()            gather local shards
          |                                       |
          v                                       v
 MuonMatrixAssignment                   _ScratchMuon on owner
                                                  |
                                                  v
                                        scatter delta shards
                                                  |
                                                  v
                                        update DTensor storage
```

## Design Answers

### Does this require an FSDP2 change?

No. FSDP2 should continue to own parameter and gradient storage during forward
and backward. `AllToAllMuon` is an optimizer-side consumer of the resulting
DTensors. It reads their public mesh, placement, shape, and local shard, then
uses public distributed collectives during `optimizer.step()`.

The ownership boundary is:

1. FSDP2, and eventually FlexShard, owns the persistent storage layout.
2. The optimizer owns any temporary layout needed by its update kernel.
3. The optimizer returns the update to the persistent storage layout before
   `step()` completes.

This requires no FSDP hooks, private FSDP imports, gradient replacement, or
trainer changes. It also means the shortest implementation can remain a single
optimizer component instead of becoming an FSDP2 feature.

### Storage Sharding vs. Optimizer Compute Sharding

These are different contracts and should not be conflated:

- **Storage sharding** describes where parameters, gradients, and persistent
  optimizer state live between operations. In this implementation they are
  DTensors with `(Shard(0),)` on the FSDP mesh. This layout is also the
  checkpoint layout.
- **Optimizer compute sharding** describes where a temporary value must live
  while an optimizer kernel runs. Muon's Newton-Schulz kernel needs one logical
  full matrix, even though its input and output are storage-sharded.

The current data flow is:

```text
parameter:       Shard(0) ---------------------------------> Shard(0)
momentum:        Shard(0) -- local elementwise update -----> Shard(0)
pre-NS input:    Shard(0) -- gather to assigned rank -----> full matrix on owner
Muon delta:      full matrix on owner -- scatter ---------> Shard(0)
```

For the lightweight implementation, storage sharding is inferred from the
actual DTensor. Optimizer compute ownership is represented separately by
`MuonMatrixSpec` and `MuonMatrixAssignment`.

If TorchTitan later exposes optimizer compute sharding, it should be attached to
the optimizer parameter group or optimizer component, not to FSDP2 and not to
`ShardingConfig.state_shardings`. The latter describes persistent module state
used by model execution. A future optimizer annotation needs to describe three
things separately:

```text
source storage layout:       the parameter/gradient DTensor layout
compute ownership policy:    replicated, or full on one assigned rank
result storage layout:       normally the original parameter layout
```

Existing `SpmdLayout` can describe symmetric layouts such as `Shard(0)` and
`Replicate()`. It cannot by itself describe "parameter P0 is full only on rank
0, while parameter P1 is full only on rank 1." That is asymmetric ownership and
requires a rank assignment and collective schedule in addition to tensor
placements.

### Reusable Matrix Owner Plan

The reusable contract is logical matrix ownership, not a particular collective:

```python
matrix = MuonMatrixSpec(
    fqn="layers.0.feed_forward.w1.weight",
    shape=torch.Size((hidden_dim, model_dim)),
    param_offset=0,
)

MuonMatrixAssignment(matrix=matrix, owner_rank=0)
```

`owner_rank` is relative to the storage mesh. The deterministic
`assign_muon_matrix_owners()` first sorts by matrix size and assigns each matrix
to the currently least-loaded rank. FQN, parameter offset, shape, and rank are
deterministic tie-breakers. It returns the result in canonical FQN order, and
the DTensor lowering validates that every rank built the same plan.

This is a largest-first greedy balance, not exact bin packing. Exact equal loads
are impossible for some indivisible sets of matrices. The flat lowering handles
that case with variable all-to-all split sizes; it does not pad or split a
logical matrix. The shape-grouped lowering applies the same assignment within
each shape group, which becomes deterministic round-robin because all matrices
in that group have equal size. The layer-pipelined lowering applies the
assignment independently within each configured layer bucket, allowing larger
buckets to balance an indivisible per-layer matrix mix across adjacent layers.

For the current Llama configuration, every parameter contains one logical
matrix and `param_offset` is zero. The element offset and logical shape make the
plan usable later when one parameter contains several independently optimized
matrices, such as attention heads or packed experts.

Storage backends consume the same assignment differently:

- FSDP2 keeps `Shard(0)` storage and treats the owner as temporary optimizer
  compute ownership. All-to-all materializes the matrix on that owner.
- A future owner-based FlexShard placement can use the assignment as persistent
  storage ownership. The owner runs ordinary `torch.optim.Muon`, and the next
  model-time unshard distributes the updated parameter.

This keeps the planner independent of both DTensor and FlexShard.

### Simple Redistribution: All-Gather

The shortest correctness path is a conventional DTensor redistribution:

```text
pre-NS:  Shard(0) -> Replicate()
compute: every rank runs upstream Muon on the same full matrix
result:  every rank keeps the shard corresponding to its storage layout
```

This PR does not implement the all-gather backend; it is included only as the
simple reference design.

This can be expressed as a `Shard(0)` to `Replicate()` redistribution and does
not need an owner schedule. It is a useful reference implementation because the
layout transition is obvious and every rank independently produces the same
answer.

It is not the preferred training path. Every rank materializes every full
matrix and repeats the same Newton-Schulz work. Running Muon only on one rank
after an all-gather would remove duplicate compute but would still replicate the
input unnecessarily and would need another scatter or broadcast for the
result. The all-gather version is therefore best kept as a test oracle or debug
backend, not as the default implementation.

### Efficient Redistribution: Owner All-to-All

All three implemented strategies follow the same logical transition:

```text
Shard(0) input -> full matrix on one temporary owner -> Shard(0) delta
```

Every matrix is sent only to its assigned owner, Newton-Schulz runs once, and a
reverse all-to-all returns the row shards. They differ in buffer layout and
collective granularity.

**Flat strategy (`all_to_all_strategy="flat"`)**

- Packs local shards of all shapes into one owner-major flat buffer.
- Uses precomputed send offsets, owner offsets, and variable split sizes.
- Reconstructs each owner-local matrix with one strided copy.
- Uses one forward and one reverse `all_to_all_single` per optimizer step.
- Adds no padding. Greedy ownership approximately, but not always exactly,
  balances the bytes received by each rank.

**Shape-grouped strategy (`all_to_all_strategy="shape_grouped"`)**

- Builds one regular `[owner, slot, local_numel]` buffer per unique shape.
- Pads each owner's slot count to the maximum for that shape group.
- Reconstructs each matrix from a regular strided slot.
- Uses one forward and one reverse `all_to_all_single` per shape group.

**Layer-pipelined strategy (`all_to_all_strategy="layer_pipelined"`)**

- Extracts and numerically sorts canonical `layers.<index>` prefixes from the
  parameter FQNs.
- Groups up to `K = num_layers_per_bucket` consecutive layers in each bucket.
- Builds one cached variable-split flat exchange plan and assigns matrix owners
  independently within each bucket.
- Uses one forward and one reverse `all_to_all_single` per bucket, for
  `2 * ceil(num_layers / K)` calls per optimizer step.
- Launches bucket `i + 1`'s gather with `async_op=True` before running bucket
  `i`'s Newton-Schulz work.
- Waits for bucket `i`'s reverse scatter before applying that bucket and
  advancing.

The pipeline is:

```text
pack bucket 0; launch gather 0
for each bucket i:
    wait for gather i
    pack bucket i + 1; launch gather i + 1 asynchronously
    run Newton-Schulz for bucket i
    launch and wait for scatter i
    apply bucket i's local update
```

`Work.wait()` establishes the current-stream dependency on the completed input
gather. The next gather is already enqueued on the process group's communication
stream before current-bucket compute is enqueued on the default stream, allowing
the two streams to progress concurrently. Adjacent buckets use distinct buffers,
so no buffer is reused while a collective is in flight. The reverse scatter is
not overlapped, and the first gather and final scatter have no adjacent work to
hide them.

All plans, owner assignments, offsets, split-size lists, and communication
buffers are built once during optimizer construction and reused every step.
Caching this metadata avoids repeated Python planning and allocation, but it
cannot remove the forward gather or reverse scatter required by an independently
completed bucket. Increasing `K` is what coalesces those exchanges. `K=1` is the
original per-layer schedule; `K >= num_layers` degenerates to one bucket and two
collectives, like `flat`, without useful inter-bucket overlap.

| Property | Flat | Shape grouped | Layer pipelined |
| --- | --- | --- | --- |
| Collective calls per step | 2 | `2 * num_shape_groups` | `2 * ceil(num_layers / K)` |
| Split sizes | Variable | Equal | Variable |
| Padding | None | Dummy matrix slots | None |
| Owner scope | Whole optimizer | Per shape | Per layer bucket |
| Expected strength | Lowest launch count | Regular buffers | Gather/compute overlap |

Compared with an all-gather baseline, each owner strategy keeps one full copy
and one Newton-Schulz computation per matrix instead of one per rank.

### Layer Buckets and Future UX

Optimizer parameter groups are not bucket boundaries. `AllToAllMuon` combines
all of its groups into one execution plan, and every configured
`all_to_all_strategy` value must agree. This avoids changing communication
behavior when a user splits parameter groups only to set learning rates or
weight decay.

For the current Llama implementation, `layer_pipelined` derives deterministic
buckets from canonical FQNs such as `layers.7.feed_forward.w1.weight` and groups
up to `num_layers_per_bucket` numerically consecutive layer prefixes. A selected
parameter without a `layers.<index>` segment is rejected during optimizer
construction. This fixed-size policy is intentionally narrower than a public
bucket API.

A general `bucket_spec` should be designed with FlexShard or the upstream
PyTorch optimizer-sharding API, not added as a Llama-specific TorchTitan
configuration. A future API could conceptually look like:

```python
flex_shard(torch.optim.Muon, bucket_spec=...)
```

Once arbitrary or heterogeneous buckets exist, collective layout policy may
belong on each bucket specification. In the current implementation the three
strategies define the optimizer's complete schedule, so
`all_to_all_strategy` remains optimizer-wide.

### Reuse with FlexShard

The useful compatibility point is the matrix owner plan, not an FSDP2 hook.
The current component already avoids FSDP2 internals.

For a placement-aware optimizer resolver, FlexShard needs to expose, directly
or through a TorchTitan adapter:

- The logical matrix shape and its location within the parameter.
- Whether local storage contains a complete matrix or a matrix partition.
- The owner rank or the placement metadata from which to construct the owner
  assignment.

With FlexShard's `GroupedOwned` placement, the assigned rank persistently holds
the complete parameter and receives its reduced gradient. It can run ordinary
`torch.optim.Muon` locally; non-owners omit the empty parameter from their
optimizer. There is no optimizer-time gather, delta scatter, or updated-
parameter redistribution. The placement's next model-time unshard consumes the
updated owner-local storage.

The all-to-all methods are therefore the DTensor lowering, not the shared
abstraction. FlexShard can consume `MuonMatrixAssignment` while constructing an
owner placement and use a placement-aware optimizer resolver to select ordinary
Muon.

### Recommendation

- Keep FSDP2 unchanged.
- Let user configuration select the `Muon` algorithm, not a storage backend.
- Keep owner assignment independent of storage and communication.
- Use owner all-to-all as the DTensor `Shard(0)` lowering.
- Keep `flat` as the conservative default because it accepts arbitrary FQNs and
  minimizes collective count, but benchmark `shape_grouped` on the target
  topology.
- Keep `layer_pipelined` as an experimental, topology-tuned option rather than
  the default. `K=8` won the 1B two-H100 sweep, while `K=1` did not recover its
  launch, scatter, and owner-imbalance costs.
- Use ordinary local Muon when storage already contains complete matrices.
- Use all-gather only as a simple correctness reference if one is needed.
- Keep persistent state in the storage layout and make only the Newton-Schulz
  input and output transient.
- Treat a future optimizer compute annotation as TorchTitan optimizer metadata,
  with a separate ownership/scheduling policy for layouts that `SpmdLayout`
  cannot express.

## Key Simplification: Keep Momentum Sharded

Muon has two distinct parts:

```text
momentum update:       elementwise
Newton-Schulz update:  full-matrix
```

For full gradient `G`, momentum buffer `B`, and FSDP shards indexed by `r`:

```text
B_r = momentum(B_r, G_r)
concat(B_r) = momentum(concat(B_r), concat(G_r))
```

The momentum and Nesterov operations commute with FSDP sharding. Therefore each
rank can update its local momentum shard first. Only the resulting matrix that
enters Newton-Schulz must be reconstructed on one rank.

This is the main difference from the FSDP-Canzona prototype. It gives the same
Muon result while keeping `state[param]["momentum_buffer"]` as a normal sharded
DTensor. Consequently, the current `OptimizersContainer.state_dict()` and DCP
resharding logic can save and restore it without an owner-state adapter.

## `AllToAllMuon` Structure

`AllToAllMuon` is a `torch.optim.Muon` subclass with a custom `step()`.
Reusing its constructor preserves upstream validation, hyperparameter names, and
state-dict conventions.

At construction:

1. Resolve one consistent strategy from the constructor and parameter groups,
   defaulting to `flat` when neither specifies one.
2. Read canonical FQNs from each param group's `param_names`.
3. Build one `MuonMatrixSpec` per selected parameter.
4. Require every parameter to be a 2D DTensor.
5. Require a one-dimensional device mesh and placements equal to `(Shard(0),)`.
6. Require `full_rows == local_rows * fsdp_world_size` and matching columns.
7. Require every selected parameter to use the same FSDP process group and dtype.
8. Assign each logical matrix to a mesh-local owner with deterministic greedy
   load balancing.
9. Build one global flat plan, one regular padded plan per shape, or one flat
   plan per consecutive-layer bucket, including cached offsets, split sizes,
   and buffers.
10. All-gather a hash of the strategy, bucket size, FQNs, shapes, groups,
    owners, and order; fail if ranks disagree.

## Optimizer Step

Flat and shape-grouped steps have four phases. The layer-pipelined strategy
interleaves the same phases across adjacent layer buckets.

### 1. Local Momentum

Use each DTensor's local shard to update the real sharded optimizer state with
upstream Muon semantics:

```python
buf = state.setdefault("momentum_buffer", torch.zeros_like(grad))
buf.lerp_(grad, 1 - momentum)
pre_ns = grad.lerp(buf, momentum) if nesterov else buf
```

The state remains associated with the real FSDP parameter and is the only
persistent Muon state.

### 2. Redistribute to Owners

Pack each rank's local `pre_ns.to_local()` shards by destination owner. The
flat plan exchanges all matrices at once with variable splits. The
shape-grouped plan exchanges a regular padded buffer once per shape. The
layer-pipelined plan exchanges one variable-split buffer per layer bucket:

```text
rank 0 sends P0 shard -> owner 0, P1 shard -> owner 1, ...
rank 1 sends P0 shard -> owner 0, P1 shard -> owner 1, ...
```

Each owner uses cached offsets or slots to reconstruct every full matrix
assigned to it. There is no per-step sorting, plan construction, or allocation.

`DTensor.to_local()` is used to access each gradient and momentum shard. For
the supported `Shard(0)` layout this is a local view operation, not another
collective, but repeated Python and DTensor dispatch still has cost. Flat,
shape-grouped, and every layer-bucket size perform the same per-parameter local
state access, so changing `K` does not reduce that cost. Cached owner offsets
and split-size lists remove per-step metadata work; coalescing layers is what
reduces all-to-all calls.

### 3. Run Upstream Muon

Use the public `torch.optim.Muon` implementation through a full-sized scratch
parameter on the owner:

```text
scratch parameter = 0
scratch gradient = reconstructed pre-Newton-Schulz matrix
scratch Muon settings:
    momentum = 0
    nesterov = False
    weight_decay = 0
    lr, ns_steps, coefficients, eps, adjust_lr_fn = real group values
scratch_optimizer.step()
full_delta = scratch parameter
```

Momentum is set to zero because the real sharded momentum was already applied.
The scratch parameter has the global matrix shape, so upstream Muon's learning
rate adjustment uses the correct dimensions. Scratch state is temporary and is
not included in `AllToAllMuon.state_dict()`.

This avoids copying PyTorch's Newton-Schulz kernel or importing private
`torch.optim._muon` functions.

### 4. Scatter and Update

Each owner overwrites its receive buffer with row shards of the full deltas and
runs the same all-to-all schedule in reverse.

Wrap the received local delta with `DTensor.from_local`, using the real
parameter's mesh, placements, global shape, and stride. Then update the real
parameter through DTensor operations:

```python
param.mul_(1 - lr * weight_decay)
param.add_(delta_dtensor)
```

`step()` finishes every collective and local update before returning. In the
layer-pipelined strategy, only the next bucket's forward gather is asynchronous;
the current bucket's gather and scatter are complete before its update is
applied.

## Configuration

The `llama3_debugmodel_fsdp_muon` config uses the existing mixed-optimizer
interface:

```python
OptimizersContainer.Config(
    implementation="fused",
    param_groups=[
        ParamGroupConfig(
            pattern=(
                r"^layers\.\d+\."
                r"(?:_checkpoint_wrapped_module\.)?"
                r"(?:attention|feed_forward)\..*\.weight$"
            ),
            optimizer_name="Muon",
            optimizer_kwargs={
                "lr": 8e-4,
                "weight_decay": 0.1,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "adjust_lr_fn": "match_rms_adamw",
                "all_to_all_strategy": "flat",
                "fused": False,
                "foreach": False,
            },
        ),
        ParamGroupConfig(
            pattern=r".*",
            optimizer_name="AdamW",
            optimizer_kwargs={
                "lr": 8e-4,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.1,
                "fused": True,
            },
        ),
    ],
)
```

For the current six-layer debug model, the first pattern selects all 30 matrix
weights under attention and feed-forward modules. The catch-all group assigns
the remaining 15 embedding, normalization, and output parameters to AdamW.

Use `llama3_debugmodel_fsdp_muon_shape_grouped` and
`llama3_debugmodel_fsdp_muon_layer_pipelined` to exercise the other layouts.
The `llama3_1b_fsdp_muon_{flat,shape_grouped,layer_pipelined}` and
`llama3_8b_fsdp_muon_{flat,shape_grouped,layer_pipelined}` configs provide the
same comparison on larger models.

TorchTitan stores optimizer kwargs in parameter-group dictionaries.
`AllToAllMuon` therefore accepts `all_to_all_strategy` either as a direct
constructor argument or as a parameter-group value, requires all specified
values to agree, and defaults to `flat` only when no value is present. The
`layer_pipelined` strategy similarly reads `num_layers_per_bucket` from the
constructor or parameter group, requires one positive value across all groups,
and defaults to `1`.

For example, override the layer-pipelined benchmark config with:

```bash
--optimizer.param-groups.0.optimizer-kwargs.num-layers-per-bucket 4
```

## Performance Study

The corrected comparison used one node with 2 H100 GPUs, FSDP degree 2, local
batch size 1, sequence length 512, and seed 42 in the `tt12` environment. Each
strategy ran twice for 30 training steps. For each step, the measurement takes
the slower rank's structured span. Each run reports the median of steps 6-30,
and the table reports the mean of the two run medians. Performance runs did not
enable deterministic mode.

| Model | Strategy | All-to-alls/step | Optimizer | Full step | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: |
| 1B, 16 layers, 80 matrices | Flat | 2 | 49.2 ms | 220.3 ms | 16.7 GiB |
| 1B, 16 layers, 80 matrices | Shape grouped | 8 | 42.8 ms | 221.3 ms | 16.7 GiB |
| 1B, 16 layers, 80 matrices | Layer pipelined (`K=1`) | 32 | 51.2 ms | 224.9 ms | 16.6 GiB |
| 8B, 32 layers, 160 matrices | Flat | 2 | 692.2 ms | 1135.2 ms | 84.7 GiB |
| 8B, 32 layers, 160 matrices | Shape grouped | 8 | 645.0 ms | 1128.0 ms | 84.7 GiB |
| 8B, 32 layers, 160 matrices | Layer pipelined (`K=1`) | 64 | 733.3 ms | 1173.0 ms | 84.6 GiB |

Shape grouped reduced optimizer time relative to flat by 13.0% at 1B and 6.8%
at 8B. Its full-step result was a 0.5% regression at 1B, which is within run
noise, and a 0.6% improvement at 8B. On this topology, four regular shape
groups were better inside the optimizer despite requiring eight collectives
instead of two.

Layer pipelining with `K=1` did overlap communication and compute, but it did
not win. It increased optimizer time relative to flat by 3.9% at 1B and 5.9%
at 8B, and
increased full-step time by 2.1% and 3.3%. Gather overlap cannot hide the other
costs: there are `2 * num_layers` launches, every reverse scatter remains on the
critical path, and per-layer packing and synchronization are more frequent.

A PyTorch profiler trace of the 1B `K=1` layer-pipelined run contained 32
`c10d::alltoall_base_` calls, as expected for 16 layers. All 15 eligible
next-layer gathers overlapped current-layer GPU work and also overlapped
Newton-Schulz GEMM kernels; the summed direct gather/GEMM overlap was 1.25 ms in
the profiled step. This verifies that the negative timing result is not caused
by failure to create overlap.

### Layer-Bucket Sweep

A follow-up sweep tested `K = 1, 2, 4, 8, 16` for the 1B model on one reserved
two-H100 80 GB MAST host. Flat and all five bucket sizes ran twice for 30 steps
on the same GPU pair. The first repetition used increasing `K`; the second used
the reverse order. The measurement is again the mean of two run medians, where
each run takes the slower rank at every step and then the median of steps 6-30.

| Layout | Buckets | All-to-alls/step | Optimizer | vs. flat | Full step | vs. flat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat | 1 | 2 | 44.6 ms | - | 214.4 ms | - |
| Layer pipelined, `K=1` | 16 | 32 | 46.8 ms | +5.1% | 217.9 ms | +1.7% |
| Layer pipelined, `K=2` | 8 | 16 | 41.8 ms | -6.2% | 211.3 ms | -1.4% |
| Layer pipelined, `K=4` | 4 | 8 | 41.7 ms | -6.3% | 211.9 ms | -1.2% |
| Layer pipelined, `K=8` | 2 | 4 | 40.8 ms | -8.4% | 210.6 ms | -1.7% |
| Layer pipelined, `K=16` | 1 | 2 | 42.3 ms | -5.1% | 212.2 ms | -1.0% |

`K=8` was the best point in this experiment. It retained one opportunity to
overlap the second gather with first-bucket Newton-Schulz while reducing the
collective count from 32 at `K=1` to four. `K=2` and `K=4` were close, and all
three improved full-step time by only 1-2%, because the optimizer is a minority
of the complete training step.

The result is not explained by collective count alone. Each Llama 1B layer has
five selected Muon matrices. Resetting greedy ownership for every `K=1` bucket
repeats an asymmetric matrix-shape mix on the same ranks; its two owners receive
33,554,432 and 27,262,976 elements per layer. In the controlled runs, the two
rank-local optimizer medians were approximately 37.6 and 46.8 ms. Every
tested `K>=2` bucket balanced aggregate elements exactly and alternated the
matrix mix between owners. Coalescing therefore fixed both launch overhead and
repeated owner imbalance. This also shows that element count alone is an
imperfect Newton-Schulz load metric: the slower `K=1` owner had fewer elements
but three matrices, including the wide feed-forward projection, while the
other owner had two.

`K=16` is a useful flat-like control: it has one bucket, the same owner
assignment, and two collectives, but uses the asynchronous gather/wait code
path. Its small difference from flat should not be over-interpreted. An
independent cross-host sweep reproduced the important ordering: flat, `K=1`,
`K=2`, `K=4`, `K=8`, and `K=16` measured 43.3, 46.8, 40.8, 42.6, 40.1, and
41.0 ms in the optimizer, respectively.

All 12 controlled runs produced the same displayed loss and gradient norm at
every logged step. Exact parity is covered separately by the distributed unit
tests; performance runs did not enable deterministic mode. This bucket sweep
is 1B-only because the measured 8B two-rank footprint exceeds the 80 GB capacity
of the MAST host class. The eight-GPU study below includes both models.

Five deterministic debug-model steps were also run for every strategy with
TensorBoard enabled. Global average loss, global maximum loss, and gradient
norm matched bit-for-bit at every step. At step 5 the full-precision average
loss was `7.67299747467041` and gradient norm was `1.7009416818618774` for all
three.

### Eight-H100 Sweep

The comparison was repeated at FSDP degree 8 on reserved single-node H100
80 GB MAST hosts with PyTorch `2.14.0.dev20260626+cu130`. Both models used
local batch size 1, sequence length 512, seed 42, and 30 steps. The first order
was flat, shape grouped, and increasing `K`; the second order was reversed.

For every run, the statistic first takes the slowest of eight ranks for each
structured span and then takes the median of steps 6-30. The tables report the
median of those per-run medians. The 8B model ran twice on one host. The 1B
model ran two forward/reverse pairs on two hosts because the first pair showed
large run-level outliers. For example, its flat optimizer medians were 21.6,
22.9, 25.1, and 43.5 ms. Its shape-grouped range was 23.4-45.5 ms and its
`K=8` range was 22.2-42.6 ms. The four-run median prevents one slow run from
determining the result; differences around 1% remain below the resolution of
this experiment.

The 1B results were:

| Layout | All-to-alls/step | Optimizer | vs. flat | Full step | vs. flat | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat | 2 | 24.0 ms | - | 159.2 ms | - | 7.88 GiB |
| Shape grouped | 8 | 27.2 ms | +13.6% | 159.7 ms | +0.3% | 7.86 GiB |
| Layer pipelined, `K=1` | 32 | 32.5 ms | +35.6% | 175.4 ms | +10.2% | 8.11 GiB |
| Layer pipelined, `K=2` | 16 | 30.6 ms | +27.9% | 160.6 ms | +0.9% | 7.60 GiB |
| Layer pipelined, `K=4` | 8 | 25.1 ms | +4.9% | 158.0 ms | -0.8% | 7.80 GiB |
| Layer pipelined, `K=8` | 4 | 23.8 ms | -0.9% | 158.4 ms | -0.5% | 7.88 GiB |
| Layer pipelined, `K=16` | 2 | 24.8 ms | +3.5% | 159.6 ms | +0.3% | 7.88 GiB |

Flat and `K>=4` were effectively tied end to end. The apparent 0.8% full-step
advantage for `K=4` is smaller than the observed run variation. Shape grouping
did not improve the full step, and `K=1` was a clear regression.

The 8B repetitions were substantially more stable. Their per-run optimizer
medians differed by at most 5.1 ms, and full-step medians differed by at most
4.0 ms:

| Layout | All-to-alls/step | Optimizer | vs. flat | Full step | vs. flat | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat | 2 | 112.0 ms | - | 489.5 ms | - | 29.63 GiB |
| Shape grouped | 8 | 107.0 ms | -4.5% | 486.5 ms | -0.6% | 29.67 GiB |
| Layer pipelined, `K=1` | 64 | 247.2 ms | +120.7% | 671.2 ms | +37.1% | 32.25 GiB |
| Layer pipelined, `K=2` | 32 | 144.9 ms | +29.4% | 488.2 ms | -0.3% | 28.75 GiB |
| Layer pipelined, `K=4` | 16 | 126.1 ms | +12.6% | 490.6 ms | +0.2% | 29.42 GiB |
| Layer pipelined, `K=8` | 8 | 110.2 ms | -1.6% | 484.0 ms | -1.1% | 29.63 GiB |
| Layer pipelined, `K=16` | 4 | 101.2 ms | -9.6% | 485.3 ms | -0.9% | 29.63 GiB |
| Layer pipelined, `K=32` | 2 | 112.6 ms | +0.5% | 488.4 ms | -0.2% | 29.63 GiB |

`K=16` had the fastest optimizer span, while `K=8` had the fastest full step.
The full-step gains were only 0.9-1.1%, so neither is a decisive training
throughput win. `K=32` is the flat-like control: its two collectives, ownership,
and timing closely reproduced flat. Shape grouping saved 4.5% in the optimizer
but only 0.6% end to end. The optimizer span is diagnostic rather than
additive: asynchronous GPU work and collective synchronization can change
which span is charged, so the complete training step is the decision metric.

The eight-rank owner assignment explains the bucket-size trend. Each layer has
only five selected matrices, so a `K=1` bucket leaves three ranks without
Newton-Schulz work. Summing the maximum owned element count over all buckets
gives 2.207 times the flat critical-owner load at 1B and 2.154 times at 8B.
For `K=2` and `K=4`, all ranks are active but that ratio is still 1.103 at 1B
and 1.077 at 8B. `K>=8`, flat, and shape grouped balance element counts exactly
on these models. Matrix shape also affects Newton-Schulz time, so element count
is an explanatory approximation rather than a runtime model.

All layouts move the same unpadded payload in this experiment. Both models have
four unique selected matrix shapes, and each shape's matrix count divides
evenly across eight owners. Shape grouping therefore uses eight regular
all-to-alls without padding; flat uses two variable-split all-to-alls. The
result depends on whether regular-buffer efficiency repays six extra launches
on the target topology.

The steady-state explicit `DTensor.to_local()` count is also identical across
the layouts. Gradient validation calls it twice per matrix, and local momentum
updates call it once for the gradient and once for the momentum buffer. That is
320 calls per 1B step and 640 per 8B step, independent of `K`. These experiments
do not isolate the absolute cost of those calls, so they neither prove nor
disprove that `to_local()` is a major common bottleneck. They do show that it
cannot explain the differences between rows. Cached owner offsets and split
sizes remove per-step metadata construction, but do not change this call count
or the collective count.

`K=1` raised peak reserved memory by 0.23 GiB at 1B and 2.62 GiB at 8B relative
to flat because its per-bucket owner assignment repeatedly concentrates buffers
on the same five ranks. Every 8B layout remained below 33 GiB on a 79.18 GiB
H100, so memory was not the limiting factor in this study.

All 1B and 8B runs produced the same displayed loss and gradient norm at every
logged step. At step 30 they were `2.75907` and `2.0538` for 1B, and `2.73636`
and `1.6897` for 8B. Exact parity remains covered by the distributed unit tests;
the performance runs did not enable deterministic mode.

Example commands:

```bash
CUDA_VISIBLE_DEVICES=4,5 NGPU=2 CONFIG=llama3_1b_fsdp_muon_flat \
  ./run_train.sh --parallelism.data_parallel_shard_degree 2 \
  --training.steps 30 --metrics.log_freq 5 --debug.seed 42
CUDA_VISIBLE_DEVICES=4,5 NGPU=2 CONFIG=llama3_1b_fsdp_muon_shape_grouped \
  ./run_train.sh --parallelism.data_parallel_shard_degree 2 \
  --training.steps 30 --metrics.log_freq 5 --debug.seed 42
CUDA_VISIBLE_DEVICES=4,5 NGPU=2 CONFIG=llama3_1b_fsdp_muon_layer_pipelined \
  ./run_train.sh --parallelism.data_parallel_shard_degree 2 \
  --optimizer.param-groups.0.optimizer-kwargs.num-layers-per-bucket 8 \
  --training.steps 30 --metrics.log_freq 5 --debug.seed 42
NGPU=8 CONFIG=llama3_8b_fsdp_muon_layer_pipelined \
  ./run_train.sh --parallelism.data_parallel_shard_degree 8 \
  --optimizer.param-groups.0.optimizer-kwargs.num-layers-per-bucket 16 \
  --training.steps 30 --metrics.log_freq 5 --debug.seed 42
```

The overlap trace used:

```bash
CUDA_VISIBLE_DEVICES=4,5 NGPU=2 \
  CONFIG=llama3_1b_fsdp_muon_layer_pipelined ./run_train.sh \
  --parallelism.data_parallel_shard_degree 2 --training.steps 10 \
  --debug.seed 42 --dump_folder=/tmp/tt-muon-layer-profile-1b \
  --profiler.enable_profiling --profiler.profile_freq 10 \
  --profiler.profiler_warmup 1 --profiler.profiler_active 1
```

The default remains `flat`: it minimizes collective launches, supports
arbitrary mixed shapes without padding, and was within 1.1% of the best full
step in every eight-H100 table. `shape_grouped` remains useful when target
profiling shows that regular buffers repay the extra launches. Layer buckets
must be large enough to keep owner ranks busy; `K=1` should be avoided for these
models at FSDP degree 8. `K=8` and `K=16` are the measured coarse-bucket
candidates for this topology, but their end-to-end gains were about 1% or less.
Collective latency, bandwidth, matrix shapes, layer count, owner balance, and
FSDP degree all change the tradeoff, so the target topology should be measured.

## Checkpointing

No custom checkpoint code should be added.

`AllToAllMuon.state[param]["momentum_buffer"]` is a DTensor with the same
sharding as the FSDP parameter. The existing flat FQN-keyed optimizer helpers
can therefore save and load it alongside AdamW state.

Save and resume with the same FSDP world size has been smoke-tested.
Cross-world-size resume may work through DCP resharding, but it has not been
validated.

`init_optim_state()` calls one zero-gradient, zero-learning-rate distributed step
before a pre-training checkpoint load. The implementation supports this path,
and a two-GPU save/reload smoke test has completed successfully.

## Tests

### Unit Tests

`tests/unit_tests/test_muon.py` compares all three strategies against upstream
`torch.optim.Muon` on a world-size-one Gloo mesh and a world-size-two mixed-shape
mesh. It exercises irregular flat offsets, shape-group padding, deterministic
layer ordering and multi-layer buckets, multi-step momentum, size-balanced
ownership, parameter-group strategy resolution, collective counts, and
next-gather launch order.

### Two-GPU Integration Test

`tests/integration_tests/h100.py::{fsdp_muon,fsdp_muon_layer_pipelined}` runs
the Llama 3 debug model with two FSDP2 ranks for two deterministic steps. This
exercises 30 Muon matrices, weight decay, Nesterov, the AdamW catch-all group,
and the asynchronous NCCL path; the layer-pipelined case uses `K=2`. Flat,
shape grouped, and the original `K=1` layer-pipelined schedule were also run
manually for five deterministic steps and produced bit-for-bit identical
TensorBoard loss and gradient-norm series.

For a TorchTitan numerical run, use `--debug.seed=42` and
`--debug.deterministic`; never use `--debug.deterministic_warn_only`. Compare
full-precision loss and `grad_norm` using `scripts/loss_compare.py`.

## Change Set

1. Add `torchtitan/components/muon.py` with flat, shape-grouped, and
   layer-pipelined plans.
2. Register the `Muon` algorithm with the `AllToAllMuon` core lowering.
3. Add unit and two-GPU FSDP2 tests.
4. Add reproducible Llama 3 debug, 1B, and 8B configs.
5. Run `pre-commit run --all-files` and the relevant test targets.

Acceptance criteria:

- Real FSDP2 parameters and gradients are used end to end.
- Multi-step parameters and momentum match full-matrix upstream Muon.
- Existing mixed-optimizer scheduling and same-size checkpoint resume work.
- Unsupported layouts fail during initialization instead of running shard-local
  Muon silently.

## Deferred Work

Only add these after the minimal path is correct and profiling shows a need:

- Uneven shards and `Shard(1)`.
- Reverse-scatter overlap or double-buffered buffer reuse if profiling justifies
  the added scheduling complexity.
- A public `bucket_spec` integrated with FlexShard or upstream PyTorch.
- More exact owner bin packing if profiling shows greedy imbalance matters.
- Per-head or packed-expert lowering through nonzero `param_offset` specs.
- TP plus FSDP two-dimensional reconstruction.
- HSDP, CP, EP, PP, and CPU offload validation.
- Cross-world-size checkpoint tests.
- SOAP and fused-projection splitting.
