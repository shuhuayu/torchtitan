# Lightweight FSDP2 Muon Plan

Status: implemented on the `fsdpmuon` branch

This plan enables exact full-matrix Muon for TorchTitan's pure FSDP2 path. It is
deliberately small: one component module, one optimizer registration, and focused
tests. It does not attempt to implement the complete Canzona paper or support
TorchTitan's full parallelism matrix.

## Scope

The first version supports:

- `torch.distributed.fsdp.fully_shard` parameters and gradients.
- A single one-dimensional FSDP shard mesh.
- Two-dimensional weights sharded uniformly with `Shard(0)`.
- Muon for selected matrix weights and AdamW for all remaining parameters.
- Flat and shape-grouped owner redistribution with cached communication buffers.
- The existing TorchTitan LR scheduler and distributed checkpoint path.

The first version does not support or validate:

- TP, HSDP, EP, and multi-axis DTensor placements.
- `Shard(1)`, uneven shards, and non-2D Muon parameters.
- CPU offload, optimizer compilation, and CUDA graph capture.
- Missing gradients for selected Muon parameters.
- SOAP, fused projection splitting, per-head lowering, and communication overlap.

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
in that group have equal size.

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

Both implemented strategies follow the same logical transition:

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

Both plans, all offsets, and all communication buffers are built once during
optimizer construction and reused every step.

| Property | Flat | Shape grouped |
| --- | --- | --- |
| Collective calls per step | 2 | `2 * num_shape_groups` |
| Split sizes | Variable | Equal |
| Padding | None | Dummy matrix slots |
| Owner layout | Mixed shapes, irregular offsets | One shape, regular slots |
| Expected strength | Lower launch overhead | Simpler, more regular copies |

Compared with an all-gather baseline, either owner strategy keeps one full copy
and one Newton-Schulz computation per matrix instead of one per rank.

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

1. Read canonical FQNs from each param group's `param_names`.
2. Build one `MuonMatrixSpec` per selected parameter.
3. Require every parameter to be a 2D DTensor.
4. Require a one-dimensional device mesh and placements equal to `(Shard(0),)`.
5. Require `full_rows == local_rows * fsdp_world_size` and matching columns.
6. Require every selected parameter to use the same FSDP process group and dtype.
7. Assign each logical matrix to a mesh-local owner with deterministic greedy
   load balancing.
8. Build either one flat plan or one regular padded plan per shape, including
   cached offsets, split sizes, and communication buffers.
9. All-gather a hash of the strategy, FQNs, shapes, offsets, owners, and order;
   fail if ranks disagree.

## Optimizer Step

Each optimizer step has four phases:

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
shape-grouped plan exchanges a regular padded buffer once per shape:

```text
rank 0 sends P0 shard -> owner 0, P1 shard -> owner 1, ...
rank 1 sends P0 shard -> owner 0, P1 shard -> owner 1, ...
```

Each owner uses cached offsets or slots to reconstruct every full matrix
assigned to it. There is no per-step sorting, plan construction, or allocation.

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

`step()` must finish both collectives and all local updates before returning.
There is no asynchronous overlap in this version.

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

Use `llama3_debugmodel_fsdp_muon_shape_grouped` to exercise the second layout.
The `llama3_1b_fsdp_muon_{flat,shape_grouped}` and
`llama3_8b_fsdp_muon_{flat,shape_grouped}` configs provide the same comparison
on larger models.

## Performance Study

The two strategies were compared on one node with 8 H100 GPUs, FSDP degree 8,
local batch size 1, sequence length 512, and seed 42 in the `tt12` environment.
Each final run used 30 training steps. The table reports the median of steps
6-30, taking the slowest rank's structured `step_end` span for each step.

| Llama 3 config | Muon matrices | Shape groups | Flat | Shape grouped | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1B | 80 | 4 | 184.9 ms | 189.5 ms | 7.86 GiB |
| 8B | 160 | 4 | 549.1 ms | 546.1 ms | 29.63 GiB |

Flat was 2.4% faster at 1B, where avoiding six extra collective launches per
step matters. Shape grouped was 0.5% faster at 8B, where its regular layout can
offset those launches. Repeated 12-step runs showed the same broad result but
enough machine noise that the small 8B difference should be treated as a tie,
not a general claim that one layout always wins.

Both strategies used the same peak memory and produced matching loss and
gradient-norm trajectories. At step 30 the reported values were loss 2.75907
and gradient norm 2.0538 for 1B, and loss 2.73618 and gradient norm 1.6651 for
8B.

Example commands:

```bash
NGPU=8 CONFIG=llama3_1b_fsdp_muon_flat ./run_train.sh \
  --training.steps 30 --metrics.log_freq 5 --debug.seed 42
NGPU=8 CONFIG=llama3_1b_fsdp_muon_shape_grouped ./run_train.sh \
  --training.steps 30 --metrics.log_freq 5 --debug.seed 42
```

The default remains `flat`: it minimizes collective launches and supports
arbitrary mixed shapes without padding. `shape_grouped` is available when
profiling on the target model and topology shows that regular buffers are more
important than launch count.

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

`tests/unit_tests/test_muon.py` compares both strategies against upstream
`torch.optim.Muon` on a world-size-one Gloo mesh and a world-size-two mixed-shape
mesh. It exercises irregular flat offsets, shape-group padding, multi-step
momentum, deterministic size-balanced ownership, and collective counts.

### Two-GPU Integration Test

`tests/integration_tests/h100.py::fsdp_muon` runs the Llama 3 debug model with
two FSDP2 ranks for two deterministic steps. This exercises 30 Muon matrices,
weight decay, Nesterov, and the AdamW catch-all group. Both layouts were also
run manually with this deterministic setup and produced identical loss and
gradient norm.

For a TorchTitan numerical run, use `--debug.seed=42` and
`--debug.deterministic`; never use `--debug.deterministic_warn_only`. Compare
full-precision loss and `grad_norm` using `scripts/loss_compare.py`.

## Change Set

1. Add `torchtitan/components/muon.py` with flat and shape-grouped plans.
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
- Gather/compute/scatter overlap.
- More exact owner bin packing if profiling shows greedy imbalance matters.
- Per-head or packed-expert lowering through nonzero `param_offset` specs.
- TP plus FSDP two-dimensional reconstruction.
- HSDP, CP, EP, PP, and CPU offload validation.
- Cross-world-size checkpoint tests.
- SOAP and fused-projection splitting.
