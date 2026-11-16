# LoopTitan architecture

This document describes the limits of the current model, recipes, training and delivery. [Recipes](recipes.md) records the selected path of each source paper. [Validation](validation.md) records the configurations that actually passed.

## 1. One product focus

**Open-source code for training and scaling recursive sparse models.**

A researcher reads one short recipe and sees how the model loops, which parameters it shares and how gradients flow. After a change to one mechanism, the same path does data and training, DP/FSDP scale-up, resume after an interruption, evaluation and model export.

#6 delivered the engineering qualification of four methods and a full pretraining comparison of ordinary MoE and SMELT. #7 added a random initial state, two memories and per-token expected CE. The four added paths completed 512 updates on real text, two-rank FSDP, full resume and standalone HF export. The full v2 stage acceptance passed. [Validation](validation.md) gives the exact configurations and scope. Claims about method quality and scaling need the matching experiments.

The core reading path is fixed:

```text
*_spec() → model construction → explicit forward → explicit objective
                           │
                 cost/comparison → TorchTitan training integration
                           │
                  checkpoint / eval / HF
```

The concept tree helps to understand the research. The code uses shallow composition. Paper names appear in recipes and source notes. An update equation that is really different gets a short dedicated `forward`. It does not have to become a switch of the base model.

## 2. The key design: physical blocks and calls are separate

A `Cell` is one Transformer/MoE block that holds parameters. A repeated call of the same cell is parameter sharing. When each visit needs its own attention, router or norm, the cell holds a list of these modules. The cell still has only one expert bank.

```text
LoopedLM
├── embedding
├── prefix: list of cells, runs once
├── core: list of cells, runs in a loop
│   └── Cell
│       ├── attentions       one set, or one set per visit
│       ├── attention_norms  one set, or one set per visit
│       ├── ffn_norms        one set, or one set per visit
│       ├── routers          one set, or one set per visit
│       ├── ffn_output_norm  separate output norm of MemoryCell
│       └── experts          one set of physical parameters
├── suffix: list of cells, runs once per readout
├── input_adapter           optional explicit input injection
├── final_norm
├── lm_head                 the only vocabulary projection
└── exit_gate               optional scalar gate shared across visits
```

The public name is `LoopedLM` (`from moeloop import LoopedLM`). `RecursiveLM` is a compatibility alias for the same class object. Old scripts run as before and get no warning. [`model.py`](../src/moeloop/model.py) defines the class as `LoopedLM`; `RecursiveLM` is only its alias.

The dense control replaces the sparse branch with a plain FFN. The always-on FFN is a separate, optional additive branch. Its name is not mixed with "expert bank shared across visits". In the first version, the embedding and the head are independent. There is no vocabulary weight alias across FSDP units.

In `ModelConfig` schema 4, `prefix/core/suffix: RegionConfig` overrides only the expert/FFN geometry, top-k, always-on FFN and residual scale of a region. All regions keep the common residual width and attention geometry. Equal configuration objects create independent modules. Sharing comes from repeated calls of the same physical owner. Old schemas 1/2/3 load with their original default behaviour and are then represented as schema 4. Added state, gate and injection options are off by default, and old weight owners do not change. `diag_decay`/`orthogonal` injection belongs to schema 4; a lower schema that declares them is rejected. A compatible configuration does not mean that a full training state from other source code can strict-resume. An unknown schema is rejected at load.

The two orders stay as two short loops:

```python
# stack: A → B → A → B
for visit in range(num_visits):
    for cell in self.core:
        state = cell(state, visit=visit)

# layer: A → A → B → B
for cell in self.core:
    for visit in range(num_visits):
        state = cell(state, visit=visit)
```

The code above leaves out masks, the cache and the returned statistics. It shows only the real execution order. The implementation passes these arguments explicitly. Do not first unroll an execution graph and then interpret the execution.

Each parameter has exactly one canonical owner. Initialization, the optimizer, checkpoints and export all work per physical owner. The state and the KV cache of each call work per logical position. **Shared parameters do not mean shared run state.**

This limit of expression is intentional. The first version gives no parameter links between arbitrary distant cells, no arbitrary sharing graph and no automatic partition planning. When a real need goes past this limit, a new explicit model solves it. The code does not add an owner-map language in advance.

## 3. How the named methods map to one code base

| Recipe | Construction and execution difference | Selected semantics |
|---|---|---|
| Ordinary dense / MoE | `num_visits=1` | Non-looped baseline; resource matching is a separate step |
| SMELT | The middle half of the physical layers runs twice | Both core residuals are multiplied by 1/2; the outer cells use 1 |
| Meta middle loop | Independent prefix/suffix, shared core stack | Fixed depth, full BPTT, terminal CE |
| Foil | core `(E,D,L)→(2E,D/2,2L)` | Base and Foil both have per-visit attention; router, norms and experts are shared within a bank |
| Tying | Consecutive groups use the layer loop and share only the expert bank | Attention, router and norm are independent; expert LR=`base_lr/sqrt(g)` |
| Huginn-inspired MoE | Live anchor, independent random latent, depth sampled per microbatch | Backpropagation through the last `min(k,r)` visits; fixed-depth zero-init evaluation is an explicit adaptation |
| LOOM full-BPTT | `MemoryCell`, global/local normalized EMA, per-visit router | The global memory resets at each forward, the local memory at each visit; memory gradients are kept |
| Ouro Stage I | Shared gate, readout at each visit, per-token expected CE−entropy | Every visit runs; HF and held-out evaluation use terminal logits |
| Sandwich (Looped Models Part I) | An independent prefix and suffix surround a shared core stack; before each core visit, `diag_decay` injects the prefix output | z_0 = e, fixed depth, full BPTT, terminal CE; an MoE path and a dense path |
| Relaxed Recursive | One shared block of layers loops, with no prefix/suffix; each core cell adds one LoRA per visit on the shared weights (`lora_attention_rank` / `lora_ffn_rank`) | Starts at `B = 0`, so the initial model equals the plain looped model; fixed depth, full BPTT, terminal CE; the MoE path relaxes only attention, `dense=True` also relaxes the FFN |
| LatentLoop residual | SMELT's split; the middle half runs twice with every core owner shared | The core attention and FFN residuals use 0.25 at the first visit and 1 at the second (`visit_residual_scales`); fixed depth, full BPTT, terminal CE |

[Recipes](recipes.md) gives the source paths of the ten named recipes in the table above, the common routing/optimizer/data replacements and the difference from Foil Base. `per_visit_router` and `input_anchor` stay components; they are not complete methods. From this version on, `loom()` means the complete memory recipe. Users of the old component use `per_visit_router()`. Bank size, initialization, normalization, routing, loss and data are each declared.

A structural recipe is not a fair control by itself. `num_visits=1` says only that the execution does not loop. It does not prove that the parameter count or the compute is matched. A specific comparison writes in a plain configuration the number of physical cells and banks, the number of calls, the bank and FFN widths and the training budget. This does not create a new platform for experiment comparison.

The P1 ordinary MoE has 10 independent cells, d672, expert hidden544, E64/top8, heads7/KV1. SMELT has 10 physical cells (2 prefix, 5 core that run twice, 3 suffix), d576, E96/top8, heads9/KV1, and 15 logical calls in total. The trained SMELT hidden440 comes from document-packing statistics of the full 20.486B-label training data. The final data, evaluation, source code and pairing are frozen. Each arm completed 19,537 updates. The exact [recipe JSON](recipes.md#p1-ordinary-moe--smelt) and the [P1 evidence](evidence/p1.json) keep the source and results of this run. Both arms share the 151669-token vocabulary and 4096ctx. "100M active" means the single-traversal active body of the ordinary baseline, about 98.5M. With the independent embedding and head, the total model is about 916.5M. [Cost and comparison](comparison.md) describes the shape selection and the document-mask budget.

A new mechanism enters in one of three ways:

1. **A new combination of existing parts**: add a plain recipe function.
2. **A local operator change**: add a small module. Keep the calls and the parameter ownership explicit.
3. **A change of the state update itself**: write a short dedicated Cell or model forward. Reuse the attention, experts, training and delivery parts.

For example, CoE runs attention once, the FFN norm once and an expert chain many times. This must not pose as a repeated plain block. The inner count (`ffn_steps` in [`recipes/ffn_chain.py`](../recipes/ffn_chain.py)) and the outer `num_visits` have separate names. A fixed expert chain can reuse full BPTT and terminal CE. It does not automatically inherit the meaning of the stack tail gradient. When a new forward changes the parameter layout, the number of cache calls or the state structure, add a short matching parallelize/export integration. There is no promise that any model gets all delivery functions with no adaptation. A multi-state method is not forced into a single-tensor state. A small local repetition is acceptable. Extract the shared part when it is stable in practice.

## 4. The core model stays plain PyTorch

The math of the base cell reads directly:

```python
x = x + residual_scale * attention(attention_norm(x), positions, mask, cache_slot)
z = ffn_norm(x)
ids, weights, aux = router(z, valid_inputs)
update = experts(z, ids, weights)
return x + residual_scale * update, aux
```

The exact signatures can change during coding. The semantics below stay stable:

- attention: GQA, RoPE, pre-norm. The configuration holds the head counts and the position and normalization rules. There is one explicit set of defaults. A recipe selects small variants such as QK-norm explicitly.
- sparse: the first version defaults to softmax top-k, normalization of the selected weights, dropless dispatch and an optional always-on FFN. Selection IDs, combine weights and valid-token statistics are separate responsibilities. Dispatch counts and aux counts after padding removal are not mixed.
- stage input injection: the default is no injection. The options are `add`, `linear` (two linear projections `W_h h + W_e anchor`), and the two operators of xllm-loop Part II: `diag_decay` (`exp(-dt·a)·h + dt·W e`, with per-channel `dt` and `a`) and `orthogonal` (OrthoInj: project the decayed `h` onto the subspace orthogonal to `q = dt·W e`, then add `q`). The docstring of [`InputAdapter`](../src/moeloop/model.py) gives the formulas and sources. The injection applies before each complete stack visit, the first visit included. The initial state is explicitly input, zero or an independent random latent. The norm/residual order is not hidden in a general boundary layer. The last two operators start from `W = I`, `a = 1`, `dt = ln(5)/2` (the `ModelConfig` defaults; a recipe can change them, and sandwich sets `dt = 1`). In the reference model, `LoopedLM.initialize_weights` initializes to the declared values. On the TorchTitan path, `TitanModel.init_states` calls it directly.
- per-visit LoRA (Relaxed Recursive): at each visit, a core cell has one `B A x` on each shared attention projection (`q/k/v/out`) and on the dense or always-on FFN (`w1/w3/w2`). There is no alpha scaling. `lora_attention_rank` and `lora_ffn_rank` set the ranks. Routed experts, routers and norms get no LoRA. A LoRA is a per-visit owner; like the per-visit modules, it is allocated by `visit_capacity`. In the resolved prefix and suffix, the ranks are 0. `A` (normal) and `B = 0` are initialized after all other parameters, so with the same seed the initial model equals the shared-weight model. On the TorchTitan path, `TitanModel.init_states` calls the same function. A third-party `attention_factory` gets `lora=` only when attention LoRA is on; [Extensions](extensions.md#replace-attention-and-per-visit-lora) gives the contract.
- default depth: it is the execution default. The capacity of the per-visit modules sets the weight shapes. A fully shared structure does not rebuild its weights when the execution depth changes. A per-visit module must not go out of bounds. Cached generation must not change depth midway.
- forward: it does not choose the training depth, does not call the optimizer and does not update persistent router counts. The random latent is drawn once, before the recurrent calls and before activation-checkpoint (AC) replay. The forward returns a differentiable aux sum/count and detached diagnostic statistics. A plain public call returns terminal logits. Expected-CE training explicitly returns the hidden state and gate of each visit. It does not keep the large vocabulary logits of each visit in memory.
- KV cache: training does not use the KV cache. At inference, each `(prefix/core/suffix, cell, visit)` has its own logical KV slot. The slots do not merge, even when the attention weights are shared.

Do not turn each small operation into a public plugin protocol. The sparse routing of the first release is limited to the paths above. Bias balancing, group routing, capacity dropping and similar features get an explicit implementation and a matching objective definition when a need appears.

**The loop body and the vocabulary head must be separable.** Large-scale training must not be forced to materialize full `[B,S,V]` logits. The public forward and HF return logits as usual. TorchTitan (TT) training calls the same whole module, gets the final hidden state and the aux, and passes them to the native chunked head/CE. The head stays a unique parameter owner with a normal FSDP call boundary. The aux is added once, outside the complete microbatch, not once per logits chunk. This is a limited training seam; it needs no separate loss/backend framework. See the [native ChunkedLossWrapper](https://github.com/pytorch/torchtitan/blob/086bf6c166ec85c1298eb5596fa9bf95f6a2d840/torchtitan/components/loss.py#L567).

`MemoryCell.forward` writes the different state update directly. First, the scaled attention output is written into the two normalized EMAs. Then `h_tilde=h+r_global+r_local`. Last, the scaled, output-normalized FFN is added. The attention is not added one more time, and the two memories are not averaged. The carry is a temporary tensor/scalar passed explicitly between functions. It is not a module buffer.

The Ouro readout at each visit uses the same suffix, final norm and head. The readout does not write back into the loop state. The last visit takes the remaining probability mass. The training entry uses the plain head call and the chunked recomputation of `expected_lm_loss`. It keeps both gradients, `p*dCE` and `CE*dp`. Do not multiply the scalar from the old chunked CE by a probability, because the old bridge ignores the outer `grad_output`.

## 5. Training dynamics must be visible

The default training stays fixed depth, full BPTT, CE on the final output and AdamW. The `optimizer` field of a recipe can change the optimizer to Muon (see "Optimizer groups" below). `TrainingRecipe` selects the training rules explicitly:

- **Training depth sampling**: one sample per optimizer update from a finite set of integers. All DP ranks and accumulation microbatches use the same depth. The seed and the number of completed updates set it. Fixed depth is always the plain path.
- **Tail gradient**: the first `D-K` complete visits of the stack loop run inside `no_grad`; backpropagation goes through the last K visits. If later injection uses the anchor, its gradient path stays. With no later injection, the truncation cuts the path from the final CE to the prefix and the embedding. No straight-through gradient is added silently.
- **Random initial state and depth**: the Huginn-inspired path uses a shifted Poisson-lognormal depth per microbatch (default mean 4, log std 0.5, no hard cap). All ranks are synchronized. Backpropagation goes through the last `min(k,r)` visits. The independent truncated-normal initial state is separate from weight initialization. Evaluation explicitly uses fixed depth and zero init.
- **Expected CE**: the CE of each token at each visit is weighted by a differentiable exit probability. The exit entropy is subtracted, and the separate aux is added. Training runs all visits. There is no claim of early-exit speed-up.

The layer loop uses full BPTT. Update-level sampling from a finite set needs `K <= min(depths)`. The Poisson-lognormal path uses `min(k,r)` by its declared definition. Truncation per cell and tail truncation of the complete stack are different algorithms.

To limit the run combinations of the first version, the supported set is:

| Training structure | First-version support target |
|---|---|
| Ordinary dense/MoE, fully shared stack loop | Fixed depth, full BPTT |
| Fully shared stack loop | Optional update-level depth sampling, tail gradient |
| Per-visit attention/router/norm, or per-visit LoRA | Fixed depth, full BPTT |
| Layer loop / consecutive expert tying | Fixed depth, full BPTT |
| Live-anchor random latent | Microbatch-level Poisson-lognormal, tail gradient |
| Normalized EMA state | Fixed depth, full BPTT; the explicit combination with per-visit attention is supported |
| Exit distribution | Fixed depth, full BPTT, per-token expected CE/entropy |

Per-visit owners with random depth are mathematically valid. The first version does not add unused parameters, changing call traces and truncation to the qualified scope at the same time. The capacity semantics of the core stay clear. Real later needs drive wider support. "Not yet qualified" does not mean that the method is invalid or that PyTorch cannot support it.

The loss is one explicit function:

- The main CE is normalized by **the global number of valid labels in the whole optimizer update**.
- The default MoE aux is first averaged over **the differentiable router calls of one microbatch**, then averaged over the logical microbatches of the whole update (global batch ÷ local batch). The coefficient does not grow automatically with the number of visits. It is also not equal to a global pooled routing loss.
- The input/router mask and the label/loss mask are separate. A local call with zero valid inputs adds zero aux. An update with no valid label in the global batch is rejected.
- The expert use of the leading no-grad calls can be logged, but it does not enter the denominator of the differentiable aux. Model recomputation does not update persistent statistics. The trainer sums them once, at the real call boundary.

This choice keeps the reduction scope readable. It also makes one limit of fair comparison explicit: a change of the microbatch geometry can change the non-additive aux, even with the same global batch. The aux does not depend on the GPU count when three conditions hold. First, the global batch and the local batch are both fixed. Second, the training depth rule does not depend on the GPU count (fixed depth or `uniform_update_v1`; `poisson_lognormal_micro_v1` is excluded, because its depth sequence depends on the accumulation steps). Third, the initial state is not drawn per rank (a random-latent initial state uses the RNG of each rank; today only `huginn_spec` does this, and its depth rule already excludes it). The data stream also does not depend on the GPU count (see section 7). #49 verifies this on GPU. Recording one balance coefficient is not enough to claim that the training objectives are fully aligned.

Undefined combinations are rejected explicitly, for example the Huginn tail gradient with the all-visit expected loss, or EMA with an optimizer step between segments. Ouro Stage II, token early exit and other multi-state mechanisms stay separate later extensions. No general engine is built in advance.

**Optimizer groups follow the physical owners.** In [`optim.py`](../src/moeloop/optim.py), `optimizer_config` maps the `optimizer` of a recipe to native param groups. Each physical parameter is in exactly one group. The gradient of a shared owner is already the sum over its calls, so it gets one update per step. AdamW has two groups: the declared zero-decay roles and all other parameters. With Muon, one rule, `MUON_PATTERN`, decides which parameters go to the TorchTitan DistMuon (in short: the hidden matrices, see `MUON_PATTERN`; AdamW still updates the embedding, lm_head, routers, the exit gate, norms and 1D parameters). `optim.MUON_PATTERN` is the exact rule. Each expert matrix is orthogonalized separately. Core experts carry `expert_lr_multiplier`, so they use a second DistMuon instance; weight decay is not compensated. When `output_lr_scale` is set, the core output projections get one more group: a third AdamW group, or a third DistMuon instance under Muon. A new matrix owner must be added to `MUON_PATTERN`. `tests/training/test_muon.py` checks this rule against every recipe constructor.

Limits of Muon: DistMuon accepts only DTensors on CUDA, so a Muon recipe trains only on the native FSDP GPU path. Each Muon parameter needs a gradient at every step; an owner that is not called raises an error. CPU tests run one real step on one rank only. #49 completed Muon training, full checkpoint saving and a WSD fork on 4-GPU and 8-GPU FSDP. That run did not cover HSDP `dp_replicate>1`, the sharding boundary where the world size is larger than the expert count, experts with no tokens, or resuming Muon from a checkpoint.

The cost model separates unique parameters, real calls and the logical KV cache. Fixed expected CE uses `fixed_expected_matrix_v1` (the document version when document pairs are given). It counts the head/coda at each visit and the head checkpoint recomputation. Sampled depth and tail gradients use `sampled_depth_matrix_v1` (the document version when document pairs are given). Its training matrix FLOPs are the expectation under the declared depth distribution. `flops.training_depth` also gives the cost of one pass at the p50/p95 depths. These are not the real depths or the measured work of a run. The nominal breakdown at the default depth is only one fixed forward. See [Cost and comparison](comparison.md).

## 6. TorchTitan integration: keep native training, collect the necessary differences

The first version keeps the checked TorchTitan `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`. The `train` extra now installs `torchtitan==0.3.0` from PyPI, which is tag v0.3.0 of that commit. A change of the upstream version can be a separate later task. The project does not follow new training-engine APIs at the same time.

| TorchTitan keeps | This project owns explicitly |
|---|---|
| Distributed environment, DP/FSDP, gradient accumulation | LoopedLM construction, repeated calls and the gradient window |
| Clipping, optimizer (DistMuon included), scheduler, training loop | Pure routing, loss/aux definitions, optimizer groups by physical owner, recipes with clear sources |
| Checkpoint scheduling and DCP saving | Complete data/RNG/real-progress state, strict resume contract |
| Profiling, native logs and metric output | Necessary depth/routing/valid-token diagnostics |

`titan.py` gives one ModelSpec, thin BaseModel/Trainer/Validator integrations and a limited parallelize function. The overridden hooks are limited to real, located differences: meta/materialize/init, passing the D/K and the masks of the current batch, the chunked head/CE with aux/metrics, and the completed progress and resume state. A short, clear local override is easier to maintain than layers of callbacks. Each override is reviewed against the upstream pin.

The first version explicitly selects `spmd_backend="partial_dtensor"`, with TP/EP/CP/PP all 1. Activations stay plain tensors; parameters use native DP/FSDP. The plain PyTorch core does not become a full tree of SPMD-annotated types. DP also uses the native Replicate/FSDP reduction contract of that pin. An arbitrary stock DDP is not a drop-in replacement without a change of the loss scaling.

The expensive expert execution reuses the selected native `RoutedExperts`. Our cell, routing and objective semantics do not change. The plain PyTorch reference implementation serves CPU, HF and semantic comparison. Between the two execution paths there is only a limited weight mapping and a numeric contract. There is no platform of freely replaceable backends.

The source code supports this seam. The native `MoE.forward` updates token counters during training, even with bias balancing off. A direct reuse of the whole MoE adds activation-checkpoint recomputation into the statistics. The native Trainer also drops the metrics that the loss returns. So a replacement of the loss class alone does not give a complete integration. See the [native MoE](https://github.com/pytorch/torchtitan/blob/086bf6c166ec85c1298eb5596fa9bf95f6a2d840/torchtitan/models/common/moe.py) and the [native Trainer](https://github.com/pytorch/torchtitan/blob/086bf6c166ec85c1298eb5596fa9bf95f6a2d840/torchtitan/trainer.py).

FSDP units follow the explicit parameter owners. A fully shared cell is a natural boundary. When per-visit attention exists, these submodules form separate units, so a call does not gather the attention of all visits. The vocabulary head also has its own call boundary for the chunked loss. Only the seed sets the initialization. `TitanModel.init_states` calls `LoopedLM.initialize_weights`. A CPU generator seeded with the run seed (`debug.seed`) draws each parameter as a whole, in canonical order. Each rank writes only its own shard, with no communication, and materializes one complete parameter at a time. So the weights at step 0 do not depend on the GPU count or the GPU model. The exact interplay of wrapping and activation checkpointing must be reviewed in the implementation and qualified in a common batch.

The first version targets DP/FSDP. The expert execution boundary, the canonical parameter identities and the local parallelize leave reasonable entry points for later EP/TP. Today there is no general parallelism planner, and there is no claim of support for axes that are not implemented.

## 7. Data, resume and export are the product core

**Data**: a plain configuration states the tokenizer, immutable sources, source weights, packing, positions and the document mask. Each source first forms blocks of equal length; the weights then select among the sources. All ranks pack the same logical block stream. It depends only on the data, the seed, the weights and `seq_len`. Each rank walks the whole stream and reads tokens only for its own blocks. The global batch of blocks of one update is cut into contiguous rank slices (rank r takes blocks `[r·K, (r+1)·K)`, where K is the number of blocks per rank). So the data and its order in each update do not depend on the GPU count. Native validation also counts in global batches. Each validation reads a fixed number of global batches (default 2; with `validation_label_tokens` set, rounded up to complete global batches) and cuts them by the same rule. The step count per rank is that number times the gradient accumulation steps. So any GPU count validates the same blocks. `PackedDataset` in [`data.py`](../src/moeloop/data.py) implements this. The public P1 DCLM path first freezes the revision, the ordered paths and the size/SHA of the raw Parquet files. It then prepares the default 24B train, 10M validation and 10M test valid labels. The completed manifest proves the real supply. The historical pre-tokenized DCLM fixture of 300M/1M/1M stays as an engineering smoke test. Its parent manifest/chunk SHA identifies it. It keeps the original EOS and prepends the BOS of this library. It does not cover the P1 endpoint. WikiText2 is the installation demo. FineWeb-Edu is an explicit optional path. [Training and data](training.md) gives the preparation commands, the tokenizer adaptation and the licences.

**Resume**: at an optimizer-update boundary, the checkpoint saves the model, the optimizer, the scheduler, the RNG of each rank, the data cursor and the partial packing buffer, the depth sampling rule and the counts of completed updates and tokens. In the first version, strict resume needs the same topology and data identity. The logical stream itself does not depend on the GPU count. But the data cursor records the rank, the world size and the blocks per rank, so a different GPU count is rejected. The evaluation RNG is isolated from training. Only a durable checkpoint can be a resume point. A change of topology or data is an explicit new branch. It does not pretend that the original run continues.

**Evaluation**: held-out evaluation gives CE and byte-exact BPB ([`evaluate.py`](../src/moeloop/evaluate.py)). The BPB byte table comes from the tokenizer assets of the manifest ([`tokens.py`](../src/moeloop/tokens.py)). Special tokens leave both the numerator and the denominator. When the manifest has no tokenizer assets, or the tokenizer is not a byte-level BPE, BPB is empty and `bpb_unavailable` states the reason. The code does not guess byte counts. [`core.py`](../src/moeloop/core.py) runs CORE (the DCLM v2 definition: the mean centered accuracy over 22 tasks) on a local HF export directory. lm-evaluation-harness stays the ready-made downstream tool. The record holds the model depth, the tokenizer, the task versions and the masks. The project reuses the existing evaluation ecosystem and builds no benchmark platform. [Training and data](training.md) gives the usage.

**HF export**: the export writes the physical weights, config, tokenizer, generation config, the necessary source code and the source metadata. A light wrapper uses the same core. An offline process with only PyTorch/Transformers loads it through AutoClass, with no TorchTitan. The full-prefix forward, fixed-depth generation and the verified cache path each declare their own qualification. A pass of an old model does not extend to a new recipe. Loading the custom source code needs `trust_remote_code=True`. The cache belongs to logical calls and fixes the session depth. An approximate KV method gets a separate name. See the [Transformers custom model mechanism](https://huggingface.co/docs/transformers/en/custom_models).

The core knows only the concrete `DecodeCache` that stores tensors. The HF Cache/Generation types stay in the wrapper. The wrapper creates a cache that matches the number of logical calls of this run, so HF does not allocate by the number of physical layers. It only converts the interface of the same KV. It does not copy the inference model or the KV storage.

The first delivery path of Huginn-inspired and LOOM is fixed depth with no cache. Huginn evaluation uses a deterministic zero initial state. The HF `labels` of Ouro still mean the standard next-token terminal CE, not the Stage I training objective. The validation page is the reference for the real native, resume and HF qualification of the added paths.

A few plain files record the training facts: the resolved config, the metrics and the outcome. CVSM/MLflow read them when needed. The public model and training need no GitHub issue, MLflow service or CVSM state machine.

## 8. Flat files, local reading

```text
src/moeloop/
  recipes.py        method *_spec functions and limited native trainer factories
  recipe_spec.py    plain serializable method/training/source record; does not import TT
  costs.py          owners, matrix FLOPs (with the expectation under sampled depth), logical KV estimate
  comparison.py     explicit comparison conditions, differences and limited pairing
  model.py          LoopedLM (`RecursiveLM` is its alias), Cell, attention, explicit recurrence
  moe.py            pure router, reference experts, clear routing statistics
  training.py       depth selection, gradient rules, objective
  latent.py         independent latent initial state, depth sampling located per microbatch
  memory.py         normalized EMA and scale factors, with no persistent buffer
  halting.py        exit probabilities, chunked expected CE/entropy
  titan.py          version-pinned TT lifecycle, parallelize, native expert binding
  data.py           source/packing/mixture/stateful data interface
  prepare.py        explicit public/local data preparation that never overwrites
  train.py          thin training entry
  hf.py             HF config/model/cache integration
  export.py         native checkpoint → portable HF
  evaluate.py       held-out CE/BPB and integration with existing evaluation tools
  tokens.py         per-token byte length of byte-level BPE
  core.py           CORE (DCLM v2 definition) evaluation
  optim.py          Muon/AdamW parameter groups by physical owner
  diagnostics.py    per-visit logit lens, residuals, routing overlap, effective depth
  testing.py        conformance check check_recipe for third-party recipes
  bench.py          tracks and submission checker of Looped MoE Bench
recipes/
  *.json, */*.json  frozen RecipeSpec JSON of the reported runs, by family
  ffn_chain.py      the extension example of a different state update
eval/bench/
  tracks.json       frozen tracks of the Bench (see the comparison document)
src/moeloop_lab/
  protocol.py       optional internal experiment protocol, separate from the public import path
```

This tree shows responsibilities; it does not fix the exact file count. If attention or the cache grows clearly, it can move to its own file. Do not pack several responsibilities into one huge file for the sake of a "flat" layout. Configuration uses plain dataclasses plus native TT Python config. There is no YAML/Hydra DSL, no automatic plugin discovery and no global configuration registry. The module path of a function is enough as a recipe entry point.

## 9. Implementation limits

The current implementation keeps these limits:

1. A small core, explicit loops, nested cell parameter ownership, paper recipes.
2. Each named method states a specific source path and its public adaptations. The code interface and the real GPU qualification are separate. Old frozen experiments keep their original identity.
3. A complete training path on native TorchTitan DP/FSDP, with strict resume and standalone HF delivery.
4. One repository; the public `moeloop` and the optional `moeloop_lab.protocol` are separate.

Local implementation, file splits and optimization serve the real method semantics. A new parameter owner, run state, gradient or objective must pass the matching semantic checks and real training/resume/export verification. A configuration that can be built does not replace run qualification.

[Development](development.md) describes the current implementation and how to contribute. [The FFN-chain extension](extensions.md) is an example of a different state update.
