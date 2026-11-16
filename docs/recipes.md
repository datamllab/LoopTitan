# Method recipes and where to start reading

`smelt_spec()` and the other functions of [`moeloop.recipes`](../src/moeloop/recipes.py) return a plain [`RecipeSpec`](../src/moeloop/recipe_spec.py) and do not import TorchTitan. Only the public `trainer_config(spec)`, or a built-in factory (the function name without `_spec`), builds a native trainer configuration. Read in this order: spec → [`LoopedLM.forward` / `Cell.forward`](../src/moeloop/model.py) → [`training.objective`](../src/moeloop/training.py). `LoopedLM` is the public name. `RecursiveLM` is a compatibility alias for the same class object (no warning). `model.py` defines the class as `LoopedLM`.

```python
import json
from moeloop.recipes import foil_spec
from moeloop.recipe_spec import RecipeSpec

spec = foil_spec(num_experts=4, core_layers=2, passes=2,
                 vocab_size=128, dim=32, hidden_dim=64,
                 num_heads=4, num_kv_heads=2, max_seq_len=64)
restored = RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
print(restored.source["selected_E_D_L"])  # [8, 1, 4]
print(restored.model.core.num_experts)  # 8
```

In `RecipeSpec` schema 1, `model` describes the parameters and the execution structure, and `training` describes the objective, gradients and depth. `source`, `preserved` and `adaptations` describe the source paper and the adaptations. `optimizer`, `inference` and `run` record the run defaults. `evidence` is for the identities of real artifacts. Neither an empty `evidence` nor `run.successful_updates=512` is evidence of completion.

## Ten selected paths

The first four have a historical engineering qualification. SMELT also has the full pretraining comparison below. The added Huginn-inspired, LOOM, Ouro and LOOM×attention paths each passed 512 updates of two-GPU FSDP, their own full resume and standalone HF. [Validation](validation.md) and the [machine-readable evidence](evidence/recipes-v2.json) give the engineering traces and scope. Sandwich, relaxed and LatentLoop residual have no engineering qualification record of this kind. They pass the recipe and conformance tests on CPU. Run issues #232 (sandwich, relaxed) and #254 (LatentLoop residual) trained them on GPU with AdamW and with Muon, exported them to HF and evaluated them.

| Method and entry | Preserved semantics | Adaptations to the common stack |
|---|---|---|
| [SMELT v3 §3/Appendix A](https://arxiv.org/html/2609.01343v3), `smelt_spec(physical_layers=8)` | prefix/core/suffix physical layer counts 2/4/2; the core runs twice in full; the core attention and FFN residuals are each multiplied by 1/2, the outer cells use 1 | Public text, common sparse cells and AdamW; the budget pairing with an ordinary MoE is defined separately |
| [Meta Looped MoE v1 C.1–C.2](https://arxiv.org/html/2609.40316v1), `meta_spec(core_layers=2, passes=4)` | Independent prefix/suffix, a shared loop in the middle; fixed depth, full BPTT, terminal CE | Common softmax top-k, normalization of the selected weights, dropless dispatch, AdamW and an independent embedding/head; no claim to verify the scaling law |
| [Foil v1 §2.1–2.2](https://arxiv.org/html/2609.35751v1), `foil_spec(num_experts=4, core_layers=4, passes=1)` | core `(E,D,L)→(2E,D/2,2L)`, where D is the number of physical banks; each logical visit has its own attention; experts, router and norms are shared within a bank | Common cell, public data and optimizer; the independent prefix/suffix stay, with the original E experts |
| [Tying v1 §3/Appendix B](https://arxiv.org/html/2606.16825v1), `expert_tying_spec(groups=2, group_size=4)` | Each group of g consecutive logical layers shares only the experts; attention, router and norm are independent per layer; expert LR=`base_lr/sqrt(g)`, other LRs do not change, the WD coefficient is not compensated | The recipe defaults to AdamW; the paper's background is Muon. With `optimizer.name` set to `"Muon"`, the core experts still scale their LR by `expert_lr_multiplier`, and WD is not compensated. Groups follow the real canonical core expert owners |
| [Huginn v2 §3/4.1](https://arxiv.org/html/2502.05171v2), `huginn_spec()` | Independent random latent, live prefix anchor, depth sampled per microbatch, backpropagation through the last `min(k,r)` visits, independent suffix | The common pre-norm MoE replaces the original dense sandwich background; the sampling parameters for a small model are explicit; fixed-depth zero-init evaluation, no cache |
| [LOOM v1 §3/4.2.1](https://arxiv.org/html/2610.01153v1), `loom_spec()` | Two normalized EMAs, embedding re-injection, residual scaling, per-visit router; full BPTT | The first path fixes H=3; no detach or optimizer step between segments; no cache |
| [Ouro v5 §3.2–3.4](https://arxiv.org/html/2510.25741v5), `ouro_spec()` | Per-token, per-visit expected CE−entropy; shared gate, the last visit takes the remaining probability, full BPTT | Zero initialization of the shared gate and the entropy coefficient 0.1 are public-stack choices; HF gives only the fixed-depth terminal readout |
| [Looped Models Done Right, Part I (v1) §2.1/3.1/3.4.2/4.1/Appendix A](https://arxiv.org/html/2610.23036v1), `sandwich_spec()`; dense path `sandwich_spec(dense=True)` | An independent prefix and suffix surround one shared core stack that runs `passes` times. Before each core visit, the state becomes `exp(-dt·a)·z + dt·W e` (`diag_decay`; e is the prefix output, with no RMSNorm; start at `W=I`, `a=1`, `dt=1`), with `z_0 = e`. Fixed depth, full BPTT, terminal CE. The default P2-R3x8-C2 keeps the 2:3:2 ratio of the paper's P8-R12x8-C8 | In the MoE path, each FFN is a top-2 expert bank with no shared experts and balance coefficient 0.001; it takes the input-initialized path of the released code (the paper's MoE tables report the random-state Huginn MoE). `dense=True` is the dense path of §3.4.2, with no aux. Softmax top-k replaces sigmoid top-2. The paper's 1536-wide cells, sequence length, vocabulary and optimizer settings are not reproduced |
| [Relaxed Recursive Transformers v3 §2.2/2.3/3.5/Appendix C](https://arxiv.org/html/2410.20672v3), `relaxed_spec()`; dense path `relaxed_spec(dense=True)` | CYCLE sharing: one shared block of layers (norms included) runs `passes` times in stack order, with no independent prefix/suffix. Each shared layer has one LoRA per visit, `h = W'x + BAx`, with no alpha scaling; the visit-l LoRA is used only at visit l. Zero initialization (`B = 0`, Gaussian `A`): training starts from a plain Recursive Transformer, and both the shared parameters and the LoRAs train. Fixed depth, full BPTT, terminal CE. Default 4 layers × 2 visits, rank 16 | MoE path: each FFN is a shared top-k expert bank; routers and routed experts get no LoRA, so only q/k/v/out are relaxed; balance coefficient 0.01, the repository default (the paper has no MoE). `dense=True` is the paper's path: q/k/v/out and SwiGLU w1/w3/w2 all get LoRA, with no aux. Training starts from random weights, not from uptraining Gemma/TinyLlama/Pythia, so the Stepwise/Average/Lower and truncated-SVD initializations do not apply (`LoRA.svd_init` gives eq. 4, for converted full-size weights). Attention and FFN each use one rank; the per-component rank ablation is not reproduced. Early exit and depth-wise batching are not in this path |
| [LatentLoop v1 §2.1/3.1, Figure 2, Table 5](https://arxiv.org/html/2610.19975v1), `latentloop_residual_spec()` | SMELT's split: the middle half of the physical layers runs twice, with every core owner (attention, router, experts, norms) shared, the paper's MoE Loop (§2.1). Each core attention and FFN residual branch is multiplied by 0.25 at the first visit and by 1 at the second (`visit_residual_scales`); the independent prefix/suffix use 1. Fixed depth, full BPTT, terminal CE | This is the fixed (0.25, 1) per-visit residual of §3.1 on SMELT's loop, not the paper's final LatentLoop method (a 2x mean-pooled first visit, a learned scaled decoder and a wider model, §3.2–3.4), which needs sequence pooling that the common model does not have. Fixed (0.25, 1) replaces learned per-visit coefficients. Attention stays shared across visits (Table 5). The common AdamW, schedule and run defaults replace the paper's training settings |

The first four, sandwich, relaxed and LatentLoop residual use fixed depth, full BPTT and terminal CE. These defaults must not overwrite the training differences of Huginn, LOOM and Ouro. All paths keep the explicit router aux reduction (the dense sandwich and relaxed paths have no aux). The `optimizer` of all ten `*_spec()` functions defaults to AdamW. To use Muon, change only `optimizer`; [Architecture](architecture.md#5-training-dynamics-must-be-visible) and [Training](training.md) give the grouping rule. RMSNorm/RoPE/SwiGLU, the independent embedding/head, the data and the run environment are the common background; record their complete identity. These are public adaptations of the selected core paths. They do not reproduce all the engineering and results of the original papers.

The historical engineering qualification of the four methods used a pre-tokenized DCLM fixture of 300M/1M/1M train/validation/test labels. It stays as a smoke test. The upstream download commit of the old copy is unknown; its parent manifest/chunk SHA identifies it. The import keeps the original tokens/EOS, does not decode or re-encode, and only prepends the BOS of this library, BOS=EOS=PAD151645. P1 uses raw DCLM that anyone can rebuild from public sources, with 24,000,067,249 /10,001,497 /10,009,889 train/validation/test valid labels completed. The Qwen3-30B-A3B tokenizer is pinned at `ae659febe817e4b3ebd7355f47792725801204c9`, vocab151669. A plain factory reads the vocabulary from the manifest explicitly before it builds the spec. [Training](training.md) gives the data identity, the tokenizer of four source files and the `--resume` entry.

## P1 ordinary MoE / SMELT

The real training configurations are published unchanged as [`p1_moe.json`](../recipes/p1_moe.json) and [`p1_smelt.json`](../recipes/p1_smelt.json). Load them directly as plain `RecipeSpec` objects. They keep the model, training, optimizer, the full schedule and the frozen identity:

```python
import json
from pathlib import Path
from moeloop.recipe_spec import RecipeSpec

spec = RecipeSpec.from_dict(json.loads(Path("recipes/p1_smelt.json").read_text()))
print(spec.model.hidden_dim)  # 440
print(spec.run["schedule_total_steps"])  # 19537
```

| Geometry | Ordinary MoE | SMELT |
|---|---:|---:|
| residual width | 672 | 576 |
| single-expert hidden | 544 | 440 |
| experts / top-k | 64 / 8 | 96 / 8 |
| physical prefix/core/suffix | 0 / 10 / 0 | 2 / 5 / 3 |
| core visits / logical cell calls | 1 / 10 | 2 / 15 |
| heads / KV heads / head dim | 7 / 1 / 96 | 9 / 1 / 64 |

Both arms have the 151669-token vocabulary, an independent embedding/head, 4096ctx, and fixed depth/full BPTT/terminal CE. Only SMELT multiplies the two core residuals by 1/2 each. The 100M-active tier means about 98.5M active body parameters in a single traversal of the ordinary baseline, without the embedding/head. Its non-embedding unique parameters are about 712.7M, and the total model is about 916.5M. These three numbers have different names and must not be mixed.

F440 comes from the 4096ctx document packing of the complete planned training data: epoch0/seed42/world16, 5,001,472 blocks, on average 4,180,216.9229844734 within-document causal attention pairs per sequence. An analytical screen of 12 pre-declared FFN widths selected the shape only by three budgets: unique non-embedding parameters, training matrix FLOPs and logical KV. It did not look at model quality. `p1_moe_spec()` and `p1_smelt_spec()` rebuild these two trained arms (F440, with the measured-pairs convention that P2 also uses). The default dense-S² analysis with no pairs selects F424, which was never trained.

Both arms completed 19,537 updates, each with 20,486,029,312 fresh labels and the single seed42. They used 16×H200 on two nodes with HSDP (replicate2×shard8) and local16/GA1/global256 sequences. The full WSD schedule is a 20-step warmup, stable up to step 10000 and a 9537-step cosine decay, with peak LR3e-4 that decays to .1× at the end. The real training source is `1bea4d6d2c539e3ae02517aaeede5f9ab68271a7`. CLI and documentation wiring after the release does not change this source.

[P1 evidence](evidence/p1.json) gives the full endpoint evaluation, the native curves and the real cost. [Cost and comparison](comparison.md) gives the analytical budgets and the limits of the scientific conclusions. [Training](training.md) gives the data rebuild and the multi-node training commands. This is an adaptation to public DCLM and the common model stack. It does not claim to reproduce the corpus of the original paper, and it does not extend a gain at one scale to all looped MoE methods.

## Foil Base and Tying topology

`foil_base_spec()` and `foil_spec()` take the same Base E,D,L. Both give each logical core visit its own attention. So the D×L attention owners do not change with flattening, and the hidden width and the single-expert FFN width also do not change. Per-visit attention alone is not the complete Foil.

With E=4,D=2,L=2, Base has two 4-expert banks with two calls per bank. Foil has one 8-expert bank with four calls. The outer cells use the original E in both. D must be a positive even number. The shared attention of Base-tied is a different ablation; give it its own name and record the difference.

Tying `groups=2, group_size=4` gives the consecutive groups `[0,1,2,3]` and `[4,5,6,7]`. The first bank runs four times in a row; then the next bank starts. After the optimizer is built, the native adapter finds the canonical core expert parameters and applies the learning-rate multiplier to them only. Attention, router and norm are not scaled. No vague regular expression on parameter names replaces the owner check.

## Change a region

In `ModelConfig` schema 4, `prefix/core/suffix: RegionConfig` can override `hidden_dim`, `num_experts`, `top_k`, `always_on_hidden_dim` and `residual_scale` per region. The regions share the residual width and the attention geometry. A reused configuration value creates no module or parameter alias. Old schemas 1/2/3 load with their original default behaviour; new state, gate and injection options are off by default. A readable configuration does not mean strict continuation across source versions.

```python
from dataclasses import replace
from moeloop import LoopedLM
from moeloop.model import RegionConfig
from moeloop.recipes import meta_spec

spec = meta_spec(vocab_size=128, dim=32, hidden_dim=64,
                 num_heads=4, num_kv_heads=2, num_experts=4,
                 max_seq_len=64)
variant = replace(spec, name="meta_core8",
                  model=replace(spec.model, core=RegionConfig(num_experts=8)))
model = LoopedLM(variant.model)
print(model.config.region_config("core").num_experts)  # 8
```

A research variant updates its source/adaptations, its comparison contract and its run evidence. After a change of parameter owners, gradients or the objective, old runs do not prove that the new path passes.

## Three added training semantics

`huginn_spec()` defaults to a shifted Poisson-lognormal: `r=1+Poisson(exp(tau))`. The parameters of `tau` give a mean depth of 4 and a log std of 0.5. The backpropagated tail is k=2. The seed, the successful update and the microbatch locate the depth. It is synchronized across ranks and has no hard cap. The training latent is an independent truncated normal with std=`sqrt(2/5)` and bounds ±3std. Evaluation explicitly uses zero-init and `eval_passes=4` instead. These two initial-state distributions are not the same experimental condition.

`loom_spec(core_layers=2, passes=3)` uses `gamma=lambda/(H*sqrt(M))`. Later visits re-inject the embedding with `g_t=lambda/(t*sqrt(M))`. The defaults are lambda=beta=0.5. The global EMA resets at each forward; the local EMA resets at each visit. After the scaled attention write, the two normalized reads are added directly; then the output-normalized MoE residual follows. The state is temporary and differentiable. It is not a buffer across samples.

In `ouro_spec(passes=3, entropy_coeff=.1)`, the shared gate gives per-token conditional halting probabilities. The first R−1 visits get probability mass by the hazard; the last visit takes the remaining mass. The objective is `sum_r p_r*CE_r - beta*H(p)`, plus the separate aux. Training runs all visits and keeps the gradients of the probabilities, the head and the backbone. The native expected CE recomputes the head per chunk to limit the vocabulary-size intermediates. The HF `labels` and the standard held-out evaluation stay terminal CE; they are not this training objective.

The example below shows only the readable configurations and a fixed-depth reference forward. It does not start training and does not claim method qualification:

```python
import json
import torch
from moeloop import LoopedLM
from moeloop.recipe_spec import RecipeSpec
from moeloop.recipes import huginn_spec, loom_spec, ouro_spec

geometry = dict(vocab_size=23, dim=16, hidden_dim=32, num_heads=4,
                num_kv_heads=2, num_experts=4, top_k=2, max_seq_len=32)
for make in (huginn_spec, loom_spec, ouro_spec):
    spec = RecipeSpec.from_dict(json.loads(json.dumps(make(**geometry).to_dict())))
    model = LoopedLM(spec.model).eval()
    with torch.no_grad():
        output = model(torch.tensor([[1, 2, 3]]), collect_readouts=False)
    print(spec.name, spec.training.objective, tuple(output.logits.shape))
```

## Combinations and components

Four plain configurations express the attention×router grid. The expert bank, the call graph, norms, the objective and the training rules are fixed; only the two owner switches change. The grid tests the combination of components. It does not claim a matched total parameter count or a complete Foil×LOOM:

```python
from dataclasses import replace
from moeloop.recipes import recursive_moe_spec

base = recursive_moe_spec(**geometry)
grid = {
    f"a{a}_r{r}": replace(base, name=f"a{a}_r{r}",
        model=replace(base.model, attention_per_visit=bool(a), router_per_visit=bool(r)))
    for a in (0, 1) for r in (0, 1)
}
```

`loom_spec(attention_per_visit=True)` is another complete state×structure combination. It keeps the two EMAs, re-injection, scaling, the per-visit router and full BPTT, and adds only per-visit attention owners. Its method qualification is recorded separately. It does not replace the base LOOM path.

| Component entry | Meaning |
|---|---|
| `dense_spec()` / `moe_spec()` | Ordinary non-looped baselines; resource matching is a separate step |
| `recursive_moe_spec()` | A shared stack loop body |
| `per_visit_router_spec()` | A per-visit router; used alone, it has no LOOM memory or state |
| `input_anchor_spec()` | Live anchor, update-level depth sampling and tail gradient; used alone, it has no Huginn random initial state or Poisson-lognormal process |
| `ModelConfig(input_injection="diag_decay" or "orthogonal")` | The two injection operators of xllm-loop Part II (schema 4); `sandwich_spec` uses `diag_decay`; `orthogonal` has no named recipe yet; [Architecture](architecture.md#4-the-core-model-stays-plain-pytorch) gives the initialization |
| `ModelConfig(lora_attention_rank=r, lora_ffn_rank=r)` | Per-visit LoRA (new fields, default 0; an old record without these two fields reads them as 0). It acts only on the core, not on routed experts. It excludes the EMA state and needs fixed depth and full BPTT. `relaxed_spec` uses it |
| `tiny_spec()` | A small configuration for installation and engineering development |

From this version on, `loom()` builds the complete memory method. Use `per_visit_router()` for the old router component. `injected()` stays a deprecated alias of `input_anchor()`. For an old checkpoint, read its real configuration and source. Do not replace it with the new method by the old factory name and then claim an exact resume.

Combinations with no gradient definition at this stage, such as the Huginn tail gradient with the all-visit expected CE, are rejected explicitly. Ouro Stage II, real token early exit and its KV strategy are not in this selected path. A new state update still uses a short explicit forward and reuses the data, training and HF seams. Continue with [Architecture](architecture.md), [Cost and comparison](comparison.md) and [Training and resume](training.md).
