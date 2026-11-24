# Write a different state update

To change an existing mechanism, such as the depth or shared attention or routers, compose `ModelConfig` and `TrainingRecipe` directly (see [Add a configuration](#add-a-configuration)). To change the order of the operators inside a block, write an explicit new forward. The [FFN-chain example](../recipes/ffn_chain.py) reuses the physical parameters of an existing cell. It adds no paper switch and no general executor to the core.

The example fixes one semantics: attention once, the FFN norm once, then several updates with a shared router and shared experts. It keeps an inner and an outer residual:

```text
a = x + scale * Attention(AttentionNorm(x))
z = FFNNorm(a)
repeat K times:
    z = z + scale * Experts(z, Router(z))
y = a + z
```

Even with K=1, this is not a plain cell. The outer residual adds the result of the whole chain, and that result includes the normalized initial state. The positions of these residuals and norms are part of the method definition, so the forward states them directly. The structure follows [the pinned CoE source](https://github.com/ZihanWang314/coe/blob/d2af562aab3348f72d5d17132df439a629b9666d/config/models/coe_deepseekv2/modeling_coe.py#L1271-L1309). This example selects only the shared router/bank and the inner+outer residual. It does not claim to reproduce the paper's routing, separate gates, data or training details.

Run the minimal example below from a Git checkout. Here `LoopedLM` (`RecursiveLM` is its compatibility alias) supplies the initialized parameter owners. The new update function calls these parameters explicitly:

```python
import torch
from moeloop import LoopedLM, ModelConfig
from recipes.ffn_chain import forward_ffn_chain

model = LoopedLM(ModelConfig(vocab_size=128, dim=64, hidden_dim=128,
    num_heads=4, num_kv_heads=2, num_experts=4, num_cells=1, num_visits=1))
ids = torch.randint(0, 128, (2, 16))
positions = torch.arange(16).expand_as(ids)
hidden, balance_sum, z_sum = forward_ffn_chain(
    model.core[0], model.embedding(ids), ffn_steps=3, positions=positions)
logits = model.lm_head(model.final_norm(hidden))
```

`balance_sum` and `z_sum` sum over the three FFN calls. For a per-call mean, divide by 3 explicitly. `attention_mask` takes a prepared causal/document mask. `router_mask` marks the valid inputs separately. The example uses full BPTT.

This is only an extension example of the reference forward. A direct call `model(ids)` still runs the original whole-block recurrence. An export of the existing `LoopedLM` also saves only the original model semantics. To make the new method a trainable and deliverable model, define a serialized identity for your own model, check the new logical attention/cache counts, and then integrate it explicitly with the native runtime and HF. The current example has none of these qualifications. Do not train through a monkey patch and then export with the old config.

The checks cover the real attention/norm/router/expert call counts, the gradients of all parameters, and a residual counterexample with a zero update: `python -m pytest -q tests/model/test_extension.py`.

The conformance suite [`moeloop.testing.check_recipe(spec)`](../src/moeloop/testing.py) checks `LoopedLM(spec.model)` itself: parameter owners, shared gradients, the call counts of the cost model, the cache, the HF round trip and the identity of a spec rebuilt from JSON. It does not run a forward that you write here. A self-written update needs its own checks, as above; [Development](development.md) gives the details.

## Replace attention, and per-visit LoRA

`LoopedLM(config, attention_factory=...)` replaces the attention module of each cell. Native training uses it to put FlexAttention on the same parameters. A cell calls the module as `attention(x, positions=..., attention_mask=..., cache=..., slot=...)`. The cell also passes `lora=`, but only when attention LoRA is on (`lora_attention_rank > 0`).

- With attention LoRA off, the module from a third-party `attention_factory` needs no change.
- To support attention LoRA, `forward` must accept `lora`: the `nn.ModuleDict` of the current visit, with the keys `q_proj`, `k_proj`, `v_proj` and `out_proj`. Add `lora.q_proj(x)`, `lora.k_proj(x)` and `lora.v_proj(x)` to the outputs of the matching projections. The `out_proj` increment acts on the merged head output `h`: add `lora.out_proj(h)` to `out_proj(h)`. The built-in [`Attention`](../src/moeloop/model.py) does exactly this. If `forward` does not accept `lora`, attention LoRA raises `TypeError` when it is on.
- The cell itself adds the FFN LoRA (`lora_ffn_rank > 0`) on `w1`, `w3` and `w2` of the dense or always-on FFN. This puts no requirement on `expert_factory` or `attention_factory`. Routed experts get no LoRA.

`lora_attention_rank` and `lora_ffn_rank` of `ModelConfig` default to 0 and act only on the core: in the resolved prefix and suffix, both are 0. They exclude the EMA memory cell (`state_update="ema"`). `lora_ffn_rank` needs a dense or always-on FFN in the core. Like other per-visit owners, they need fixed depth and full BPTT. Each LoRA starts at zero by default (normal `A`, `B = 0`), so the initial model is bit-identical to the shared-weight model. To convert a full-size model into a relaxed model, use `LoRA.svd_init` for the truncated-SVD initialization of eq. 4 of the paper. It makes `W' + B A` the best rank-r approximation of the weight `W_l` of that layer at that visit:

```python
import torch
from moeloop.model import LoRA

shared, full = torch.randn(64, 32), torch.randn(64, 32)  # W' and W_l, shape [out, in]
lora = LoRA(32, 64, rank=8)
lora.svd_init(full - shared)
approx = shared + lora.B @ lora.A  # best rank-8 approximation of W_l
```

`NoLoRAAttention` in [`tests/model/test_relaxed.py`](../tests/model/test_relaxed.py) checks this calling contract. With attention LoRA off, the result is bit-identical to the built-in attention. With attention LoRA on, it raises `TypeError`. For use at the recipe level, see `relaxed_spec` in [Recipes](recipes.md).

## Add a configuration

A new configuration is a new recipe: a `RecipeSpec` with the fields that you change. It uses the existing forward. The four steps below run on a CPU from the root of a Git checkout. They start no training.

**1. Build it.** Load a reference recipe and change its model fields with `dataclasses.replace`. This example adds a per-visit router to the Phase 1 SMELT arm `phase1/m3` and writes the result to the current directory:

```python
import json
from dataclasses import replace
from pathlib import Path
from moeloop.recipe_spec import RecipeSpec

reference = RecipeSpec.from_dict(json.loads(Path("recipes/phase1/m3-adamw.json").read_text()))
variant = replace(reference, name="m3-router-per-visit-adamw",
                  model=replace(reference.model, router_per_visit=True))
Path("m3-router-per-visit-adamw.json").write_text(json.dumps(variant.to_dict(), indent=2) + "\n")
```

`replace` copies `source`, `preserved` and `adaptations` from the reference. Write them again for the new configuration before you publish it.

**2. Cost it against the reference.**

```sh
python scripts/compare_recipes.py diff recipes/phase1/m3-adamw.json m3-router-per-visit-adamw.json
python scripts/compare_recipes.py compare recipes/phase1/m3-adamw.json m3-router-per-visit-adamw.json --seq-len 4096
```

`diff` prints each changed field with its `reference` and `candidate` values. Here these are `model.router_per_visit` and the derived `model.visit_capacity`. `compare` prints one JSON document. Its `matched` section holds the three budgets: `unique_non_embedding_parameters`, `training_flops_per_token` and `logical_kv_bytes`. Each budget has both values, the relative difference and the relative tolerance (default 0.05; change it with `--tolerance`). For this example, the unique non-embedding parameters are 1315329312 against 1314771744 (relative difference 0.00042). The training FLOPs per token and the KV bytes are equal. `costs` holds the full breakdown of each arm, and `recipe_differences` repeats the output of `diff`. Without `--attention-pairs-per-sequence`, the attention cost uses dense S² arithmetic. To use the document convention, give the measured mean number of within-document causal attention pairs per packed sequence.

**3. Run the conformance check.** `check_recipe` builds `LoopedLM(spec.model)` at the geometry of the spec. Run it on a small instance of the same change:

```python
from dataclasses import replace
from moeloop.recipes import smelt_spec
from moeloop.testing import check_recipe

small = smelt_spec(vocab_size=128, dim=32, hidden_dim=64, num_heads=4,
                   num_kv_heads=2, num_experts=4, max_seq_len=64)
small = replace(small, name="smelt-router-per-visit",
                model=replace(small.model, router_per_visit=True))
report = check_recipe(small)
print({name: result["status"] for name, result in report["checks"].items()})
# {'ownership': 'pass', 'shared_gradients': 'pass', 'cost_calls': 'pass', 'cache': 'pass', 'hf_roundtrip': 'pass', 'identity': 'pass'}
```

If a check fails, `check_recipe` raises `ConformanceError`, and its `report` holds the result of every check.

**4. See the lineage.**

```sh
python scripts/lineage.py
```

The command prints the tree of the published recipes. The root is the standard MoE `phase1/m0`. Each node lists the `ModelConfig` fields that it changes against its parent. In this tree, the reference `phase1/m3` is a child of `phase1/m0`. Against `phase1/m3`, the new configuration changes one field, `router_per_visit` (step 2). `--json PATH` also writes the nodes, the parents and the changed fields with their old and new values. The tree holds only the published recipes under `recipes/`, so the new configuration does not appear in it.

### What a new forward pass gets today

Every step above applies to a configuration, because it uses the existing forward. A new forward pass, such as the FFN chain above, is different.

A new forward pass gets:

- the parameter owners of `LoopedLM` and their initialization;
- the existing modules (attention, routers, experts, norms), which it calls explicitly;
- its own tests, such as `tests/model/test_extension.py`.

A new forward pass does not get:

- `model(ids)`: the call still runs the original recurrence;
- the cost model: `estimate_cost` and `scripts/compare_recipes.py` count the calls of the original forward from the configuration;
- the conformance check: `check_recipe` checks only `LoopedLM(spec.model)`;
- native TorchTitan training and HF export: both run or save the original forward.
