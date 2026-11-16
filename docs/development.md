# Implementation and contribution

Develop according to the [architecture](architecture.md). Several developers or subagents divide the work by files that do not overlap, and an integrator reviews it. Local interfaces, tests and implementation details iterate autonomously. Substantive changes to research questions, method deliverables, resources and formal comparison conditions are stated separately. CVSM serves formal controlled experiments.

The core stays as plain functions, modules and explicit loops. A new combination of existing building blocks adds a recipe. A new local operator adds a small module. A truly different state update keeps its own short forward. When a new forward changes the parameter owners or the number of cache occurrences, add a bounded runtime/export integration. Do not put all responsibilities into one huge file only to keep the directory flat. Do not build a GraphExecutor, a Policy or a plugin platform in advance for unknown needs.

A new method first provides a `*_spec()` that can be imported on its own. The spec states its source, what it keeps from the original and what it adapts. Use `compare_recipes.py inspect/diff/compare` to check the actual configuration. [Recipes](recipes.md) gives the boundaries of the nine named methods. [Comparison](comparison.md) gives the cost estimates and the formal comparison questions. When you finish a `*_spec()`, run the [conformance tests](#conformance-tests-and-diagnostics) first. The legacy component aliases `loom/injected` do not count as a complete method contribution.

## Development environment and checks

The control protocol environment:

```sh
uv sync --frozen
uv run --frozen python -m pytest -q
```

For the model environment, install CPU Torch and the HF/data extras as the README says. This is enough to check the plain model, the data and the independent consumers. For the native adapter checks, also install the train extra and the Triton that this pin needs:

```sh
uv pip install -e '.[data,hf,train,report]' 'triton==3.8.0' 'pytest==8.3.3'
python -m pytest -q tests/model tests/data tests/training tests/hf tests/delivery tests/reference tests/qualification tests/comparison tests/setup
python scripts/compare_baselines.py --output /tmp/qwen3-reference.json
git diff --check
```

CI checks the Torch-free control interface and the complete CPU model/native integration separately. Tests that depend on Torch use `pytest.importorskip("torch")` at the top of the file. The control environment skips them, and the model environment job covers them. Tests should check actual semantics. Examples: a shared gradient equals the sum over its calls; different loop orders really differ; no-grad/anchor paths; packing/cursor; the official baseline; real FSDP/AC/chunked loss; a separate HF process. Do not copy the implementation line by line to increase the number of tests.

## Conformance tests and diagnostics

[`moeloop.testing.check_recipe`](../src/moeloop/testing.py) is the conformance test suite for any recipe. It instantiates the recipe at a small geometry and runs it on a CPU:

```python
from moeloop.recipes import smelt_spec
from moeloop.testing import check_recipe

spec = smelt_spec(vocab_size=128, dim=32, hidden_dim=64, num_heads=4,
                  num_kv_heads=2, num_experts=4, max_seq_len=64)
report = check_recipe(spec)  # raises ConformanceError when a check fails; its report lists each result
print({name: check["status"] for name, check in report["checks"].items()})
```

The six checks are:

- `ownership`: parameter owners are unique and do not share storage.
- `shared_gradients`: a shared gradient equals the sum over its calls.
- `cost_calls`: the call counts and parameter counts of the cost model equal one real forward.
- `cache`: chunked cache decoding equals full recomputation, and the KV slots and bytes equal the cost model. A recipe that does not use a cache gets not_applicable.
- `hf_roundtrip`: the logits are the same after `save_pretrained`/`from_pretrained`.
- `identity`: a spec rebuilt from JSON keeps the continuation identity and the initial weights.

The suite reuses the oracles of `tests/model` and `tests/hf`. It checks the contract of `LoopedLM(spec.model)` on the float32 CPU reference path. It says nothing about quality, training stability, native GPU kernels or the export command line. A hand-written forward (see [Extensions](extensions.md)) is also outside it.

After you train a model, [`moeloop.diagnostics`](../src/moeloop/diagnostics.py) reads the per-visit behaviour (logit lens, residual updates, routing overlap, effective depth). For the commands, see the section on export, evaluation and generation in the [training documentation](training.md). A Looped MoE Bench submission checks itself with `moeloop.bench.validate`; see [Comparison](comparison.md#looped-moe-bench). Diagnostics help you understand a model. They are not method qualification.

## GPU development and debugging

GPU training uses only authorized and allocated resources. Deployment addresses, nodes and credentials do not go into public recipes.

Debug continuously in the same available environment, and reuse the data and cache. When it works, fix the actual source identity and run the qualification in a clean process. Each attempt keeps the source hash, configuration, command, runtime and raw results. A failure does not overwrite them. Put the durable checkpoint, the tokenizer/data manifest, the HF export and the evidence into a verified persistent directory. Do not hand over a temporary container path as a result.

## Method qualification and engineering probes

The real-text qualification entry point for the four methods uses a given manifest and the visible GPUs. It does not manage the cluster:

```sh
export MOELOOP_DATA=/data/dclm-qwen3-v1
export MOELOOP_EVIDENCE=/runs/v1-evidence
python scripts/recipe_qualification.py --stage v1 \
  --manifest "$MOELOOP_DATA/manifest.json" --output "$MOELOOP_EVIDENCE" --gpus 2
python scripts/check_release.py --stage v1 --evidence "$MOELOOP_EVIDENCE" --engineering-only
```

The commands above only create and check the engineering qualification of the four methods. `--engineering-only` does not mean complete V1 acceptance. Complete V1 acceptance also needs the raw assets of the actual P1, of public performance and of 8/16-GPU resume, which together make the complete evidence. Run `python scripts/check_release.py --stage v1 --evidence "$MOELOOP_FULL_EVIDENCE"` on that complete directory, without engineering-only. The [validation record](validation.md) gives the completed scope and the difference between raw assets and public projections.

The default target is 512 successful updates for each of the four methods, context2048 and a global batch of 16×2048 positions. The target also checks native FSDP, full resume and independent HF consumption of the actual checkpoint. `--pilot` is a bounded development check. It does not count as method qualification. The checker reads real artifacts. A missing item or an identity that does not match must fail. A command that exists does not mean that its run is complete. For results, see the [validation record](validation.md).

The old small batch is for regression of engineering seams. `scripts/qualification.py` does not allocate resources. It only prepares an offline engineering fixture, checks a durable full checkpoint and compares continuation:

```sh
export PYTHONPATH="$PWD/src:$PWD/scripts:$PYTHONPATH"
export MOELOOP_DATA=/runs/qualification/data
python scripts/qualification.py prepare --output "$MOELOOP_DATA"
export MOELOOP_OUTPUT=/runs/qualification/shared
export MOELOOP_QUAL_VARIANT=shared MOELOOP_STEPS=4
torchrun --standalone --nproc-per-node=4 -m moeloop.train \
  --module qualification_recipes --config smoke
python scripts/qualification.py verify --directory "$MOELOOP_OUTPUT" --steps 4
```

The 2–4 updates of `shared`, `partial_attention`, `partial_router`, `expert_tying`, `injected` and `dense` only check existing components. They do not replace the real-text short training of the four methods. When you compare 4 continuous steps with stop2/resume4, keep the fixture, source snapshot, topology, horizon16 and recipe the same. Only the stop target and the explicit checkpoint load location differ:

```sh
python scripts/qualification.py compare --uninterrupted /runs/qualification/full4 \
  --resumed /runs/qualification/resumed4 --output /results/continuation.json
```

Model tensors use explicit numerical tolerances. The RNG, cursor, schedule and discrete completion state of all ranks must match exactly. Also run export, held-out evaluation and cached generation of the actual native checkpoint. A multi-node configuration needs real multi-node evidence. It cannot be inferred from a green single-machine run. Deployment addresses, nodes, credentials and scheduler resource settings stay with the user and do not go into public recipes.

The BF16 numerical comparison of native grouped experts/FlexAttention uses an [independent script](../scripts/compare_native.py). It needs an allocated CUDA GPU and the fixed TorchTitan environment:

```sh
python scripts/compare_native.py --output /results/native-kernel.json
```

The script fixes the seed, the document/padding inputs, and the shared and injected-suffix configurations. It compares the model forward and the gradients of all physical owners, balance/z included. Exact routing is checked as the expert set per logical call and per valid token. The top-k return order can differ. An aggregate histogram cannot replace this check. Gradients must meet both the per-tensor/global relative L2 bound and the BF16 tensor-peak bound. The [validation record](validation.md) has the exact thresholds and the machine records of the first failure and of the re-test. Keep the failure facts and the raw results. After a change of the acceptance policy, run again separately. Do not overwrite the old state.

## Public commitments

Use the [validation record](validation.md) to describe the specific hardware, versions and supported combinations that passed. Readable source code or a passing CPU check does not prove every run combination, performance or scientific effect. When you find a local execution problem, first keep the algorithm and simplify or fix the implementation. Discuss it only when the change affects the core mathematics or the public support scope.

The same applies to the Muon path. The grouping rule and one real step on a single CPU rank have tests (`tests/training/test_muon.py`). The data flow that stays the same across GPU counts was tested on 1, 2 and 4 CPU FSDP ranks (`tests/training/test_topology.py`). #49 completed Muon training, a complete checkpoint save and a WSD fork on 4-GPU and 8-GPU FSDP. That run did not cover HSDP `dp_replicate>1`, the sharding boundary where the world size is larger than the number of experts, experts with no tokens, or Muon checkpoint continuation.
