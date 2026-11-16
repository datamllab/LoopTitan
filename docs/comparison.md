# Cost and comparison contract

[`moeloop.costs.estimate_cost()`](../src/moeloop/costs.py) and [`moeloop.comparison`](../src/moeloop/comparison.py) return plain JSON. They do not allocate a GPU and do not start training. Parameter storage, logical execution, data exposure and measured results are reported separately.

```python
from moeloop.recipes import foil_base_spec, foil_spec
from moeloop.costs import estimate_cost
from moeloop.comparison import compare_recipes, diff_recipes

base, foil = foil_base_spec(), foil_spec()
cost = estimate_cost(foil, seq_len=2048,
                     activation_checkpointing="full_cells")
print(cost["formula_version"])  # fixed_terminal_matrix_v1
print(cost["calls"]["forward_cells"])  # 6
comparison = compare_recipes(base, foil, seq_len=2048)
print(sorted(comparison["unknown"]))  # ['data', 'evaluation', 'quality', 'runtime']
print(len(diff_recipes(base, foil)) > 0)  # True
```

A recipe with sampled depth also returns numbers. It no longer returns `None`:

```python
from moeloop.costs import estimate_cost
from moeloop.recipes import huginn_spec

depth = estimate_cost(huginn_spec(), seq_len=2048)
print(depth["formula_version"])  # sampled_depth_matrix_v1
print(depth["flops"]["training_depth"]["quantile_visits"])  # {'p50': 4, 'p95': 8}
```

## Reporting definitions

| Field | Meaning |
|---|---|
| `parameters.unique_total/unique_trainable` | Each actual owner counts once. Trainable defaults to the unfrozen model. |
| `parameters.unique_non_embedding` | Excludes both the separate input embedding and the output head. All other parameters stay in. |
| `parameters.by_component/by_region` | Attention, norm, router, expert and head, and the owners and geometry of each region |
| `calls` | Training forward, backward and recompute Cells, router, expert assignments and terminal head, and separately `inference_cells` |
| `flops` | Matrix forward, backward and recompute per padded input position, with separate head and router items |
| `tokens` | The batch, context and packed positions that the estimate uses. The run supplies the actual valid inputs and labels. |
| `kv` | K+V per logical attention call, computed with `spec.inference.num_visits`, in bytes at the given context, batch and dtype |
| `unknown` | Operators that are not estimated, the optimizer, runtime memory, actual tokens and measured work |

The default formula `fixed_terminal_matrix_v1` counts one MAC as 2 FLOPs. Attention QK and AV count as two dense S×S matrix multiplications. The formula does not subtract the causal triangle or the document mask. When you pass `attention_pairs_per_sequence` explicitly, the formula is `fixed_terminal_document_matrix_v1`. For this value, count the causal query/key pairs (diagonal included) that the actual input mask and document boundaries allow in each packed sequence, and pass their mean. Do not guess the value from the label count or from the mean document length. `estimate_cost()`, `compare_recipes()` and `smelt_p1_pair()` all accept this parameter.

In both definitions, the full-BPTT matrix backward is estimated as two times the forward. `full_cells` adds one replay of the Cell matrices. The terminal head and the input adapter are not in these Cells. `activation_checkpointing=None` means that the recompute is unknown. It does not mean zero. The document-pairs correction applies to the analytical matrix work. It is not equal to the actual block or kernel work of FlexAttention, and it is not wall time.

Sampled depth and tail gradients (`sampled_visits`/`depth_rule` and `grad_visits` of `TrainingRecipe`) use `sampled_depth_matrix_v1` (`sampled_depth_document_matrix_v1` when `attention_pairs_per_sequence` is given). The forward counts at the expected depth. Backward and recompute cover the prefix, the suffix, the head and the last `min(grad_visits, k)` core visits. The estimate takes the expectation over the declared depth distribution (the work is linear in k and in `min(grad_visits, k)`). `flops.training_depth` also gives the cost of one step at the p50 and p95 depths of this rule (`quantiles.p50/p95.training_per_token`, per padded position, for one update or one microbatch). The work does not decrease with depth, so these values are a typical cost and a high cost for one step. These values are estimates on the declared distribution. They are not the depths that one run sampled, and they are not measured work. For those, read the actual r of each microbatch in the training record. With sampled depth, the logical call counts in `calls` are `None`. The values at the nominal depth are in `calls.nominal`. When tail gradients have no input injection (the path from the final loss to the prefix is cut), or when the objective is not supported, there are no reliable training FLOPs. The value is then `None`, with a reason. Dispatch, norm, softmax, CE, optimizer, communication and allocator items are not counted and stay unknown. Matrix FLOPs do not replace wall time, MFU or peak memory. Parameter sharing does not merge logical KV slots.

Fixed-depth training calls and FLOPs use `model.num_visits`. A reliable training estimate also requires that `training.num_visits` matches it. (With sampled depth, the depth rule sets the depth, and `model.num_visits` gives only the nominal forward.) The KV cache uses `spec.inference.num_visits` independently. Only when that value is not set does it fall back to the model depth. The output fields `calls.inference_cells`, `kv.num_visits` and `kv.logical_slots` show this difference. Thus a change of only the declared inference depth does not change the training FLOPs, but it does show the actual logical KV difference. A per-visit module above `visit_capacity` is rejected. A comparison must fix the training depth and the evaluation depth separately. It must not use the training depth to override the inference declaration.

## CLI and plain JSON

```sh
python scripts/compare_recipes.py inspect smelt --seq-len 2048
python scripts/compare_recipes.py diff foil_base foil
python scripts/compare_recipes.py compare moe smelt --seq-len 2048 \
  --activation-checkpointing full_cells
python scripts/compare_recipes.py compare recipes/p1_moe.json recipes/p1_smelt.json \
  --seq-len 4096 --attention-pairs-per-sequence 4180216.9229844734
python scripts/compare_recipes.py pair > /tmp/moeloop-smelt-pair.json
```

Run the commands from the checkout root, or use the absolute path of the script. A recipe argument can also be the path of a JSON file saved with `RecipeSpec.to_dict()`. `inspect` prints the recipe and its cost. `diff` shows the actual differences of the resolved model, training, optimizer, run and inference sections. `compare` makes a comparison record. Region inheritance is resolved first. Thus an inherited value that is written out explicitly does not show as a mechanism difference. `inspect` and `compare` accept explicit document pairs. `diff` does not compute costs, and `pair` keeps its own small dense-S² contract. Neither of these two accepts the parameter.

`compare_recipes(reference, candidate, seq_len=..., data=..., evaluation=...)` can take plain data and evaluation identities. Before the comparison, fix these items: the question and arms, the manifest and tokenizer, validation/test isolation, the effective data exposure, the objective, the optimizer and schedule, the seed, the endpoint, the inference depth and readout, the common held-out metric (CE or byte-exact BPB) and the selection rule. When the CLI does not attach actual identities, these items stay unknown.

The four categories are `matched`, `controlled`, `measured-different` and `unknown`. The three analytical budgets are matched only when they are within the tolerance. A setting is controlled only when the configurations are actually identical. `measured-different` keeps the budget differences that were computed, with `basis=analytical_estimate` to show their source. The group name does not mean that a GPU measurement was made. An unknown item is never judged as matched.

## Historical small engineering pair

`smelt_budget_pair()` enumerates a fixed set of five SMELT shapes, E∈{8,10,12,14,16}, and returns the comparison for the selected E=12. The baseline has 12 independent layers with 8 experts each. SMELT has 8 physical layers (2 prefix, 4 core×2, 2 suffix) with 12 experts per bank. Both have 12 logical Cell calls. Both use d=128, a hidden size of 256 per expert, top-k=2 and heads=4/2.

With the API defaults (vocab50257, context2048, batch1, BF16 KV, no AC), the unique non-embedding parameters are 10,042,496 / 9,844,864. The matrix training FLOPs per position are 94,114,560 / 94,151,424. The logical KV is 6,291,456 bytes for both. For all three, the absolute difference relative to the baseline is ≤5%. The 5% is an engineering rule. It is not presented as a tolerance from the original paper. This is the default analytical example. Actual DCLM runs must use the manifest's vocab151669 explicitly and compute the work again with the head included. `pair` fixes its CLI assumptions. For other AC, dtype, batch or tolerance values, use `compare`.

This JSON proves the declared analytical budget pair. The real pair pilot, the four-method short training, resume and the HF evidence are checked against the actual artifacts; see the [validation record](validation.md). The JSON gives no quality ranking.

## P1 and the within-document attention budget

`smelt_p1_pair()` fixes the ordinary baseline as d672/L10/E64/top8/F544/heads7/KV1. SMELT is d576 with 10 physical layers (2+5+3, the middle 5 layers run twice) and E96/top8/heads9/KV1. The shape is selected only among 12 pre-declared FFN widths (384 to 472, step 8). The selected shape has the smallest maximum analytical budget difference. Held-out quality is not used to select the shape. Both arms use vocab151669 and context4096. SMELT has 15 logical Cell calls in total.

The 100M-active tier of the ordinary baseline is an active non-embedding body of about 98.5M for one pass. Its unique non-embedding parameters are about 712.7M. Its total parameters, with the two separate vocabulary owners, are about 916.5M. The actual SMELT F440 has about 737.8M unique non-embedding parameters and about 912.6M total parameters. The output fields explain the active, unique and logical-visit work separately. Do not use "100M" for the total model size.

The data of this run has train 24,000,067,249, validation 10,001,497 and test 10,009,889 valid labels. Each arm completed 19,537 updates and 20,486,029,312 fresh labels. The budget uses the [statistics of the complete training prefix](evidence/p1-data-statistics.json): epoch0, seed42, world16, 5,001,472 blocks of 4096 tokens, no padding. The mean number of within-document causal query/key pairs allowed per sequence is **4,180,216.9229844734**, diagonal included. This value covers the complete planned training prefix and was cross-checked with the actual loader. It is not from a small sample of 20M labels. It is not equal to dense-S² or to the full causal triangle.

Load the published exact JSON directly to compute the values again. The example below runs only the CPU analysis. It does not start training.

```python
import json
from pathlib import Path
from moeloop.recipe_spec import RecipeSpec
from moeloop.comparison import compare_recipes

reference = RecipeSpec.from_dict(json.loads(Path("recipes/p1_moe.json").read_text()))
candidate = RecipeSpec.from_dict(json.loads(Path("recipes/p1_smelt.json").read_text()))
comparison = compare_recipes(
    reference, candidate, seq_len=4096, batch_size=1, bytes_per_element=2,
    activation_checkpointing="none", tolerance=.05,
    attention_pairs_per_sequence=4180216.9229844734,
    data={"manifest_fingerprint": "e67a956d2fe51dd1784505a6f58d7790f397a09624330700c0ebaffcd344f152"},
    evaluation={"seq_len": 4096, "readout": "terminal", "full_validation_and_test": True},
)
print(json.dumps(comparison["matched"], indent=2))
```

| Analytical budget, relative difference to the reference | Ordinary MoE | SMELT | Difference |
|---|---:|---:|---:|
| unique non-embedding parameters | 712,656,672 | 737,845,056 | +3.5344% |
| training matrix FLOPs / padded position | 1,284,757,348.671 | 1,248,742,044.863 | −2.8033% |
| logical KV bytes, batch1 / 4096ctx / 2 bytes | 15,728,640 | 15,728,640 | 0% |

All three budgets are within the pre-declared 5%. batch1 is only the unit of the cost report. The actual training global batch is 256 sequences. The budget screen over the fixed 12 FFN candidates selected F440 and used no quality metric. `p1_smelt_spec()` gives this F440 arm (with the measured-pairs definition, the same as P2). Without pairs, `smelt_p1_pair()` and `recipe_anchor.py` use dense-S² and select F424, which was never trained. A newly generated provisional JSON does not reproduce the actual frozen identity of this run. For the data rebuild, the 16-GPU 2×8 HSDP and the complete WSD commands, see the section on the actual P1 recipes and two-node training in the [training documentation](training.md).

The training source of this run is `1bea4d6d2c539e3ae02517aaeede5f9ab68271a7`. Both arms use the same data order, seed42, batch and microbatch, optimizer, endpoint and readout. The training, data, run and evaluation identities were frozen before the start. Later patches to the public entry points only expose existing parameters and actual artifacts. Their new HEAD is not written down as the training source.

### How to read the P1 results

The [P1 evidence](evidence/p1.json) reports these items separately: the complete validation/test endpoints of about 10M labels each, the [native validation curves of about 1M labels](evidence/p1-native-curves.svg) with a [zoom on the late part](evidence/p1-native-late-curves.svg), the actual GPU hours and the three budgets. The endpoints at the same data amount D answer the question of quality at equal data. The curves at common analytical compute C interpolate only inside the range that both arms observed. They keep the neighbouring points and weights, and they do not extrapolate into a region that was not trained. C is the frozen matrix formula times the actual padded positions. Do not call it measured GPU FLOPs. The complete FP32 HF endpoint evaluation and the native BF16 fixed validation slice are not merged into one curve.

A negative SMELT−MoE CE difference means that SMELT is better. Paired document evaluation requires the same data, context and readout, and the label count of each document. It does not estimate the variance across training seeds. This run trained only seed42. A small advantage or no advantage is kept as it is. A single-point lead is not used to claim a general gain or a complete reproduction of the SMELT paper.

Run cost and the implementation check are also separate. P1 reports the wall-clock time and the GPU hours of the two structures as measured. The [same-geometry 8-GPU FSDP check](evidence/performance-fsdp8.json) of the ordinary MoE against the official TorchTitan measured a median step-time overhead of 1.95%. That check used GPU-resident synthetic data, with no data loading, validation or checkpoint saving. It does not cover multi-node runs or all methods. It is also not a complete proof of BF16 equivalence gradient by gradient. SMELT has 15 logical Cell calls and the MoE has 10. Thus the cost difference between the two models cannot be attributed directly to a performance regression of the same model implementation.

## Questions that a comparison answers

A mechanism ablation on the same base keeps unrelated mechanisms and the evaluation the same. A resource-matched recipe competition allows stated differences in structure or optimizer, and records the budgets and the substitutions. An allocation of data and depth under a total training compute also needs the actual work and the data exposure. Report the three kinds of question separately. Cross-method quality uses a common standard LM held-out set. The Bench and comparisons across tokenizers use byte-exact BPB (`bpb` of `evaluate_heldout`), and CE is kept as well. The training objective is reported separately. Later Ouro expected/entropy objectives must not be ranked together with terminal CE.

## Looped MoE Bench

[`eval/bench/tracks.json`](../eval/bench/tracks.json) freezes four tracks. S: C-grid scale, about 26M unique non-embedding parameters, a single-GPU entry track. M: P1 geometry, a screen at 5B labels. L: P1 geometry and the frozen P1 schedule, about 20.5B labels. P2: P2 geometry, about 200M active parameters, 5B labels, the main track of the Phase 1 study, with the other rules the same as M. The file is the authority for the values. The label budget of S is marked provisional, and a maintainer freezes it again before the first reference run. Each track fixes the data manifest and tokenizer, the sequence length, the global batch, the schedule, the optimizer settings, the seed rule and the learning rate (a fixed learning-rate grid for S and M, one learning rate per optimizer for L and P2). The evaluation is byte-exact BPB on the complete validation and test splits.

Budgets are judged with `compare_recipes` from this page. Unique non-embedding parameters, training matrix FLOPs per token and logical KV bytes must each be within 5% of the track's reference recipe. The accounting uses dense S² and no recompute. It counts model work, not the execution policy. For S and M, the LR is selected on the grid by validation BPB. L and P2 have one fixed LR per optimizer and no trials. Ranking uses test BPB only. Seed 42 is required, and other seeds are optional. The leaderboard gives the mean, the range and the number of seeds. GPU hours and throughput are only reported. They are not eligibility conditions.

[`moeloop.bench`](../src/moeloop/bench.py) reads one submission JSON and gives one verdict of `eligible`, `budget-mismatch`, `protocol-mismatch` and `missing-evidence`, with a list of all reasons. `leaderboard` ranks the eligible submissions by mean test BPB:

```python
import json
from pathlib import Path
from moeloop.bench import Submission, leaderboard, load_tracks, validate

submission = Submission.from_dict(json.loads(Path("submission.json").read_text()))
report = validate(submission, load_tracks()[submission.track])
print(report.verdict, report.reasons)
print(leaderboard([report])["rows"])
```

The snippet checks the declaration alone, so its verdict is never `eligible`. A submission is judged as a folder: `validate(submission, track, folder)`, or `scripts/bench.py validate --dir`, hashes the files the submission declares and compares their content with the declaration and the track, as [Benchmark submission folder](#benchmark-submission-folder) describes. It does not re-run training or evaluation and it does not judge quality; `scripts/bench.py` gives the command line (run, sweep, validate, leaderboard).

## Cost boundaries of state updates and multi-visit objectives

Fixed expected CE uses `fixed_expected_matrix_v1` (the matching document version when document pairs are given explicitly). It counts these items separately: the coda, head and gate forward of each visit, the matrix backward, and the extra head-projection recompute of the chunked CE checkpoint. The probability of the final gate does not count as an extra exit. Training still runs all visits. Under the current policy, non-matrix work such as EMA, RMS and entropy is not counted in the matrix FLOPs, and the output states this. This work is not treated as measured GPU cost.

For random depth and tail gradients, the training work is the expectation under `sampled_depth_matrix_v1` above. The costs of one step at p50 and p95 are listed separately. The nominal fixed-depth example is only one forward. It is not the actual work. Training progress records the actual r and min(k,r) of each microbatch and the real token counts separately. Parameter owners, the fixed inference KV cache and cache eligibility are reported independently. An uncached budget does not mean that a cache is implemented. Losses of different training objectives are not ranked directly. Use a common terminal LM held-out set with an explicit inference policy.

The [evidence for the new methods](evidence/recipes-v2.json) records the theoretical cost with the default dense-S² matrix definition of the frozen estimator, separate from the actual native GPU hours. The budget of #6, with its specific document statistics, keeps its original identity. This evidence is older than the sampled-depth cost model. In it, the `training_per_token` of Huginn is still `None` (the nominal record of `fixed_terminal_matrix_v1`). Evidence records are not rewritten. A new estimate now gives the expected values above.

## Benchmark submission folder

A Bench entry is a folder, and `python scripts/bench.py validate --dir FOLDER` judges the folder rather than a JSON file alone. `scripts/bench.py run` and `sweep` write it to `DIR/submission`:

| File | Content |
|---|---|
| `submission.json` | the entry: recipe, identities, per-seed and per-LR-trial values, `files` (the sha256 of every other file, by path), and for each seed and trial its `run` folder (`.` for the top) |
| `config.json` | the run's captured configuration: the spec as launched, launch topology, data manifest fingerprint, source commit and whether a tracked file differed from it, the training process's runtime, trainer package versions |
| `metrics.json` | completed updates, validation and test BPB, LR, GPU hours |
| `eval-validation.json`, `eval-test.json` | the evaluation output of each split, as written by `moeloop.evaluate` |
| `train.log` | the log of the last training launch |

The selected run (seed 42) sits at the top; on a track with an LR grid every other trial has the same files in `lr-<LR>/`. Checkpoints stay outside the folder: a seed names its checkpoint by `checkpoint_sha256`, the sha256 of the DCP `.metadata` file (tensor layout, not weights), and an optional `checkpoint_url`.

The check binds the declaration to these files. It recomputes the sha256 of every file and compares it with `files`: a declared file that is absent, a file that does not match its hash, or a file with no declared hash is `missing-evidence`. For each seed and trial it then compares the captured `config.json` with the declared recipe (seed, LR, schedule and number of updates, global batch, model, optimizer) and with the frozen track (launch topology against the global batch, data manifest, the pinned versions of the `train` extra of `pyproject.toml`: torch as recorded by the bench process and by the training process (a captured configuration without the training process's record is `missing-evidence`), TorchTitan as recorded by the bench process only; the declared source commit with no tracked file changed when the trial was launched, untracked files such as prepared data not counting), and the evaluation outputs and `metrics.json` with the declared final update and BPB, the recipe's inference depth, and the track's tokenizer and evaluation identity; any disagreement is `protocol-mismatch`. A submission is `eligible` only when every declared hash has its file, every file matches, and every captured configuration agrees.

The report lists under `checked` what was compared and under `unchecked` what it does not compare: that training this configuration produced these outputs, which weights the evaluation outputs scored (they carry no model identity), that the source commit exists (the check uses no network), the recorded source file hashes against the files of that commit (`source.dirty` is the state of the tracked files at launch), the content of `train.log` beyond its hash and runtime fields other than package versions, the TorchTitan version inside the training process, the declared `gpu_hours` total against the `gpu_hours` in each run's `metrics.json`, the declared `core` (bound to no file), the test evaluation output of an LR trial other than the selected one (checked by hash only), the data behind the manifest fingerprint, and each checkpoint, named by hash and URL. The pins are read from the checkout that runs the check, so changing the `train` extra changes what every stored entry is judged against. Anyone who edits a file and its declared hash together produces a consistent folder; the folder makes a claim inspectable and comparable, and only a rerun reproduces it.
