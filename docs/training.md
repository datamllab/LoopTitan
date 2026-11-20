# Training, data and model delivery

A researcher reads only plain Python recipes, JSON files and an explicit model and objective. TorchTitan supplies the outer training lifecycle. The public code does not need GitHub, MLflow or an internal deployment service. If you already have prepared data, go directly to [Run and resume from JSON](#run-and-resume-from-json), then to [Export, evaluate and generate](#export-evaluate-and-generate).

## Set up the environment

The verified versions are Python 3.12, Torch 2.14.0 and TorchTitan `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`. The `train` extra installs `torchtitan==0.3.0` from PyPI. This is the tag v0.3.0 of that commit. In a CUDA environment, first install Torch 2.14.0 for your hardware. Then install the project:

```sh
uv venv --python 3.12
. .venv/bin/activate
uv pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu130
uv pip install -e '.[data,hf,train]'
python -m moeloop.train --config tiny --help
```

For the CPU reference model and HF, the official CPU index and `.[data,hf]` are sufficient. Keep the training environment separate from the control environment. `uv run --frozen` is for the experiment control environment, which does not contain Torch. After you install the extra training dependencies, use the `python` of that environment directly.

The training entry point checks the upstream source files that it actually imports, not only the distribution name. On an offline cluster, you can put the same upstream source on `PYTHONPATH`. You then do not need to change a shared Python environment. A different upstream version needs a separate review of the adapter. The entry point does not silently fall back to a different model.

## Data preparation and mixtures

WikiText2 is an installation and engineering example. Do not use it to show that a method is valid, or as data for a final screen:

```sh
python -m moeloop.prepare --recipe wikitext2_gpt2 --output /data/demo
```

The recipe pins WikiText-2 raw and the GPT-2 tokenizer revision. It processes only the original non-empty lines. Each line is one document. The recipe adds BOS and EOS explicitly and does the next-token shift inside each document. This is an example of the workflow. It is not the context protocol of external WikiText PPL leaderboards.

P1 completed the data preparation from public raw DCLM. To replay it, use the published [ordered catalog of the 1024 raw files](evidence/p1-dclm-source.json) directly. Do not freeze a different list:

| Identity | P1 value |
|---|---|
| Raw data | `mlfoundations/dclm-baseline-1.0-parquet@817d6752765f6a41261085171dd546b104f60626` |
| tokenizer | `Qwen/Qwen3-30B-A3B@ae659febe817e4b3ebd7355f47792725801204c9`, vocab 151669 |
| Valid labels | train 24,000,067,249; validation 10,001,497; test 10,009,889 |
| Prepared manifest fingerprint | `e67a956d2fe51dd1784505a6f58d7790f397a09624330700c0ebaffcd344f152` |

Preparation reached the budget of every split at Parquet file 198. It scanned 19,624,997 raw documents and removed 10,762 duplicate documents. There were no empty documents. Deduplication is global, by raw document ID and by the SHA of the exact UTF-8 content. The split uses the first 8 bytes of the content hash as a big-endian integer mod 100: 0 = test, 1 = validation, all other values = train. Each split reaches its budget at a complete document boundary. **Keep the complete catalog of 1024 files.** You need to download only the prefix that preparation actually reads. If you cut the catalog to 198 files, the provenance and the manifest identity change.

First, download the pinned tokenizer to a new directory. **Get only the four source files below.** Additional source files, such as `config.json`, also go into the identity record, so do not mix them in:

```sh
export QWEN3_TOKENIZER_FOUR_FILES=/assets/qwen3-p1-tokenizer
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-30B-A3B", revision="ae659febe817e4b3ebd7355f47792725801204c9",
    allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    local_dir=os.environ["QWEN3_TOKENIZER_FOUR_FILES"],
)
PY
```

`DCLM_PARQUET` points to the local Parquet root directory that contains the relative paths of the catalog. `P1_DATA` is a new prepared directory. Prepare with the budget and shard size of this run:

```sh
export P1_DATA=/data/dclm-p1
python scripts/recipe_data.py --dclm-source docs/evidence/p1-dclm-source.json \
  --source-directory "$DCLM_PARQUET" --tokenizer "$QWEN3_TOKENIZER_FOUR_FILES" \
  --train-label-tokens 24000000000 --validation-label-tokens 10000000 \
  --test-label-tokens 10000000 --shard-tokens 128000000 --output "$P1_DATA"
cmp "$P1_DATA/manifest.json" docs/evidence/p1-data-manifest.json
```

This command does not download model weights. If you omit `--source-directory`, the command uses the Hugging Face cache, or downloads, as the frozen catalog specifies. Other data users can make their own list with `--freeze-dclm-source`. That list is then not the raw identity of this P1. The SHA of the original tokenizer.json is `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`. Preparation explicitly normalizes BOS = EOS = PAD = 151645. The SHA of the output tokenizer.json is `a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794`. For the full hashes of the source and output assets, see the [manifest](evidence/p1-data-manifest.json).

After an interruption, add `--resume` to the original command, with the same source list, tokenizer, budget and output directory. Preparation then continues from the last committed input shard. After completion, the token and index shards and the manifest stay immutable. If the data is not sufficient, do not loop a small set to imitate fresh labels. The public repository contains the metadata and the hash of each payload. It does not contain the prepared tokens, which are about 96GB.

Early engineering qualification also used a legacy importer for existing pre-tokenized DCLM documents. Its default fixture of 300M/1M/1M is still used for smoke tests. It does not supply the complete data for P1. `DCLM_PREPARED` points to the source manifest and payload in that old format:

```sh
export MOELOOP_DATA=/data/dclm-qwen3-v1
python scripts/recipe_data.py --packed-source "$DCLM_PREPARED/manifest.json" \
  --tokenizer "$TOKENIZER_DIR" --output "$MOELOOP_DATA"
```

The source format is existing per-document little-endian uint32 tokens, cumulative document-end offsets and a blake2b64 sidecar of the original text. The import pins the first two train chunks. It does not silently append a third chunk. It checks the size and SHA of the original files, the offsets, the token range and the EOS at the end of each document. The default is at least 300M train, 1M validation and 1M test valid shifted labels. Complete document boundaries make the counts slightly larger. This keeps the same content identity available for a later horizon of 268M positions.

The tokenizer is pinned to `Qwen/Qwen3-30B-A3B@ae659febe817e4b3ebd7355f47792725801204c9`, with an actual vocab_size=151669. The source sequences are EOS-only (151645). The import does not decode, does not re-tokenize and does not append a second EOS. It only prepends one explicit BOS=151645. PAD is also set to 151645. The original token body and EOS do not change. This BOS is not a BOS that the original Qwen configuration sets. The SHA of the source tokenizer assets and the SHA of the normalized output role/config are recorded separately. The vocabulary ID mapping does not change.

The SHA256 is calculated on the uint32 little-endian bytes of each original token document (including the original EOS). Deduplication is global. The first 8 bytes of the hash, as a big-endian integer mod 100, assign the split: 0 = test, 1 = validation, 2..99 = train. Each split stops taking documents when it reaches its budget at a whole-document boundary. The sidecar locates each document by the parent chunk identity and the document ordinal. It keeps the token SHA and the blake2b64 of the original text. These are not complete original HF/WARC document IDs.

The source is the public [DCLM baseline family](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0). Its public card records the license as CC-BY-4.0. The upstream download commit of the legacy copy above is unknown. Do not use the public HEAD of a different observation in its place. The parent manifest and the tokens/docs/dochash SHA identify its content revision. The new raw Parquet path uses the public revision and the per-file identity that `--freeze-dclm-source` records. The complete list, license, selection rules, actual labels, tokenizer and packing go into the manifest of each dataset. New data cannot silently inherit the identity of the old fixture.

The [public evidence summary](evidence/dclm-v1.json) of the historical 300M engineering fixture keeps the manifest fingerprint, the SHA of the two parent chunks, the hashes of the source and normalized tokenizer assets, the actual split counts and the token SHA isolation check. The actual valid labels are train 300,002,777, validation 1,000,401 and test 1,000,504. Among the 235,913 selected documents, the count of global token SHA duplicates and of cross-split duplicates is 0. The exact pre-tokenized binary chunks are not published yet, and the upstream download commit is still unknown. The summary identifies the actual content. For the training, resume and HF qualification, see the [validation record](validation.md).

FineWeb-Edu/GPT2 is a separate optional example. It is not a prerequisite for DCLM, and it is not a claim of better data:

```sh
python scripts/recipe_data.py --fineweb-edu --output "$FINEWEB_OUTPUT"
```

This explicit fallback pins `sample/10BT/000_00000.parquet` of FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. It downloads or caches the file and checks its size and SHA. To reuse a local file, replace `--fineweb-edu` with `--source-file "$FINEWEB_SHARD"`. If you give no source option, the command stops with an error. It no longer downloads FineWeb implicitly. It uses GPT2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The license is [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) plus the [CommonCrawl terms](https://commoncrawl.org/terms-of-use). The tokenizer is MIT. A change of source, tokenizer, split or budget needs a new fixture. The test split stays separate and is not used for tuning.

Local JSONL or text uses the same writer:

```sh
python -m moeloop.prepare --input /data/train.jsonl --validation /data/valid.jsonl \
  --test /data/test.jsonl \
  --tokenizer /assets/tokenizer --bos-token-id 1 --eos-token-id 2 \
  --output /data/prepared
```

The preparation stage writes immutable token and index shards, the tokenizer and the manifest. Explicit special IDs normalize the roles in the copied tokenizer. The original assets do not change. Training, export and generation keep the same BOS/EOS/PAD. The output directory must be a new directory.

The data packs the prediction pairs of each document into fixed-length blocks. Positions start from zero in each document segment of a block. Attention does not cross documents. When a long document continues across blocks, the next block does not keep the KV of the previous block and does not add BOS again. Tail padding uses a separate input mask, with labels=-100. PAD can have the same ID as EOS.

Each source first forms blocks of equal length. Then the loader selects blocks by weight. Thus the weights describe the input-token mixture, not the ratio of document counts. Define sources, splits and weights in a plain Python data configuration:

```python
from moeloop.data import PackedDataset
data = PackedDataset('/data/prepared/manifest.json', seq_len=1024,
                     source_weights={'web': 0.7, 'code': 0.3}, seed=42)
```

The writer function `prepare_dataset` accepts `{source: {split: texts}}` iterators. It does not load the tokens of the whole corpus into a Python list. At run time, mmap uses a bounded cache. At training start, rank 0 checks all hashes. The other ranks check the same manifest and the file sizes. After that, the result of the check is reused. All ranks must see the same immutable data namespace. Do not change the shards during a run.

## Change the research mechanisms

In `moeloop.recipes`, the nine selected paths each have a `*_spec()` function. It returns a `RecipeSpec` that you can inspect and change on a CPU. The functions include `huginn_spec()`, `loom_spec()`, `ouro_spec()`, `sandwich_spec()` and `relaxed_spec()`. For their JSON, region overrides and method adaptations, see [Recipes](recipes.md). Keep the model configuration of the user explicit. The actual vocabulary of DCLM is different from the default of the built-in small examples. A plain factory must read the manifest and pass the vocabulary to the spec.

`trainer_config(spec)` uses `run.activation_checkpointing` directly: `none` turns AC off, and `full_cells` uses the native Full AC. It rejects all other policies explicitly. `optimizer.zero_weight_decay_roles` uses a declared subset of `norm/embedding/lm_head/router`. An empty list means that all parameters use the given weight decay. The common adapter currently supports AdamW and Muon (`optimizer.name`), FP32 master/reduce, a core expert LR scope and no weight-decay compensation. Any other explicit setting causes an error. The adapter keeps no metadata that does not agree with the execution.

To select Muon, set `optimizer.name` to `"Muon"`. In `optimizer.muon`, give exactly the fields that `optim.MUON_FIELDS` lists. One field more or one field less causes an error. For the shortest form, see `muon_spec` in `tests/training/test_muon.py`. For the values that the Bench fixes, see `tracks.<track>.optimizers.fixed.Muon` in [`eval/bench/tracks.json`](../eval/bench/tracks.json). DistMuon updates the hidden matrices (see `MUON_PATTERN`). All other parameters are in separate AdamW parameter groups, which use the AdamW settings in `optimizer`. Only one place defines which parameters are hidden matrices: `MUON_PATTERN` in [`optim.py`](../src/moeloop/optim.py). For the reasons and limits, see [Architecture](architecture.md#5-training-dynamics-must-be-visible). Muon runs only on native FSDP on GPU. For the GPU validation, see #49.

The general method template defaults to context 2048, a global batch of 32768 packed positions, 512 successful updates, a fixed validation every 64 updates, and BF16 native experts. These recipes have a different scale from the actual P1 JSON below. `tiny` is a short installation demo of 100 successful updates. For the qualification of actual runs, see the validation page.

For example, save the code below as `my_recipes.py`. The public short adapter `trainer_config(spec)` returns a native Trainer config:

```python
# my_recipes.py
import os
from moeloop.data import PreparedManifest
from moeloop.recipes import smelt_spec, trainer_config

def experiment():
    manifest = PreparedManifest.load(os.environ["MOELOOP_DATA"])
    spec = smelt_spec(vocab_size=manifest.tokenizer["vocab_size"])
    return trainer_config(spec)
```

```sh
export MOELOOP_DATA=/data/dclm-qwen3-v1
torchrun --standalone --nproc-per-node=4 -m moeloop.train \
  --module my_recipes --config experiment --dump-folder /runs/experiment
```

The CLI uses the native Python ConfigManager. The model config sets the parameter shapes and the fixed inference defaults. The training recipe sets the training depth, the gradients and the objective. In the example, you can replace `smelt_spec` with any new spec. It still connects through the same `trainer_config(spec)`. The built-in factories of the same names are templates. With real data, you must match the vocabulary explicitly.

| Training rule | Actual behavior |
|---|---|
| Fixed or finite-set depth | Fixed depth, or one choice per update. The choice is the same on all ranks and microbatches of one update. |
| Huginn-inspired sampling | Samples r per microbatch from a Poisson-lognormal distribution and synchronizes it across ranks. Backpropagates only through the last `min(k,r)` stack visits. The anchor stays live. |
| LOOM full-BPTT | Fixed H>2. Global memory is reset to zero at each forward. Local memory is reset to zero at each visit. No detach and no optimizer step between segments. |
| Ouro Stage I | Runs all visits. Per-token expected CE−entropy. Trains the single head and the gate. Standard validation uses terminal CE separately. |

The random initial state is generated only once, before the recursion and before AC. The depth is located by the coordinates of the successful update and the microbatch. Validation uses an isolated RNG. The depth sequence of `uniform_update_v1` does not depend on the GPU count. `poisson_lognormal_micro_v1` samples per microbatch. The trainer uses the microbatch index of rank 0, so the sequence depends on the number of accumulation steps: global batch ÷ (local batch × GPU count). Partial sharing, layer loop, EMA and expected CE use fixed depth and full BPTT. A combination without an explicit gradient definition causes an error.

The main CE uses the valid labels of the whole update across all ranks as the denominator. aux is `local_micro_call_mean`: it is first averaged over the differentiable router calls of one microbatch, then over the logical microbatches of the whole update (global batch ÷ local batch). A change of microbatch geometry can change the non-additive aux, even if the global batch stays the same. aux does not depend on the GPU count when all of these conditions are true:

- The global batch and the per-GPU batch are both fixed.
- The depth rule does not depend on the GPU count (fixed depth or `uniform_update_v1`, not `poisson_lognormal_micro_v1`).
- The initial state is not drawn per rank. (A random latent initial state uses the RNG of each rank. Currently only `huginn_spec` does this, and its depth rule already excludes it.)

For the GPU validation, see #49. One of the preconditions is a data stream that does not depend on the GPU count. All ranks pack the same logical stream of blocks. The blocks of one update go to the ranks in contiguous slices: rank r takes blocks `[r·K, (r+1)·K)` of that update, where K is the number of blocks per rank. Thus runs with different GPU counts read the same blocks in the same order at each update. The data stream also fixes the blocks of each logical microbatch (rule name `logical_blocks_v1`; see `PackedDataset` in [`data.py`](../src/moeloop/data.py)). The forward pass does not update persistent router counts. AC recomputation does not commit the statistics twice.

Native grouped experts run in BF16. The arithmetic tolerance of the FP32 reference and the native GPU qualification are recorded separately. The training head and CE run in chunks. This avoids a forced full `[B,S,V]` logits tensor. CUDA graphs and model compile are off by default. DP/FSDP uses `partial_dtensor`. Attention that is independent per visit can use Full AC. The combination of Selective AC with this nested FSDP is rejected explicitly.

Ouro calculates probabilities and entropy with stable precision. An ignored label is excluded from both CE and entropy. Expected CE is reduced over the global valid labels of the whole update. aux is reduced separately. Training can keep the hidden state of each visit. It recomputes the head and CE in chunks through a non-reentrant checkpoint. This avoids storing the full-vocabulary intermediates of each visit. This head recomputation has a compute cost even when Cell AC is off. `fixed_expected_matrix_v1` shows it as a separate item. For Huginn random depth with tail gradients, the training FLOPs are an expectation over the declared depth distribution (`sampled_depth_matrix_v1`). The cost of one pass at the p50/p95 depths is in `flops.training_depth`. The nominal breakdown at the default depth is only one fixed forward. Neither of them is a measured cost.

When you change code or recipes in parallel, use a separate worktree for each change. Each experiment uses its own output root and its own frozen recipe/source identity. Data manifests and immutable shards can be reused read-only. Different worktrees do not write to the same checkpoints, metrics or attempts.

## Run and resume from JSON

`scripts/recipe_experiments.py` reads a plain `RecipeSpec` JSON and a prepared manifest. It calls the same native Trainer on GPUs that are already allocated. The example below uses the DCLM/Qwen3 data above. It uses one JSON of C to run 8 steps and then continue to 16 steps. It keeps the full schedule of 2048 steps. It is not the formal acceptance of the C panel. Other recipes must match the vocabulary and the batch/topology of the manifest explicitly.

```sh
export SPEC="$PWD/recipes/c_a0_r0.json"
export RUN=/runs/experiment
python scripts/recipe_experiments.py run --spec "$SPEC" \
  --manifest "$MOELOOP_DATA/manifest.json" --output "$RUN" --gpus 2 --stop 8
python scripts/recipe_experiments.py run --spec "$SPEC" \
  --manifest "$MOELOOP_DATA/manifest.json" --output "$RUN" --gpus 2 --stop 16 \
  --resume "$RUN/checkpoint/step-8"
```

`--gpus` is the total number of ranks on all nodes. On one node, the default is `--standalone`. For multiple nodes, run the same command on each node and set `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`. All nodes of one experiment share the same output directory and the same input identity. Concurrent experiments each use their own output root and rendezvous, and do not share ports. The launcher does not request GPUs.

The first run saves `spec.json`, `request.json` and a `source/` snapshot of the code that it runs. It then appends the process facts, `metrics.jsonl`, `outcome.json` and the full DCP. If updates already exist, you must pass `--resume` explicitly. Before the launch, the runner keeps the complete checkpoint in `protected/step-N`. This prevents the rolling keep 2 cleanup from deleting the resume point. Keep the original source code, recipe, runtime, data paths and topology. `--stop` is an absolute coordinate in successful updates. It does not reset the optimizer or the schedule. After a change of code or of experiment controls, use a new output directory and a new lineage.

After installation, the public factory is `moeloop.experiment:experiment`. Use it through `torchrun ... -m moeloop.train --module moeloop.experiment --config experiment`. Give plain file paths in `MOELOOP_EXPERIMENT_SPEC`, `MOELOOP_EXPERIMENT_MANIFEST` and `MOELOOP_EXPERIMENT_OUTPUT`. Set `MOELOOP_EXPERIMENT_STOP` and `MOELOOP_EXPERIMENT_RESUME` when you need them. The runner option `--installed` selects the installed package instead of the source snapshot. The complete wheel user journey of source97 passed the V3 acceptance in an actual run. For the exact source and the receipts, see the [validation record](validation.md).

Optional CVSM/MLflow only records existing commands and a limited set of artifacts; see [Experiment protocol](protocol.md). Data preparation, training, resume and HF use do not depend on them.

## Stage qualification of new methods

The complete v2 stage qualification passed; see the [actual evidence](evidence/recipes-v2.json). You can replay on GPU the four new paths (Huginn, LOOM, Ouro and LOOM×attention) and the attention×router four-arm grid. This stage entry point inherits the complete evidence of #6. It does not repeat the original pretraining:

```sh
python scripts/recipe_qualification.py --stage v2 \
  --manifest "$MOELOOP_DATA/manifest.json" \
  --v1-evidence "$V1_EVIDENCE" --v1-source "$V1_SOURCE" \
  --output "$MOELOOP_EVIDENCE" --gpus 2 --sharing-grid
python scripts/check_release.py --stage v2 --evidence "$MOELOOP_EVIDENCE"
```

`V1_EVIDENCE` is the accepted directory of complete durable artifacts. `V1_SOURCE` is the JSON of its original frozen P1 source identity. The public `docs/evidence/p1.json` is only a projection. It cannot replace the original bundle. The default is 512 steps per new method, 2048ctx and a global batch of 16 sequences. This is an engineering qualification. The short updates of the four-arm grid are not a quality ranking. For the actual resume, the HF scope and the final status of the new methods, see the validation record.

## P1 recipes and two-node training

The recipes of this run are [`recipes/p1_moe.json`](../recipes/p1_moe.json) and [`recipes/p1_smelt.json`](../recipes/p1_smelt.json). Load these JSON files directly. `p1_smelt_spec()` gives the F440 arm that was actually trained (measured-pairs convention, the same as P2). `recipe_anchor.py` without arguments still generates the dense-S² F424 template. It cannot replace the actual F440. Both arms completed 19,537 updates, each with **20,486,029,312 valid labels**: no padding, no repeated epoch and only one training seed, 42. For the model geometry, see [Recipes](recipes.md#p1-standard-moe--smelt). For the actual curves, the complete endpoint evaluation and the GPU-hours, see the [P1 evidence](evidence/p1.json).

The training source is pinned to `1bea4d6d2c539e3ae02517aaeede5f9ab68271a7`. TorchTitan is the pinned commit above. The actual environment is Python3.12.15, Torch2.14.0+cu130 and CUDA13.0. HF use ran on Transformers5.19.0. Later wiring of public artifacts and of the budget CLI does not change the model mathematics. It also does not mean that the release HEAD was trained again.

| Common run control | Value in this run |
|---|---|
| Hardware / mesh | Two nodes, 8×H200 per node; DP replicate2 × shard8 (HSDP) |
| Batch / context | local16, GA1, global256 sequences, 4096ctx; 1,048,576 positions/update |
| Optimization | AdamW, peak LR3e-4, betas(.9,.95), eps1e-8, WD.1, clip1 |
| Training arithmetic | BF16 compute, FP32 masters/reduce, native grouped experts, FlexAttention, head chunks8 |
| Objective | full BPTT, terminal CE, balance coeff.01, aux=`local_micro_call_mean` |
| WSD | warmup20 updates; stable end10000; cosine decay over 9537 updates to .1 of the peak LR; full horizon19537 |
| Other | finite epoch0, shuffle, num_workers0; no AC/compile/CUDA graphs; TP/EP/CP/PP=1 |

The WD of norm, embedding, lm_head and router is 0. Neither arm scales the expert LR. 1e-4 is an unused stability fallback. It is not an additional experiment. The local batch and GA are also comparison controls, because aux is not linear in the microbatch geometry.

All ranks must be able to read the same prepared data and run directory. On each of the two allocated nodes, run the command below from the checkout root. Set only `NODE_RANK` to 0 or 1. Use your own rendezvous address for `MASTER_ADDR` and `MASTER_PORT`. For the other arm, change `P1_SPEC` to `recipes/p1_smelt.json` and use a separate `P1_RUN`. Before the first complete training, run `unset MOELOOP_ANCHOR_STOP MOELOOP_ANCHOR_RESUME`. For short runs and continuations, set them as described below. The torchrun command does not change.

```sh
export P1_SPEC=recipes/p1_moe.json
export P1_RUN=/runs/p1-moe
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export MOELOOP_ANCHOR_SPEC="$P1_SPEC"
export MOELOOP_ANCHOR_MANIFEST="$P1_DATA/manifest.json"
export MOELOOP_ANCHOR_OUTPUT="$P1_RUN"
torchrun --nnodes=2 --nproc-per-node=8 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  -m moeloop.train --module recipe_anchor --config anchor \
  --training.local-batch-size 16 \
  --parallelism.data-parallel-replicate-degree 2 \
  --parallelism.data-parallel-shard-degree 8 --comm.init-timeout-seconds 1800
```

For a short check first, set `MOELOOP_ANCHOR_STOP=8` before the launch. It sets only the absolute stop coordinate of this launch. The full WSD of 19537 in the JSON stays. After the short run completes, protect the full DCP as the next section describes. Then remove the stop, or change it to a larger diagnostic coordinate, and continue. Keep the same controls on both arms. Do not add seeds. A fixed validation and a checkpoint run every 477 updates. The native validation of about 1M labels actually uses 1,048,576 labels per point. A stop point adds an endpoint evaluation. All of these are recorded separately from the complete 10M endpoint evaluation.

To explore a new shape only, you can use `recipe_anchor.py --attention-pairs-per-sequence 4180216.9229844734 --output /runs/new-pair`. It does not start training. It also does not generate the frozen source/runtime/data identity of this run automatically. The pair count comes from the actual document packing of the complete 20.486B training prefix. For the budget definition and the command that recalculates it, see [Costs and comparison](comparison.md#p1-and-the-within-document-attention-budget).

## Continue training and recorded results

The native checkpoint directory is `checkpoint/step-N`. The final save holds the full training state. When you use `moeloop.train` directly, a new launch with the same output directory loads the latest checkpoint. You can also set `--checkpoint.initial-load-path /runs/original/checkpoint/step-N` explicitly. The runner above always requires an explicit `--resume`. Keep the model, source code, data/tokenizer, world size, workers, optimizer/loss/AC and schedule horizon unchanged. You can extend the target stop step.

The P1 entry point keeps the two most recent rolling checkpoints. To use an early checkpoint as a resume point that others can review, **first let all training processes exit. Then copy the complete checkpoint to a location outside the rolling directory.** Then start the next segment. For example, run this once on shared storage:

```sh
mkdir -p "$P1_RUN/continuation"
test ! -e "$P1_RUN/continuation/step-8"
cp -a "$P1_RUN/checkpoint/step-8" "$P1_RUN/continuation/step-8"
export MOELOOP_ANCHOR_RESUME="$P1_RUN/continuation/step-8"
unset MOELOOP_ANCHOR_STOP
```

On both nodes, use the same protected path and run the torchrun command above again. Keep `MOELOOP_ANCHOR_RESUME` set for the continuation. A full DCP includes the metadata and the shards of all ranks. A copy of only the model, or of only one rank, is not a full-state resume. A public replay generates its own process and path lineage. The evidence keeps the source, the resume proof and the result identity of the original run.

Strict continuation in the first version requires `num_workers=0` and the same topology. The logical data stream itself does not depend on the GPU count. But the saved data state records the rank, the world size and the number of blocks per rank, so a change of GPU count is rejected. The checkpoint saves the RNG of each rank, the data cursor and partial buffer, the source/mixture state, the completed updates and the actual tokens. A separate generator for the DataLoader and an isolated validation RNG make sure that continuation does not use extra values of the training random sequence. A change of data stage, a change of topology or a model-only warm start must be done explicitly, as a new lineage. Do not use `strict=False` to hide a mismatch.

CE, aux, progress and routing counts are recorded at each successful update. Random depth also records the actual r and the number of gradient visits for each microbatch. Expected CE also records the training expected CE, the entropy and the exit distribution. Each training validation appends one record to `validation.jsonl` in the output directory. In addition to CE, the record has `label_bytes`, `ce_sum_text`, `bpb`, `byte_table_sha256` and `bpb_unavailable`. These fields use the same definition as `moeloop.evaluate` (see "Export, evaluate and generate" below). The numerator and the denominator are both summed over the ranks, so BPB does not depend on the GPU count. BPB is also logged as the native metric `validation_metrics/bpb`, in the same group as the TorchTitan metric `validation_metrics/loss`. In the JSONL, the field name is still `bpb`. Without a byte table, these four byte fields are None, `bpb_unavailable` gives the reason, and the metric is not logged. The implementation is `LoopValidator.validate` in [`titan.py`](../src/moeloop/titan.py).

Gradient norm, native throughput and peak memory appear only at the steps where TorchTitan actually samples its log. When `metrics.log_freq > 1`, the other rows omit the fields that were not observed. P1 used `log_freq=1`.

The output directory contains the model/resolved config, the continuation identity, metrics.jsonl, outcome.json and small facts for each attempt. Successful updates are counted separately from the native step, which increments early. Input tokens, valid supervised labels, padding positions and recursive work are each counted separately. Data exhaustion, or a checkpoint that was not written to disk, is not marked as complete.

Relative manifest, tokenizer and output paths are resolved against the directory where training starts. The training entry point saves absolute paths in the metadata. The manifest input can also be a directory that contains `manifest.json`. You can then export from a different working directory. Automatic discovery and identity checks still need access to the original prepared data and tokenizer files.

## Export, evaluate and generate

The commands below continue the 8→16-step example above. For other runs, use the step that was actually saved durably, and the matching context. Use new paths for the export directory and for the evaluation result files.

```sh
python -m moeloop.export --checkpoint "$RUN/checkpoint/step-16" \
  --tokenizer "$MOELOOP_DATA/tokenizer" --output "$RUN/hf-step16"
python -m moeloop.evaluate --model "$RUN/hf-step16" \
  --manifest "$MOELOOP_DATA/manifest.json" --seq-len 2048 --device cuda \
  --output "$RUN/heldout-step16.json"
python -m moeloop.generate --model "$RUN/hf-step16" --device cuda \
  --prompt 'The model' --max-new-tokens 32
```

Export reads only the model tensors of the DCP. It does not allocate an optimizer. It builds the owner on the meta device and then loads strictly. It saves the physical parameters once, together with the config, the tokenizer, the generation config, the relatively imported source code, LICENSE and the hash/provenance. The export directory must be new. The model configuration and the tokenizer can be discovered from the training metadata, or you can give them explicitly. A wrong tokenizer is rejected.

For held-out data from a single source, you can add `--document-loss-output /results/documents.json`. The same evaluation then saves the CE sum/count of each document, aligned by the document index in the split. It merges the parts of one document across blocks. The output contains the data, context and readout identities and the completeness flags. The aggregate result records the path and SHA of this file.

The result gives both CE and byte-exact BPB: `bpb = ce_sum_text / (ln 2 · label_bytes)`. The byte table comes from the tokenizer assets of the manifest ([`tokens.py`](../src/moeloop/tokens.py)). Labels with 0 bytes (EOS, PAD and all special or added tokens) go into neither `ce_sum_text` nor `label_bytes`. This agrees with `evaluate_bpb` of nanochat. If the CE of EOS stays in the numerator, the result is larger. `byte_table_sha256` identifies the byte table. If the manifest has no tokenizer assets, or the tokenizer is not a byte-level BPE, the byte fields are None, and `bpb_unavailable` gives the reason. For a tokenizer with NFC normalization (such as Qwen3), bytes are counted on the normalized text. For the exact definition, see the docstring of `evaluate_heldout`.

CORE (the DCLM v2 definition) runs on an exported HF directory. The user downloads the evaluation bundle manually and checks its hashes. The evaluator does not download it:

```sh
python -m moeloop.core --bundle-info   # official URL, hashes and download steps
python -m moeloop.core --model "$RUN/hf-step16" --bundle eval_bundle \
  --device cuda --output "$RUN/core-step16.json"
```

`core` in the result is the mean centered accuracy over 22 tasks. It has a value only when all tasks run and `--max-per-task` is not set. Otherwise it is None. `core_v1` uses the old random baseline. Use it to compare with nanochat and with DCLM numbers from before September 2025. For the prompts, the scoring rules and the known limits, see the docstring of [`core.py`](../src/moeloop/core.py).

Per-visit diagnostics run on the same export directory and prepared split. For each visit, they output the logit lens, the residual update, the routing load entropy and the overlap between visits. They also output the loss truncated to 1..K visits:

```sh
python -m moeloop.diagnostics --model "$RUN/hf-step16" \
  --manifest "$MOELOOP_DATA/manifest.json" --seq-len 2048 --device cuda \
  --output "$RUN/diagnostics-step16.json" --plot "$RUN/diagnostics-step16.svg"
```

`--plot` needs the `report` extra (matplotlib). Visits count from 0. The logit lens and the residuals need the state of a whole visit, that is, `order="stack"` or a core with only one Cell. Otherwise these two items are skipped, and the output gives the reason. BPB uses the same definition as `evaluate`. For the definition of each quantity, see the docstring of [`diagnostics.py`](../src/moeloop/diagnostics.py).

For P1, export the true `step-19537` of each arm. Evaluate the complete validation/test with the FP32 HF model. Do not pass `--max-batches`:

```sh
python -m moeloop.export --checkpoint "$P1_RUN/checkpoint/step-19537" \
  --tokenizer "$P1_DATA/tokenizer" --output "$P1_RUN/hf"
for split in validation test; do
  python -m moeloop.evaluate --model "$P1_RUN/hf" \
    --manifest "$P1_DATA/manifest.json" --split "$split" \
    --seq-len 4096 --batch-size 4 --device cuda:0 \
    --output "$P1_RUN/full-$split.json" \
    --document-loss-output "$P1_RUN/full-$split-documents.json"
done
```

The endpoint in the P1 evidence is CE. The evaluation at that time had no BPB and no CORE. For the later BPB and CORE measurements on the P1/P2 test split, see the [evidence](evidence/p1p2-test-bpb-core.json). The command above does not change. The output contains BPB when the manifest includes the tokenizer files. Otherwise `bpb_unavailable` gives the reason. The complete validation/test actually has 10,001,497 /10,009,889 labels. The document order and the label count of each document must be the same for both arms. Paired per-document differences help to understand held-out differences. They do not estimate the variance across training seeds. The native BF16 fixed validation curve and this complete FP32 endpoint evaluation are reported separately.

The HF Cache type stops at the wrapper. The KV of the core is independent for each logical occurrence, even when parameters are shared. One cache session has a fixed depth. It cannot add, in the middle of the session, a visit that has no history KV. The public model can compute gradients normally in full-prefix eval. The automatic cache is on only for inference without gradients.

The first paths of Huginn-inspired and LOOM use fixed depth and `use_cache=False`. Each forward rebuilds the initial state/memory. Huginn evaluation uses a deterministic zero latent. This is a declared eval adaptation. The trained Ouro gate is exported with the model. HF generation still uses terminal logits. HF `labels` computes terminal CE with the standard next-token shift. It does not compute the Stage I expected CE or entropy. For the qualification of independent consumers of the new paths, the validation record is authoritative.

The HF consumer supports the standard `model.resize_token_embeddings(len(tokenizer), mean_resizing=False)`. You can pass `pad_to_multiple_of`. The actual vocabulary size updates the HF config, the core config and the saved config together. The original embedding/head rows stay. New rows are initialized with `init_std`. The two parameter owners stay independent. Then save the model and the tokenizer together, and you can reload them offline. For native pretraining with a different tokenizer, rebuild the prepared manifest and make a new run identity.

Downstream evaluation reuses the existing lm-evaluation-harness:

```sh
uv pip install -e '.[eval]'
python -m lm_eval --model hf \
  --model_args pretrained=/models/experiment,trust_remote_code=True,add_bos_token=True \
  --tasks arc_easy --num_fewshot 0 --device cuda --batch_size 1
```

Real results must pin the task version, the complete evaluation scope and the checkpoint identity. `--limit 2` is only for an engineering connectivity check. Held-out CE uses the document/packing protocol of this project. Do not compare it directly with PPL from a different tokenizer/context processing. BPB normalizes by bytes, which removes the difference in tokenizer segmentation. But the context, packing and mask are still those of this protocol.

## Release on Hugging Face

`scripts/hf_release.py` publishes exported models with a generated model card; it never trains. A plan of single models binds each model to one run of a results file by its configuration path. The Phase 1 template is [`docs/release/phase1-models.example.json`](release/phase1-models.example.json): it fixes the 16 repository names, their titles, the short name and collection note of each model, the configuration of each model, the collection and the organisation card ([`docs/release/naturalscale-org-card.md`](release/naturalscale-org-card.md)). Copy it, and for each model fill in `export_dir` (the export above), `files` (every file to upload, with its SHA-256) and `evaluation` (the names, among `files`, of the `moeloop.evaluate` outputs for the validation and test splits). The evaluation files are uploaded with the model.

```sh
python scripts/hf_release.py my-plan.json --namespace my-org             # dry-run: files, hashes, cards, collection, organisation card
HF_TOKEN=... python scripts/hf_release.py my-plan.json --namespace my-org --upload
HF_TOKEN=... python scripts/hf_release.py my-plan.json --namespace my-org --cards       # cards only, to existing repositories
HF_TOKEN=... python scripts/hf_release.py my-plan.json --namespace my-org --collection  # collection title, description, notes
HF_TOKEN=... python scripts/hf_release.py my-plan.json --namespace my-org --org-card    # the organisation card
```

The script refuses the whole run if a file differs from its hash in the plan, or if an evaluation file's `bpb` differs from the run's `validation_bpb` or `test_bpb` at 6 decimals. Every number on the card comes from the results file and the configuration: test and validation BPB, the difference from the standard MoE under the same optimizer, and the three budgets against it. The card's figures under `assets/` (the banner with the model's layers, where the model stands among the designs of the plan, the budgets, and the README's destination capsules) use the drawing of the README figures, each in a light and a dark file and, for a phone, a narrow one; Hugging Face shows the file of its theme. The dry-run prints a model whose export is not filled in yet; `--upload` refuses it. With `--upload`, each model gets one commit to its own repository (private unless `--public`), and one collection in the namespace then lists the repositories in plan order. `--cards` renders the cards from the results file alone, so it needs no export: it uploads each `README.md` and its `assets/` to a repository that already exists, removes the figures of the earlier card (`banner.svg`, `budgets.svg`) and nothing else, and never creates a repository or changes its visibility. `--collection` applies the collection's title, description (at most 150 characters) and one note per model, in plan order. `--org-card` uploads the organisation card to the namespace's existing `README` Space. A plan of `pairs` (a looped model and its control, bound to an evidence file) works as before; the script's docstring describes both forms.
