# Validation record and support boundaries

2026-10-25. V1/P1 and V2 are inherited from their original sources. For V3, the two P2 arms, the C four-arm grid, the complete four-model downstream evaluation, the real installed-source journey and the complete HF endpoints have all run. **The final assembler and the independent `check_release.py --stage v3` both actually exited 0. The contents of their reports agree. scope=`v3_full_acceptance`.** For the original data, configurations, curves, processes and artifact identities, the [public V3 projection](evidence/recipes-v3.json), the [analysis that can be redrawn](evidence/v3-analysis.json) and the actual records of [#8](https://github.com/datamllab/LoopTitan/issues/8) are authoritative. The sections below state engineering qualification, scientific results and hardware performance separately.

## V3: second scale, combinations and delivery

### Pairing, data and execution sources

P1/P2 use the same public DCLM prefix, tokenizer, document isolation and order, 4096ctx, local16/global256/GA1, seed42, BF16 compute/FP32 master and reduce, the same terminal CE/aux rules and the full WSD. Each arm has **19,537 updates /20,486,029,312 fresh valid labels**, with no padding. Stop and resume do not change the original schedule horizon. The shapes are different, so the record does not claim that the initial weights are identical element by element.

| P2 configuration /main analysis budget | Ordinary MoE | SMELT |
|---|---:|---:|
| width /expert FFN width | 1280 /432 | 1056 /384 |
| physical layers /logical calls | 12 /12 | 12 (3+6+3) /18 |
| experts /top-k | 64 /8 | 88 /8 |
| query heads /KV heads /head dim | 10 /1 /128 | 12 /1 /88 |
| unique non-embedding parameters | 1,318,288,640 | 1,314,771,744 (−0.2668%) |
| total parameters, including separate embedding/head | 1,706,561,280 | 1,635,096,672 |
| document-aware training FLOPs /padded position | 2,573,863,361.534 | 2,515,805,029.899 (−2.2557%) |
| logical KV bytes at 4096ctx, batch1/BF16 | 25,165,824 | 25,952,256 (+3.1250%) |

The physical active non-embedding size of the baseline is 203.52M. For the candidate it is 146.92M, or 220.38M when summed over the logical calls. Do not claim that the two arms have equal active sizes. E88 is an adaptation to the common stack. Before any quality result existed, the shape was selected from at most 32 cost-only candidates by the residuals on the three budgets. The 5% engineering limit on each budget does not remove the limit that the residuals put on interpretation.

Each P2 arm actually used two nodes, **16 H200 /HSDP replicate2×shard8**. The two groups of 16 GPUs ran concurrently. This is not 32-rank model training. Both main runs had a real resume from a durable full DCP, which saved the model, the optimizer, the scheduler, the rank RNG and the cursor/progress. The complete curves and the independent HF results are kept with their actual sources. The 32-GPU collective connectivity check is not evidence of 32-rank model training or of its efficiency. EP/TP/PP and elastic-topology resume are not verified.

| Record | Frozen source |
|---|---|
| P1 scientific training | `1bea4d6d2c539e3ae02517aaeede5f9ab68271a7` |
| V2 qualification of new methods/combinations | `78d799dfa11c6be4130be63b252de9b91ae5aa8b` |
| P2 and C native training | `17d51e4df9e807f81c5d31301441daba9ce527b2` |
| Real source→package→installed journey | `97f4348678ae0213965c60d25a280777c878a51c` |
| Final read-only assembler/checker | `d94c268dd5d74c668f079e1b8d468c7a7989822f` |

The final gate and the documents have their own later sources. They do not claim to be the training commits above. The old P1 schema2 HF has no `labels_objective`. It is consumed read-only with the terminal semantics, because it is known to have no exit gate. The original config/weights/provenance were not migrated. The serialized config snapshot that the tests use is the same dictionary as the original JSON. The byte SHA of each file identifies that file. The two are not mixed.

### Native curves, complete HF evaluation and actual cost

| P2 fixed endpoint | Ordinary MoE | SMELT | Candidate − ordinary |
|---|---:|---:|---:|
| native fixed validation slice CE, 1,048,576 labels | 2.82716393 | 2.81225824 | −0.01490569 |
| complete 4096ctx HF validation CE, 10,001,497 labels /8167docs | 2.89231853 | 2.87802890 | −0.01428963 |
| complete 4096ctx HF test CE, 10,009,889 labels /8212docs | 2.88725790 | 2.87307719 | −0.01418072 |
| GPU-hours of all kept native processes | 135.01837 | 148.03375 | +9.6397% |

The complete HF evaluation is FP32, with terminal depth 1/2, no `max_batches` and `complete_split=true`. Its denominators and per-document identities match the frozen held-out data. The native curve has 46 points in total, each with about 1M labels. It cannot replace the complete 10M endpoint. The complete P1 test is 3.00927801/3.00338042 (Δ−0.00589759). These two same-D pairs show a lower CE for the candidate, and the difference is larger at P2. From two scales and a single seed, the record does not infer a scaling law, a general advantage of loops or a strict reproduction of the original papers.

The main compute coordinate is **document-aware analytical model compute**: `fixed_terminal_document_matrix_v1`, 4096ctx and the frozen mean attention pairs of 4,180,216.9229844734, multiplied by the actual padded_positions of each point. dense-S² is a secondary estimate. Logical compute is not FLOPs measured on the GPU. Common compute is interpolated only locally, inside the covered range. Analytical CE savings of P1, such as 6.0311%, come from the native validation slice of about 1M labels. They are not savings on the complete HF evaluation or in wall time.

GPU-hours count wall time × local GPU count for all kept native processes in the original train directory, including initialization, validation/checkpoint and failures. P2 has 12 receipts for each arm. The main-run subset, 133.23214/146.16926, does not replace 135.01837/148.03375. The foreground reservation overlaps with these and is not added. Independent HF/downstream work and idle environment time are listed separately. They are not interpreted as GPU busy time. The actual P1 values are 78.02286/103.43271, with the candidate 32.5672% higher. At P2 the candidate is 9.6397% higher. There is no conclusion of a hardware speed-up.

### Continuous performance windows

The table below shows real P2 update windows on the same hardware/software, at 4096ctx/global256 and without Cell AC: warmup steps 5–9, continuous measurement 10–29. Valid labels/s is a global value. The peak is the local allocator value that global rank0 observed. It is not taken as the maximum over all ranks. The full process cost is counted separately.

| P2 | median step (min–max) s | median valid labels/s | peak allocated /reserved GiB |
|---|---:|---:|---:|
| MoE | 1.57763 (1.52090–1.74038) | 664,652 | 102.46 /111.93 |
| SMELT | 1.69898 (1.66362–2.06691) | 617,182 | 123.71 /133.68 |

A representative window for partial owners is B+A of C (a1_r0): 4H200, 2048ctx/global16, no AC, steps1–5 warmup and 6–25 measured. The median step is **0.127723s** (0.123826–0.334440). The median valid labels/s is **256,555**, the aggregate is **237,596**, and the rank0 peak allocated/reserved is **4.997071/6.181641GiB**. For the actual window and the source/metrics identity, see the [V3 analysis projection](evidence/v3-analysis.json). The fully shared C-B is not used in its place. These are measured geometries. Do not extrapolate them to EP/TP, to 32-GPU model efficiency or to any larger model. Before larger pretraining, first check the activation/head memory, the data capacity, save and resume, and the throughput. Then decide on AC and parallelism.

### C: owner four-arm grid on a fixed expert pool

The common settings are d512/F1024/E8/top2, 2 banks×4 visits, heads8/kv2, 2048ctx/global32768 positions, LR3e-4 and warmup64/cosine to 0.1peak. Each arm uses seed42, 2048 updates and 67,108,864 valid labels. The CE here is the common native endpoint definition. It is not a complete 10M HF evaluation.

| Arm | Final CE | unique non-embedding | Core increase relative to B | native GPU-hours |
|---|---:|---:|---:|---:|
| B | 5.16462517 | 26,487,296 | 0 | 0.44051 |
| B+R | 5.15252876 | 26,511,872 | +24,576 (0.0928%) | 0.43174 |
| B+A | 5.10578394 | 30,419,456 | +3,932,160 (14.8457%) | 0.45228 |
| B+A+R | 5.10317230 | 30,444,032 | +3,956,736 (14.9385%) | 0.45974 |

The percentages use the core non-embedding size of B, 26,486,784, as the denominator. Embedding/head total 155,309,056. They are not used to dilute the increase. `I_AR=CE_AR−CE_A−CE_R+CE_B=+0.00948477`. The combined arm is only 0.00261164 lower than A. The four arms are not iso-parameter, and they are not a complete Foil×LOOM. They describe only this short panel. They do not announce a final recipe. The dense matrix cost of C is listed separately. It is not mixed with the document-aware coordinates of P1/P2.

### Complete downstream evaluation of four models, and installation

`lm_eval[hf]==0.4.13`, 0-shot, FP32, terminal depth MoE1/SMELT2, the complete ARC-Easy test2376 /HellaSwag validation10042. `auto:4` actually selected batch64. The native `task_hashes` and the harness source identity are equal for the four models. The task YAML/helpers, the data revisions and the sample logs are all kept (the sample logs are not public). `--limit` was not used as a substitute:

| Model | ARC acc /acc_norm (%) | HellaSwag acc /acc_norm (%) |
|---|---:|---:|
| P1 MoE | 54.76 /50.08 | 33.52 /39.63 |
| P1 SMELT | 56.23 /49.96 | 34.03 /40.43 |
| P2 MoE | 58.63 /52.15 | 36.39 /45.32 |
| P2 SMELT | 58.38 /52.53 | 37.11 /46.39 |

On ARC, P1 improves the raw acc, and the norm decreases slightly. P2 is the opposite. HellaSwag improves on both metrics at both scales. All metrics are kept. The single-seed/document stderr does not give the variance across training seeds. The actual scoring used a foreground reservation of 0.238624 GPU-hours in total. The read-only merge, 0.007671, is counted separately.

The real journey was source97 → sdist → a wheel rebuilt from the sdist, followed by a Torch-free base import outside the checkout. It then completed prepare on a small public WikiText2 subset, actual train4 → full-state resume8, full DCP → HF, a complete 2057-label demo validation and an independent offline deterministic generation. Training used 4096 valid labels. This is an actual user journey, not the scientific P1/P2/C. Three failures are kept: the original uv path was missing, the native parent environment had no TT metadata, and the HF-only dependency versions were judged incorrectly. The final pass used a normally built TT0.3.0 at pin086 and a real dependency environment with 5.20.2. No dist-info was faked, and no substitute weights were built. The current installation follows the public pin 5.20.2. The historical native/HF-only records with 5.19.0 keep their original identity. The record does not claim in general that the environments are the same.

The final acceptance tool source is `d94c268dd5d74c668f079e1b8d468c7a7989822f`. The actual assembler and the independent CLI both exited 0. The public analysis binds the original analysis SHA `4283eaf816c93ea87a4c65a825c0013d46d476f8b2d77692d561636e43699a7f`. The public JSON/CSV/figures are projections that keep the values and identities. A new verification still needs the original data/DCP/HF, or a complete run that you rebuild yourself. The projection does not replace the raw evidence. The old strict full-network BF16 gradient failure is still described below in [Performance and mathematical qualification of the public implementation](#performance-and-mathematical-qualification-of-the-public-implementation). A stage pass does not mean full-network parity.

## Qualification of new training dynamics and combinations

2026-10-24. **The actual CLI passed completely, scope=`v2_full_acceptance`.** The [machine-readable evidence](evidence/recipes-v2.json) and its four CSV files keep the executed source code, the recipe/runtime, the actual progress, the native validation, the independent HF results, the complete resume and the actual cost. The frozen training/acceptance source is `78d799dfa11c6be4130be63b252de9b91ae5aa8b`. Later release commits cannot claim to be the training commit. The complete P1 of #6 is inherited by the SHA of its original source/acceptance files. It was not retrained.

All four paths use real public DCLM, the Qwen3 vocabulary of 151669, dim128/expert hidden256, E4/top2, sharded FSDP on two H200 GPUs (replicate1/shard2), context2048, global16/local8/GA1, BF16 compute/FP32 master and reduce, and no Cell AC. Each path has **512 successful updates and 16,777,216 valid labels**, with only one training seed, 42.

| Path | Actual irreducible semantics | native final CE | HF small-sample CE | Max error after resume, model/optimizer |
|---|---|---:|---:|---|
| huginn | Random latent; depth 1–19 over 512 microbatches, tail2 with a live anchor | 7.093894 | 6.550015 | 0 / 0 |
| loom | M2/H3 dual EMA, re-injection, independent router, full BPTT | 6.772820 | 6.237102 | 0 / 0 |
| ouro | R3 per-token expected CE−entropy; shared gate/head actually updated | 6.853518 | 6.302900 | 0 / 0 |
| loom_attention | Full LOOM state and independent attention per visit, M2/H3 | 6.761012 | 6.233232 | 0 / 0 |

These are **engineering learning and delivery trajectories of different objectives/structures. They are not a fair quality ranking or a reproduction of the results of the original papers.** The native validation uses a fixed 2048ctx and 65536 valid labels each time, with 9 points in total. HF uses 256ctx, 1024 valid labels and `complete_split=false`. The two sets of values are not native/HF loss parity.

- All four paths are exported from a real full DCP and loaded by an independent offline `python -I` consumer. Huginn/LOOM/hybrid generate with fixed depth and without cache. The zero-init evaluation of Huginn is a public adaptation. Ouro fixed-terminal generation passed with cache and with recomputation. HF labels compute only the standard terminal CE.
- For each path, 8 continuous steps and 4 + resume 4 steps gave a maximum model/optimizer error of 0. The RNG of both ranks, the data cursor, the completed counts and discrete states such as the scheduler agree. Ouro also read the real DCP Adam moments and confirmed that the gate, the head and the backbone all update. The distribution entropy and the per-visit probabilities/CE are actually recorded.
- The attention×router four-arm grid ran 8 real updates per arm. The arms share the expert pool, the call graph and the normalization rule, and change only the owner switches. The configuration and the cost can be rebuilt. This 8-step record is a run qualification. For the formal C results of #8, see above.
- The native processes of the four formal 512-step main runs total 0.44319 GPU-hours. With their resume forks and the four-arm grid, the total is **1.38247 GPU-hours (20 native receipts, no failures)**. The definition is native process wall time × actual GPU count, including initialization/validation/saving. It does not include HF post-processing, early smoke tests or idle environment time. It is not GPU busy time.

The new equations have independent FP32/closed-form gradient oracles. An actual two-process CPU FSDP run covers the expected head, EMA+FullAC and the backpropagation of the random-latent tail. The GPU evidence covers the two-rank execution configuration above. It does not extend to a claim that the new methods passed on multiple nodes or on any parallel matrix. The public comparator keeps the total training FLOPs of random-depth training as unknown. Expected CE lists the head checkpoint recomputation separately. Do not claim that the hazard gives an early exit from the forward pass.

### Semantic regression for parameter changes

After an independent re-review, the CPU checks below were added. They compare the complete logits, CE/aux and the gradients of all parameters. They are recorded separately from the GPU training evidence above:

| Path | Non-default conditions covered | Independent reference |
|---|---|---|
| LOOM | M1/H1, M3/H4; beta 0/.25/.8, different lambda; independent attention, GQA/QK norm, always-on FFN, heterogeneous outer regions and padding | Closed-form weighted history, which covers dual memory, re-injection and reset. Long-history combinations also use FP64 to exclude rounding from the summation order; the others use FP32 |
| Huginn-inspired | M3, actual r=1/3/5, tail=1/2/full; non-default latent distribution, injection scale and heterogeneous outer regions | Explicit unrolling, then the computation graph is cut. Keeps a live anchor. Independent of the no-grad execution segment of the production code |
| Actual two-rank FSDP | Ouro expected CE×FullAC; LOOM/Ouro M3/H4, independent attention, a different expert sharding axis with E1<shard2 | Gradient comparison of all parameters, two microbatches, two updates; covers repeated calls/recomputation |
| HF vocabulary | Grow, align to a multiple, shrink; plain recursion/LOOM/Ouro; original weights kept and new rows initialized | Config and logits agree after save/load. Also an offline `python -I` AutoClass consumer |

The re-review found and fixed two problems. The standard HF `resize_token_embeddings()` updated only the outer vocabulary and did not update the core/nested config. With `mean_resizing=False`, new rows did not use `init_std`. The fix changes only the HF consumer interface. The old training did not resize the vocabulary. The original training equations, configuration and DCP, and the HF export of that time, keep their original identity. GPU training was not repeated for this fix. The new regressions can reject actual temporary mutations, such as an average of the dual memory, a detached anchor or a beta hard-coded to its default.

These checks support parameter changes inside the existing mechanisms. They do not mean that the Cartesian product of all switches is qualified. A change of state, gradient or objective needs the matching independent oracle, kept or added. When you change the target training scale, precision or topology, first do a short GPU run at the target geometry. Then start long comparisons.

## P1: actual configuration, data and complete training

To replay this configuration directly, read the [MoE JSON](../recipes/p1_moe.json) and the [SMELT JSON](../recipes/p1_smelt.json). Do not regenerate it from the defaults of the general factory. The training source code is frozen at `1bea4d6d2c539e3ae02517aaeede5f9ab68271a7`. Later documentation and CLI release commits cannot rewrite the source of this training. TorchTitan is pinned to `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`. The actual environment is Python3.12.15, Torch2.14.0+cu130, CUDA13.0 and Transformers5.19.0.

| P1 configuration or analysis budget | Ordinary MoE | SMELT |
|---|---:|---:|
| width / expert FFN width | 672 / 544 | 576 / 440 |
| physical layers / logical Cell calls | 10 / 10 | 10 (2+5+3) / 15 |
| experts / top-k | 64 / 8 | 96 / 8 |
| attention heads / KV heads / head dim | 7 / 1 / 96 | 9 / 1 / 64 |
| unique non-embedding parameters | 712,656,672 | 737,845,056 (+3.5344%) |
| total parameters, including separate embedding/head | 916,499,808 | 912,567,744 |
| training FLOPs / padded position | 1,284,757,348.671 | 1,248,742,044.863 (−2.8033%) |
| logical KV bytes / 4096ctx, batch1/BF16 | 15,728,640 | 15,728,640 |

The "100M-active tier" refers to the active non-embedding body of the ordinary MoE, which is about 98.5M. It is not the total parameter count. In SMELT, the middle 5 layers share whole Cells and run twice. The two core residuals are each multiplied by 1/2. The outer regions are not scaled. All three budgets are inside the 5% pairing range that was declared in advance. The FFN width was selected from 12 candidates declared in advance, by the budget residuals, not by quality. These are `fixed_terminal_document_matrix_v1` analytical budgets. Actual kernel time, memory and GPU-hours are reported separately. Analytical compute does not replace measured resources.

The [public source list](evidence/p1-dclm-source.json) pins the 1024 ordered metadata files of DCLM revision `817d6752765f6a41261085171dd546b104f60626`. When preparation reached the budgets, it had actually read the first 198 files. It did not download or train on the complete dataset. The [actual manifest](evidence/p1-data-manifest.json) identifies train **24,000,067,249**, validation **10,001,497** and test **10,009,889** valid labels, with the fingerprint `e67a956d2fe51dd1784505a6f58d7790f397a09624330700c0ebaffcd344f152`. The data is deduplicated globally by original document identity and exact content, and whole documents are assigned to splits. The Qwen3 tokenizer is pinned to `ae659febe817e4b3ebd7355f47792725801204c9`, with vocabulary 151669. The hashes of the source tokenizer and of the normalized prepared assets are kept separately.

The [complete prefix statistics](evidence/p1-data-statistics.json) cover the 5,001,472 blocks planned for training: 20,486,029,312 labels, no padding and enough fresh data for each rank. At 4096ctx and world16, the mean document-causal query/key pairs are **4,180,216.9229844734 / sequence**, including the diagonal. This measured statistic enters the budget pairing. Dense-S² or an early 20M-label small sample does not replace it.

Formal training ran on two nodes with 8 H200 GPUs each, as **HSDP replicate2 × shard8**: local batch16, GA1, global256 sequences/update, 4096ctx; TP/EP/CP/PP=1. Compute is BF16 and master/reduce is FP32, with native grouped experts/FlexAttention, head chunks8, no AC and no compile. Both arms use the same AdamW/LR3e-4 and clip1. The full WSD is 20 steps of warmup, stable up to step 10000, and 9537 steps of cosine decay to 0.1×. Finite train epoch0, shuffle, `num_workers=0`. The actual stop/resume does not reset the horizon, the data or the optimizer, and adds no training seeds.

Each arm actually passed the common stop points 8, 95, 954, 4769, 10000 and 19537. The actual resume lineage is kept. The native finite validation has 46 points in total, each with **1,048,576 labels**. A successful finish adds a final point outside the period. A stop step that already had a periodic validation is not repeated. The final CE is MoE **2.9485197** and SMELT **2.9438329**. This is the fixed slice of about 1M labels. It is separate from the complete validation/test of about 10M in the independent HF evaluation below. The same-D curves and the curves on common analytical compute are reported separately. Common compute is interpolated only inside the range that was actually covered.

### P1 results and interpretation

The [public P1 evidence](evidence/p1.json) collects the frozen configuration, the [complete curves](evidence/p1-native-curves.svg), the [late-stage zoom](evidence/p1-native-late-curves.svg), the evaluation and the resource definitions. The complete HF endpoint evaluation uses 4096ctx and untruncated splits. The per-document identities and the valid label counts of the two arms agree:

| Complete HF evaluation | labels / documents | Ordinary MoE CE | SMELT CE | ΔCE (SMELT−MoE) |
|---|---:|---:|---:|---:|
| validation | 10,001,497 / 8,167 | 3.01446893 | 3.00836268 | −0.00610625 |
| test | 10,009,889 / 8,212 | 3.00927801 | 3.00338042 | −0.00589759 |

From 6,002,049,024 training labels onward, the SMELT CE is lower at 31 consecutive native validation points on the same data. The end of the common analytical compute range is **2.5581766134191145×10¹⁹ FLOPs**. At that point, the ordinary MoE CE, interpolated between the adjacent measured points, is **2.95114454**. SMELT is **2.94383287**. The difference is **−0.00731166**. The compute needed to reach the same native CE is compared only inside the range that both arms observed. The analytical compute saving is **6.0311%**. The main result does not use endpoint estimates of each arm outside the common range. It does not mix native CE of about 1M labels with complete HF CE.

This result is classified as **`clear_positive`, only for this public DCLM, the frozen model pair and seed42**: the complete val/test and the late curve agree in direction. It does not remove the budget residual of 3.5344% more unique non-embedding parameters in SMELT. It does not estimate the variance across training seeds. It does not prove a causal gain for all MoE models, scales or data. Per-document pairing is not a significance test of training randomness.

The actual native training resources are **78.02286 GPU-hours** for the ordinary MoE and **103.43271 GPU-hours** for SMELT, a total of **181.45557 GPU-hours**. They count real process wall time × local GPU count, including initialization, native validation and saving. They do not include independent HF post-processing or idle environment time. The actual training cost of SMELT is **32.5672%** higher. This round has **no GPU-time gain**. The 6.0311% above is a quality comparison on analytical compute. It is not a speed-up ratio.

This run adapts the core sharing structure of SMELT and its idea of fair comparison to public DCLM and a common model stack. It does not claim to reproduce the original corpus, the complete engineering, or savings in absolute loss or on the scaling frontier.

## Distributed training, full-state resume and model delivery

With the selected P1 source/data/HSDP16 configuration, a real run of 8 continuous steps and a 4+4 resume were compared. The comparison covered **103 model tensors and 309 optimizer tensors (1545 state keys), and the maximum error was 0 for all of them**. The RNG, progress, cursor/source and scheduler/scalar/hyperparameters of the 16 training ranks and the 16 data ranks agree exactly. The two 20.486B main runs also actually resumed to the end from their own protected full DCP. Thus the verification covers more than a model-only warm start.

A historical 8-rank FSDP resume at the target shape (source `33f246b`) and a two-node 16-rank full FSDP resume (`bae23d7`) also compared the same 103/309 tensors. The error of the model, the optimizer and the continuation state of each rank was 0. They use their own data, batch and source, and are reused as verified execution paths. Throughput at different local batches is not called strict strong scaling. For the scope of the earlier four-method qualification, see below.

Final delivery requires an HF export from the real DCP at step 19,537. It also requires a check of fixed-depth cache/greedy in an independent consumer that runs with `python -I`, offline and without an import of the training library. The complete validation/test evaluation uses 4096ctx and complete splits. It saves the loss sum/count of each document, and pairs them by the same document identity and valid labels. The native validation of about 1M labels, the historical 256ctx small sample and the complete HF evaluation of about 10M each report their own denominator. Both arms completed the final HF export, the independent offline cache/greedy check and the complete validation/test. All commands returned 0. The label/document counts of the complete held-out evaluation and the per-document sidecars were checked.

The final stage checker uses the fix in `ad1a21f` for the identity boundary of JSON tuple/list round trips. The training source is still frozen at `1bea4d6`. The fix does not change training, the model mathematics or the original numerical tolerances. The related real CPU DCP regression gave 39 passed. Before release, a logging problem was also fixed: steps without a log reused the previous observation of gradient norm/throughput/memory. Now fields that are not sampled are omitted for that step. CE/aux/progress are still recorded at each step. P1 actually used `log_freq=1`, so the frozen evidence is not affected. Its source is still `1bea4d6`. The general trainer adapter also applies the declared AC and zero-WD roles in one consistent way, and rejects common-stack settings that it does not support. The existing P1 configuration has no AC, four zero-WD roles and FP32 master/reduce. The original frozen artifacts stay as they are.

Data rebuild, two-node native training, strict resume, DCP→HF and complete evaluation all use the [public training entry points](training.md). They do not depend on a specific scheduler or on extra helpers. The public metadata identifies the actual artifacts. It does not contain the 96GB token payload or the training checkpoints. If you rebuild, check the manifest/content hashes and generate your own actual run records.

This evidence covers the recorded model tiers, execution settings and topology. For a larger model or more nodes, first run a short smoke test, a save/resume check and a throughput check with the same source and data semantics. Then start complete training. A pass on 16 GPUs does not show that any future scale needs no adaptation. EP/TP/CP/PP, elastic-topology resume and arbitrary compile combinations are still outside the verified scope.

## Performance and mathematical qualification of the public implementation

The [public reference script](../scripts/recipe_benchmark.py) runs on one H200 with the target shape of the ordinary MoE, BF16, batch2/context4096, head chunks8 and no AC. It checks the same parameter/mathematical configuration and natural routing. After 5 warmup steps, it measures 20 complete steps. Each step includes the chunked head/CE, backpropagation and the same AdamW. The median of this library is **147.04ms**. The public TorchTitan Qwen3 path is **141.43ms**. The ratio is **1.0397**. It does not reach the predefined `>1.20` investigation line. Initialization, data I/O, checkpoint/eval and distributed communication are not in this measurement.

In this record, the original complete BF16 deep-gradient check is still **`correctness.status="failed"`**. The errors and the 3% threshold are kept as they were, and the natural top-k divergence is recorded. With explicit `--component-math`, the same-input Flex/SDPA, the FP32 router and the official HF experts, with all their component derivative checks, passed. There is also a complete-update oracle on a small official HF FP32 model. The qualification scope is `component_equivalence_with_observed_end_to_end_bf16_route_divergence`. It does not claim complete-network gradient parity. The performance measurement did not fix the routing.

The [8-GPU performance summary](evidence/performance-fsdp8.json) comes from a real FSDP comparison on 8 H200 GPUs in one node. At the time of the run, the published [213-line measurement script](../scripts/recipe_benchmark_fsdp8.py) was byte-identical to the executed file (SHA256 `6e363bb2000ff9c863ed00cb385d2be83fc43a836d8788983e958c1114f95ab4` in the summary). The current file has changed since then and is not byte-identical to it. Both arms use the ordinary P1 geometry, the same initial weights, local batch16/context4096, accumulation2/global batch256, head chunks8, no AC, FP32 master/BF16 FSDP/FP32 reduce, the same fused AdamW and clip1. After 1 safe update and 3 warmup updates, each arm measured 10 complete updates. Each step takes the maximum synchronized wall time of the 8 ranks. The barrier is outside the timing.

| One-node 8-GPU FSDP | median step | global valid labels/s | peak allocated GiB |
|---|---:|---:|---:|
| This library | 1.650406s | 635,344 | 77.61 |
| Official TorchTitan Qwen3 at the pinned commit | 1.618800s | 647,749 | 76.81 |

The step ratio is **1.01952**. It does not reach the original `>1.20` investigation line. Timing uses GPU-resident synthetic tokens, one causal document per row without padding, and natural routing. Data I/O, mask construction, compile/warmup and checkpoint/eval are not timed. In the first untimed complete update, the gradients and the updated values of all parameters are finite, and CE is 12.06318665/12.06318092. The loss/grad norm of the 10 timed updates are finite.

Each side keeps its own native FSDP grouping/reshard, Flex autotune and router implementation. Actual hooks record that on both sides the router input is FP32 and the materialized gate weight is BF16. The official scores are FP32. The output dtype alone cannot prove the same precision for each operator. The mathematical basis is still the single-GPU component checks and the HF FP32 oracle above. The original strict BF16 failure is kept. This run adds distributed timing and run safety checks. The ratio covers only 8 GPUs on one node. It is not an official two-node comparison, complete distributed BF16 gradient equivalence or complete scientific P1 acceptance.

In a training environment with the software versions listed in the summary and the pinned TorchTitan checkout, you can rerun the measurement with the commands below. First, extract the two model configurations of this run from the public summary. This performance measurement uses the ordinary P1 geometry. But its synthetic data and 8-GPU FSDP/GA2 are recorded separately from the real data and 16-GPU HSDP/GA1 of formal P1.

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts:${TORCHTITAN_SOURCE:?set pinned checkout root}"
python - <<'PY'
import json
from pathlib import Path
record = json.loads(Path("docs/evidence/performance-fsdp8.json").read_text())
out = Path("/tmp/moeloop-perf")
out.mkdir(exist_ok=True)
for name, config in (("p1", record["model_config"]),
                     ("hf", record["mathematics_evidence"]["hf_fp32"]["model_config"])):
    (out / f"{name}.json").write_text(json.dumps(config))
PY
python scripts/recipe_benchmark.py --model-config /tmp/moeloop-perf/p1.json \
  --mode benchmark --component-math --batch-size 2 --seq-len 4096 --head-chunks 8 \
  --output /tmp/moeloop-perf/math.json
python scripts/recipe_benchmark.py --model-config /tmp/moeloop-perf/hf.json \
  --mode sanity --reference transformers --candidate plain --attention sdpa \
  --device cpu --dtype float32 --seq-len 8 --head-chunks 2 \
  --output /tmp/moeloop-perf/hf-math.json
torchrun --standalone --nproc-per-node=8 scripts/recipe_benchmark_fsdp8.py \
  --math-report /tmp/moeloop-perf/math.json --hf-report /tmp/moeloop-perf/hf-math.json \
  --local-batch 16 --accumulation 2 --warmup 3 --steps 10 \
  --output /tmp/moeloop-perf/fsdp8.json
```

## Historical engineering qualification: four-method DCLM short runs

The [historical machine-readable evidence](evidence/recipes-v1.json) keeps the request/spec, the source and artifact hashes, the progress, the held-out results, the resume and the independent HF results of that time. At that time, the `check_release.py --stage v1` CLI returned 0, and the verifier reported passed. This batch of training is frozen at `1aad50570c58c57ddb6f8770d30f78637bc75865`. The hashes of its 20 source/training files identify only that snapshot. The current model, data preparation and training code have changed since then. Do not claim that they are byte-identical to that snapshot. The old JSON stays unchanged as historical engineering evidence. The scope of reuse is decided separately, by the paths that a change touches. The current PRs/evidence record the final CI, the independent review and the current P1 acceptance. The later source `79bbc5f` also actually completed 8 updates of Foil on 2 ranks, with Full AC and DCP→independent HF/cache/greedy. This adds execution evidence for the newer shared operators. It does not count retroactively as a 512-step rerun of each of the four methods on the current source.

All four methods completed **512 successful updates, context2048, global batch16 (32768 positions/update)** on **2 H200 GPUs, 2-rank sharded FSDP**. For each method, the cumulative packed positions, valid inputs and valid labels are all **16,777,216**. `dp_replicate=1/dp_shard=2`, TP/EP/CP/PP=1; Torch 2.14.0+cu130, CUDA13.0, Python3.12.15, TorchTitan pin `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`. Compute parameters are BF16. Master parameters/gradient reduce are FP32. Foil uses Full AC. The other three methods use no AC.

### Support matrix and evaluation conventions

Each method has native validation at step1, 64, 128, 192, 256, 320, 384, 448 and 512, **9 points** in total, with context2048 and 8192 global valid labels per point. The independent offline HF held-out evaluation is **a fixed small validation sample with context256 and 1024 labels**, `complete_split=false`. The two sets of CE differ in context and coverage. They are not native/HF loss parity. They do not give a scientific ranking of the methods.

| Method | core passes | AC | native held-out CE: step1 → 512 | HF CE: 256ctx/1024labels | Additional 8 vs 4+4 resume fork in this round |
|---|---:|---|---|---:|---|
| SMELT | 2 | none | 11.927265 → 7.219949 | 6.691771 | Supports the common resume entry point; no separate new fork |
| Meta | 4 | none | 11.952620 → 7.269093 | 6.750134 | Supports the common resume entry point; no separate new fork |
| Foil | 2 | Full | 11.934988 → 7.192571 | 6.657285 | Passed |
| Tying | 4 | none | 11.928846 → 7.300015 | 6.786260 | Passed |

All four methods were exported from their own real durable full DCP. HF was loaded in an independent process with **`python -I`, offline and without an import of the training repository package**. Fixed-depth held-out evaluation, cache and cached/uncached greedy all passed. Each method checked 3 incremental cache positions and 4 new greedy tokens. The maximum absolute error of the cache logits was ≤`3.338e-6`, with the check threshold `atol=1e-5/rtol=5e-5`. The greedy token IDs are the same. This qualifies the configurations that ran and the limited inputs. It does not promise any combination of scale, context and cache.

The actual new forks are those of Foil/Tying: 8 continuous steps compared with 4 steps + resume + 4 steps, with the same schedule horizon, source, data and topology. Foil compared 51 model tensors/153 optimizer tensors. Tying compared 65 model tensors/195 optimizer tensors. The maximum absolute difference of the model and of the optimizer was 0 in both. The optimizer scalars/hyperparameters, the RNG/progress of the two training ranks, the cursor/source state of the two data ranks and the discrete scheduler state agree exactly. Each fork ended at 262,144 positions/labels. The actual expert LR multiplier of Tying was 0.5. The common resume interface and the evidence of these two new forks are stated separately. The record does not say that each of the four methods had a fork.

### Budget pairing and data scope

The ordinary MoE/SMELT also completed **a real 8-step pilot per arm**, with 262,144 positions/labels per arm, context2048/global batch16 and the same DCLM fixture. From five fixed candidates, the selected pair is a 12-physical-layer×8-expert MoE against an 8-physical-layer×12-expert SMELT, both with 12 logical Cell calls; vocab151669, dim128, expert hidden256, top-k2, heads4/2. Below are the three analytical budgets of `fixed_terminal_matrix_v1`, without AC, with batch1/BF16 KV:

| Budget | Ordinary MoE | SMELT | Relative difference |
|---|---:|---:|---:|
| unique non-embedding parameters | 10,042,496 | 9,844,864 | -1.968% |
| matrix training FLOPs/position | 171,998,976 | 172,035,840 | +0.0214% |
| 2048ctx logical KV bytes | 6,291,456 | 6,291,456 | 0 |

All three are inside the predeclared 5%, and the real pilot ran. This does not mean wall-time/peak-memory matching. It does not extend to resource matching among the four 512-step recipes, or to which one has better quality. The [comparison definition](comparison.md) keeps unknown values and estimate ranges.

The data manifest fingerprint is `1dd1c249342678f5d758bd30aa7e5944e151b12f76844dcfb45505202e7d446b`. For the source, the Qwen3 tokenizer pin, the actual 300M/1M/1M split counts and the isolation, see the [DCLM evidence](evidence/dclm-v1.json). The original tokens/EOS are kept, and this library prepends BOS. The upstream download commit is unknown, and the exact pre-tokenized parent chunks are not published yet. The public report gives the content identity. The actual data/checkpoint artifacts are saved separately. These limits do not go away because the short runs completed.

The CPU model regression of this historical batch was **137 passed**. The Torch-free control suite was **286 passed, 19 skipped**. This does not mean that the current head reran with the same results. They are separate from the GPU qualification, and skips do not count as passed method checks. The four-method loss trajectories, the short generations and the 8-step pair pilot are engineering learning/delivery evidence. They are not reproductions of paper results, long-term stability or scaling conclusions.

## 0.1 engineering baseline history (2026-10-21)

This section keeps the original 0.1 record and the original failure facts. Its 2–4-step engineering fixtures, old CPU counts and multi-node results cannot be used as the four-method DCLM qualification of this round.

2026-10-21. The qualification on one node with four GPUs and on two nodes with one GPU each, and the native BF16 operator comparison, passed. CPU, GPU, multi-node and scientific training are recorded separately.

### Evidence obtained

| Path | Actual check | Result at that time |
|---|---|---|
| Control protocol | Complete suite in a frozen Torch-free environment | 286 passed, 15 skipped; optional model tests are skipped correctly |
| Model/data/training/HF/delivery | Complete CPU suite | 108 passed, including host RNG seed, export license, relative paths, FFN-chain update and an exact routing negative control |
| Community baseline | Official Qwen3-MoE, six configurations, all gradients, 4 AdamW steps, exact cache | Actually run; for values and tolerances, see [baseline](baselines.md) |
| Native distributed semantics | Real two-rank CPU FSDP, accumulation, Full/selected Selective AC, chunked head + aux | Passed. Gloo uses SUM communication, so this does not show CUDA/NCCL behavior |
| Data | Actual pinned-revision WikiText-2/GPT-2 prepare, mmap loader, exact resume | Passed. Explicit special roles agree with the manifest |
| HF delivery | Real DCP selective model load, offline new-process AutoClass, generation | Passed. FP32/BF16 dtype and cache had an independent review |
| Downstream connection | lm-eval 0.4.13 ARC-Easy with 2 questions, and HFLM likelihood/generation | The engineering chain passed. Random weights and a small limit do not show model quality |
| Native CUDA | H200 × 4, Torch 2.14.0+cu130, NCCL 2.30.7 | Six representative configurations, GA2, BF16/FSDP, Full AC, validation/full DCP, strict resume and actual model delivery all passed |
| CUDA baseline | Official Qwen3-MoE, six configurations, 2 AdamW steps each | Forward, all gradients, updates and cache passed; recorded separately from the four-step CPU check |
| Native BF16 operators | One H200, grouped experts/FlexAttention against a plain expert loop/SDPA | 80 physical parameter tensors in total. Forward/auxiliary scalars are identical. The expert set for each token and each logical call is exactly the same. Gradients passed the dual constraint that was declared after the first failure and before the rerun |
| Installed artifact | Wheel built from the sdist, clean install outside the checkout | Torch-free base import, model CLI, prepare → DCP → HF → heldout/generate → AutoClass numerical round trip passed. The LICENSE bytes/hash of the final installed-wheel export agree with the metadata |
| Multi-node | Two H200 hosts with one GPU each, native NCCL/FSDP | Actual TCP collective, GA2, validation/full DCP, strict 4 vs 2+2 resume and partial attention + Full AC passed |

CPU environment: Python 3.12.3, Torch 2.14.0+cpu, Transformers 5.20.2. Native training pins TorchTitan `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`. The actual GPU environment and the source snapshot are recorded separately. The versions of the CPU environment do not replace the versions of the GPU environment.

The GPU measurement snapshot is `6f5893daa5d5b337af1818a7862cb21312885d1d`, with Python 3.12.15, Transformers 5.19.0 and CUDA 13.0. Training parameters use BF16. Master parameters and gradient reduce use FP32. The six configurations are shared, partial attention, partial router, layer expert tying, input injection and dense. Of these, partial attention/injection use Full AC. Injection actually went through depths 3 and 2 and one visit of suffix gradients. Each configuration completed 2 updates. The shared continuous path completed 4 updates.

Strict resume compared 4 continuous steps with 2 steps + resume + 2 steps. Over 27 model tensors and 230272 elements, the maximum absolute difference is **0**. The Python/NumPy/Torch CPU/CUDA RNG, the data cursor/source counts, the progress and the discrete LR scheduler state of the four ranks agree completely. The cumulative valid input/label tokens are both 1024. The comparison reads only the model and the continuation state. The optimizer is kept in the full checkpoint, but its tensors were not all compared element by element.

After the real training DCP was exported as an FP32 HF artifact, the CE on 64 valid held-out targets is 3.910506. Cached generation and an independent HF load outside the project/TorchTitan paths passed. This fixture is generated from local text. It does not show pretraining quality. For the complete machine-readable results, see [four-GPU qualification](evidence/gpu-single-node.json), [CPU baseline](evidence/qwen3-cpu.json) and [CUDA baseline](evidence/qwen3-cuda.json).

The two-node qualification uses the pinned source `2fac1074fcd1e92cd3f08865e7b98851374611c2`. The training library is the same as in the four-GPU snapshot above, and the GPU software versions are the same. A native node identity check confirmed two different physical hosts, each with one visible H200. The NCCL BF16 all-reduce on the two ranks gave `[3, 5]` on both. The native log confirmed the use of NET/Socket TCP. Training used `dp_shard=2`, GA2, global batch 4 and sequence length 32. It completed shared continuous 4 steps, stop2/resume4, and partial attention + Full AC 2 steps. All of them had per-step validation and a durable full DCP.

For the two-node continuous and resume paths, the maximum difference over 27 model tensors and 230272 elements is **0**. The Python/NumPy/Torch CPU/CUDA RNG, data cursor, progress/source counts and discrete scheduler state of the two ranks agree exactly. The per-step metrics after resume are the same as the tail of the continuous path. The cumulative valid input/label tokens are both 512. The full optimizer checkpoint was actually loaded, but the optimizer was not compared tensor by tensor. The record adds no equality conclusion for it. The [two-node machine-readable evidence](evidence/gpu-multi-node.json) also keeps the real topology check, the source identity and the run results. This is a correctness qualification on a small engineering fixture. It is not proof of RDMA, throughput or scaling performance.

The native BF16 operator comparison uses the public script `scripts/compare_native.py`. The early execution commit is `2fac1074fcd1e92cd3f08865e7b98851374611c2`, and the script SHA256 is `bbaa9c02922fd90d0bfb706dd8d787b65d7a28a2d2d7d040c11e5f9adb602e0d`. The model/training library is the same as in the four-GPU qualification snapshot above. Two configurations were checked: shared, and one with prefix/suffix, linear input injection and suffix backpropagation. They have 27 and 53 physical parameter tensors. Both include the padding/document mask and the balance/z auxiliary objectives. For both configurations, the logits, loss, auxiliary scalars and summary counts of router assignments are identical. The global gradient relative L2 is `0.00597881` and `0.00528446`. The worst relative L2 over all tensors is `0.00995642`. The worst value of the maximum coordinate difference divided by the peak of its reference tensor is `0.013216`.

The execution commit of the additional exact-routing qualification is `75543a360c4f398e7c0c836c689a23e0c6141e3f`, and the script SHA256 is `de2ccdbf9b3e95aa0280494aafae8cf839fc886dace14fee8d86b4ce5f0c6555`. For each logical Router call, the script checks the owner, the valid input mask and the top-k expert set of each valid token. It allows a different return order of top-k, and it includes no-grad visits. The 4 calls and 400 expert assignments of shared, and the 8 calls and 800 assignments of the injected suffix, are all exactly the same. The forward check and the gradient check of 80 physical parameters also passed, with unchanged precision thresholds. The [schema-2 complete retest](evidence/native-kernel-routing-qualified.json) saves the hash of the IDs of each call and the actual comparison results. The old schema-1 field `router_assignments_equal` compares only summary counts. It cannot prove that each token/call is the same. The old record is kept. A new token/visit swap negative control enforces this boundary.

Gradient acceptance requires both conditions: a relative L2 ≤ `0.03` for each tensor and for the global vector, and for each tensor `max_abs(g_native - g_ref) ≤ 2 * eps_bf16 * max_abs(g_ref) + 1e-6`, where `eps_bf16 = 0.0078125`. This qualification on limited inputs refers to the BF16 precision and the tensor scale. It is not an error guarantee for any reduction, model or hardware.

In the first element-wise check with `atol=0.0005, rtol=0.05`, 1 of the 4096 gradient coordinates of the shared embedding failed. The original task and report still keep **Failed**. After the first failure, a review of the numerical policy changed the check to the dual constraint above, and declared it before the rerun. Then the public script ran again with the same model, the same inputs and the unchanged training library, and it passed. For the two complete per-tensor results, see the [first failed record](evidence/native-kernel-initial-failed.json) and the [requalification record](evidence/native-kernel-qualified.json). Both original records are kept.

As of **2026-10-21 10:11:36 UTC**, the cumulative conservative resource count of the completed GPU qualification batches is **0.51972 GPU-hours**. This includes the two earlier failed attempts and **0.09056 GPU-hours** for this two-node qualification. This run is counted as 163 seconds on two GPUs. This is a snapshot of the historical qualification batches.

The [installed-artifact check](evidence/package-installation.json) uses a separate, independent environment. It builds an sdist from a git archive and then builds a wheel from the sdist. The model, data, training and HF runtime can all be installed. The source package does not contain the research documents and the qualification scripts. Use them from a Git checkout. The Git capture of the control protocol also needs the original Git checkout, by design.

The [final package check](evidence/package-final.json) uses a fresh archive/sdist/wheel of `2e000a49ff788e1ecdb7d816c4cfbfdf23b966c8`. It exports with `python -I` outside the checkout and actually reads the license of the installed distribution. The exported LICENSE is byte-identical to the wheel metadata, and the provenance hash covers it. 14 public modules are byte-identical to the source archive. A separate HF-only environment without the project/TorchTitan installed completed a local load and an exact comparison of cache and greedy recomputation. The cache error is zero.

The source `a19c6fa611f564078a45533d8c92ce096b5564e1` after the path fix completed the [package check](evidence/package-paths.json) again. It covers the new archive/sdist/wheel, the byte identity of the 14 public modules, the installed LICENSE, a real DCP export, heldout/generate and an independent HF-only consumer. The original package record is kept.

With the same source, a [relative-path end-to-end regression](evidence/relative-path-export.json) also ran on H200. It started real native BF16 training from a relative manifest, tokenizer and output directory. It completed GA2, two updates, per-step validation and a durable full DCP. The training metadata records absolute paths. After a change of working directory, export with automatic discovery and export with an explicitly given tokenizer both passed. The 27 model tensors are identical. The 64 held-out targets and the cached generation CLI passed. The first check script used a provenance field that does not exist, and it failed. After the field was corrected, the check passed. The original driver failure and the product command success are kept separately. This change touched only the path normalization of the training entry point. It did not change the model or the optimization computation. The earlier four-GPU/cross-node numerical evidence still matches the source snapshots that each record names.

### Problems found and fixed

- Creating a StatefulDataLoader iterator used values of the global CPU RNG of training. Fix: the loader has its own generator. A real next-batch/RNG regression passed.
- Selective AC is not compatible with the nested FSDP recomputation of independent attention per visit. Fix: this combination is rejected explicitly. Full AC and AC off stay supported.
- The training continuation identity did not include the complete loss config. It now includes the loss, AC/compile, seed policy, source hash, data/tokenizer and parallelism/optimizer/schedule.
- Explicit BOS/EOS/PAD roles could be lost in export. Fix: the preparation stage normalizes the copied assets, and the manifest, training and export check the same identity.
- A normal gradient call in HF eval built an inference cache by mistake. Fix: the automatic cache is used only for inference without gradients. An explicit config choice is kept.
- The initialization of the native framework set the Torch seed, but did not initialize Python/NumPy. Fix: the rank-specific host RNG is initialized explicitly, and the Torch generators do not change. A real four-GPU continuation retest passed.
- The standalone HF export did not include the license of the inference source code. Fix: it now reads LICENSE from the source archive or from the installed wheel metadata, writes it with the artifact and calculates its hash. The actual installed-wheel path passed.
- Training and export resolved a relative manifest path against different directories. Fix: the training entry point records the absolute manifest, tokenizer and output paths under the start directory. A manifest given as a directory is normalized to the `manifest.json` in it. The CPU regression and a real two-step training followed by an HF export from a different directory passed.

### 0.1 support set

The target scope is: plain reference core, single-state causal LM, softmax normalized top-k/dropless, GQA/RoPE, terminal CE, AdamW, same-topology continuation and independent HF exact cache. Fixed/full supports ordinary/stack/layer/typical partial sharing. Sampled depth and suffix are limited to a fully shared stack. Strict data continuation currently requires `num_workers=0` and immutable shared manifests.

CUDA graphs, arbitrary model compile, EP/TP/CP/PP, elastic-topology resume, dynamic token halting, approximate KV, multi-state models and other optimizers were not part of the verified commitment of 0.1. Later stages extended this scope for two items only: multi-state models (see [Qualification of new training dynamics and combinations](#qualification-of-new-training-dynamics-and-combinations)) and one other optimizer, Muon on GPU (see #49). The other items of this list remain outside the verified scope. That a combination can be built on CPU does not mean that it has native machine qualification.

The final GPU qualification runs use a pinned source snapshot, an explicit program time limit and few steps. They record the real successful updates, the durable checkpoint, the state of each rank and the program exit.

### Inferences that the historical evidence does not support

This evidence cannot prove any model scale, any hardware, any combination of all switches, long-term stability, a throughput lead or paper scores. The loss decrease on engineering fixtures only finds breakpoints in training updates. It does not show a data recipe, a pretraining effect or a scaling conclusion.
