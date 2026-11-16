# Phase 1 results

<!-- Written by scripts/results_table.py from phase1.json; edit the data, not this file. -->

Protocol: every row is trained on the same data (5.0B tokens of the P2 track's DCLM manifest) with the same schedule (4,769 updates of 1,048,576 tokens, warmup-stable-decay) and seed 42. Learning rate: tuned once per optimizer on the ordinary MoE (AdamW 1.2e-3, Muon 1.7e-3; [#180](https://github.com/datamllab/LoopTitan/issues/180)); the same grid on the chain-end loop selects the same value. Every other row inherits that value without a search of its own; learning-rate variants and checks in the record are not in the table (listed below, and one rerun at another learning rate is a note under AdamW).

This page shows a subset of the Phase 1 record. Not shown, trained with the same data, schedule and learning rate: `phase1/m1`, `phase1/m1s`, `phase1/m2` and `phase1/m6` ([#200](https://github.com/datamllab/LoopTitan/issues/200)), `shape/m0r` ([#218](https://github.com/datamllab/LoopTitan/issues/218)), `shape/d24` and `shape/d36` ([#242](https://github.com/datamllab/LoopTitan/issues/242)) and `shape/m0i` ([#252](https://github.com/datamllab/LoopTitan/issues/252)). Not shown, learning-rate variants of shown rows: `phase1/m4-elr` and `phase1/m7-elr`, with the middle layers' expert learning rate ×√2 ([#266](https://github.com/datamllab/LoopTitan/issues/266)), and `shape/m0lr`, with the middle layers' output-projection learning rate ×0.5 ([#252](https://github.com/datamllab/LoopTitan/issues/252)). The learning-rate grids and checks are in [#180](https://github.com/datamllab/LoopTitan/issues/180) and [#243](https://github.com/datamllab/LoopTitan/issues/243); the studies are [#177](https://github.com/datamllab/LoopTitan/issues/177), [#216](https://github.com/datamllab/LoopTitan/issues/216) and [#225](https://github.com/datamllab/LoopTitan/issues/225).

- Test, Validation: bits per byte (BPB) on the complete held-out test and validation splits after the last update.
- Δ test: test BPB minus the ordinary MoE's with the same optimizer, in mBPB (0.001 BPB).
- Parameters, FLOPs, KV: unique non-embedding parameters, training FLOPs per token and logical KV cache relative to `recipes/phase1/m0-<optimizer>.json`, from `moeloop.comparison.compare_recipes` with the P2 track's accounting in `eval/bench/tracks.json`. Matched budgets are within 5% on all three.
- Source: the published designs the configuration and its ancestors in `scripts/lineage.py` follow, numbered in [Sources](#sources).
- Run: the issue whose results comment holds the values. The design links to its frozen configuration.

## AdamW

### Matched budgets: one-change chain from ordinary MoE (`phase1/m0`) to chain-end loop (`phase1/m3`)

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [ordinary MoE: 12 layers at width 1280, 64 experts per layer, expert hidden size 432, top-8](../../recipes/phase1/m0-adamw.json) |  | 0.999533 | +0.00 | 1.001293 | +0.0% | +0.0% | +0.0% | 1.2e-3 | [#180](https://github.com/datamllab/LoopTitan/issues/180) |
| [finer experts: 12 layers at width 1280, 72 experts per layer, expert hidden size 384, top-9](../../recipes/shape/g-adamw.json) |  | 0.995309 | -4.22 | 0.997202 | +0.0% | +0.0% | +0.0% | 1.2e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [deeper and narrower: 18 layers at width 1056, 58 experts per layer, expert hidden size 384, top-8](../../recipes/shape/u-adamw.json) | [1](#sources) | 0.992619 | -6.91 | 0.994388 | -0.3% | +2.3% | +3.1% | 1.2e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [expert layout: 18 layers at width 1056, 88 experts in each of the first 3 and last 3 layers, 44 in each of the middle 12, expert hidden size 384](../../recipes/shape/r-adamw.json) | [1](#sources) | 0.993624 | -5.91 | 0.995442 | +0.8% | +2.3% | +3.1% | 1.2e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [untied twin of the chain-end loop: 18 layers at width 1056, 88 experts in each of the first 3 and last 3 layers, 44 in each of the middle 12, expert hidden size 384, middle residual ×0.5](../../recipes/phase1/m4-adamw.json) | [1](#sources) | 0.990645 | -8.89 | 0.992400 | +0.8% | +2.3% | +3.1% | 1.2e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [shared experts: 12 layers at width 1056, 88 experts per layer, expert hidden size 384; the middle 6 looped twice, residual ×0.5; attention, router and norms per visit](../../recipes/phase1/m7-adamw.json) | [1](#sources) | 0.994194 | -5.34 | 0.996015 | +0.9% | +2.4% | +3.1% | 1.2e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [shared experts and router: 12 layers at width 1056, 88 experts per layer, expert hidden size 384; the middle 6 looped twice, residual ×0.5; attention and norms per visit](../../recipes/phase1/m5-adamw.json) | [1](#sources) | 0.990329 | -9.20 | 0.992225 | +0.8% | +2.4% | +3.1% | 1.2e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [chain-end loop: 12 layers at width 1056, 88 experts per layer, expert hidden size 384; the middle 6 looped twice, residual ×0.5; every part shared across visits](../../recipes/phase1/m3-adamw.json) | [1](#sources) | 0.990544 | -8.99 | 0.992211 | -0.3% | +2.4% | +3.1% | 1.2e-3 | [#180](https://github.com/datamllab/LoopTitan/issues/180) |

### Matched budgets: configurations in `recipes/methods/`

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [10 layers at width 1056, 88 experts per outer layer and 134 per middle layer, expert hidden size 384; the middle 4 looped 3 times, residual ×1/3](../../recipes/methods/smelt-v3-adamw.json) | [1](#sources) | 0.992698 | -6.83 | 0.994245 | +0.1% | +2.5% | +3.1% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [9 layers at width 1056, 88 experts per outer layer and 179 per middle layer, expert hidden size 384; the middle 3 looped 4 times, residual ×1/4](../../recipes/methods/smelt-v4-adamw.json) | [1](#sources) | 0.992454 | -7.08 | 0.994377 | +0.0% | +2.7% | +3.1% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [12 layers at width 1056, 88 experts per layer, expert hidden size 384; the middle 6 looped twice, residual ×0.25 on the first visit and ×1 on the second](../../recipes/methods/latentloop-residual-adamw.json) | [1](#sources), [2](#sources) | 0.992019 | -7.51 | 0.993853 | -0.3% | +2.4% | +3.1% | 1.2e-3 | [#254](https://github.com/datamllab/LoopTitan/issues/254) |
| [6 layers at width 1280, 130 experts per layer, expert hidden size 432; the middle 2 looped 4 times](../../recipes/methods/meta-adamw.json) | [3](#sources) | 1.002254 | +2.72 | 1.003940 | -0.1% | +0.2% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [8 layers at width 1280, 64 experts per outer layer and 128 per middle layer, expert hidden size 432; the middle 4 looped twice, attention per visit](../../recipes/methods/foil-adamw.json) | [4](#sources) | 0.997246 | -2.29 | 0.999034 | +0.0% | +0.1% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [12 layers at width 1280, expert hidden size 432; each 4 consecutive layers share one bank of 256 experts; attention, router and norms per layer](../../recipes/methods/expert-tying-adamw.json) | [5](#sources) | 1.003949 | +4.42 | 1.005800 | +0.2% | +0.6% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [4 layers at width 1280, 86 experts per layer, expert hidden size 984; the middle 2 looped with linear input re-injection; training samples the loop count per micro-batch (Poisson-lognormal, mean 4) from a random start, with gradients through the last 2 loops; evaluated at 4 loops from a zero start](../../recipes/methods/huginn-adamw.json) | [6](#sources) | 1.435139 (note 1) | +435.61 | 1.438455 | +0.0% | -0.1% | +4.2% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [5 layers at width 1280, 39 experts per layer, expert hidden size 1728, top-2; the middle 1 looped 8 times with diagonal-decay input injection](../../recipes/methods/sandwich-adamw.json) | [6](#sources), [7](#sources) | 1.015433 | +15.90 | 1.017309 | -0.3% | +2.4% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [4 layers at width 1280, 196 experts per layer, expert hidden size 432; all 4 looped 3 times with a router per visit and EMA state updates](../../recipes/methods/loom-adamw.json) | [8](#sources) | 1.007497 | +7.96 | 1.009361 | +0.0% | +0.4% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [4 layers at width 640, 562 experts per layer, expert hidden size 304; all 4 looped 3 times; trained on the expected loss over an exit distribution (entropy weight 0.1), evaluated after 3 loops](../../recipes/methods/ouro-adamw.json) | [9](#sources) | 1.033283 | +33.75 | 1.035037 | -0.1% | +0.0% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [6 layers at width 1280, 129 experts per layer, expert hidden size 432; all 6 looped twice with per-visit LoRA (rank 160) on attention](../../recipes/methods/relaxed-adamw.json) | [10](#sources) | 0.997343 | -2.19 | 0.999244 | +0.3% | +3.1% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |

### Outside matched budgets

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [ordinary MoE looped: 12 layers at width 1280, 64 experts per layer, expert hidden size 432; the middle 6 looped twice, residual ×0.5](../../recipes/methods/m0-loop-adamw.json) | [1](#sources) | 0.984730 | -14.80 | 0.986424 | +0.0% | +31.5% | +50.0% | 1.2e-3 | [#247](https://github.com/datamllab/LoopTitan/issues/247) |
| [ordinary MoE with twice the experts: 12 layers at width 1280, 128 experts per layer, expert hidden size 432](../../recipes/phase1/m0-2e-adamw.json) |  | 0.985241 | -14.29 | 0.986998 | +96.7% | +0.2% | +0.0% | 1.2e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |

Notes (AdamW):

1. At the protocol learning rate the loss of this configuration fell slowly under AdamW; a short check over four learning rates (1.5e-4, 3e-4, 6e-4 and 1.2e-3, 597 updates each) chose 3e-4, and the full run at 3e-4 gave test BPB 1.049098 and validation BPB 1.051026 ([#243](https://github.com/datamllab/LoopTitan/issues/243)).

## Muon

### Matched budgets: one-change chain from ordinary MoE (`phase1/m0`) to chain-end loop (`phase1/m3`)

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [ordinary MoE: 12 layers at width 1280, 64 experts per layer, expert hidden size 432, top-8](../../recipes/phase1/m0-muon.json) |  | 0.974376 | +0.00 | 0.976141 | +0.0% | +0.0% | +0.0% | 1.7e-3 | [#180](https://github.com/datamllab/LoopTitan/issues/180) |
| [finer experts: 12 layers at width 1280, 72 experts per layer, expert hidden size 384, top-9](../../recipes/shape/g-muon.json) |  | 0.973517 | -0.86 | 0.975193 | +0.0% | +0.0% | +0.0% | 1.7e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [deeper and narrower: 18 layers at width 1056, 58 experts per layer, expert hidden size 384, top-8](../../recipes/shape/u-muon.json) | [1](#sources) | 0.970165 | -4.21 | 0.971959 | -0.3% | +2.3% | +3.1% | 1.7e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [expert layout: 18 layers at width 1056, 88 experts in each of the first 3 and last 3 layers, 44 in each of the middle 12, expert hidden size 384](../../recipes/shape/r-muon.json) | [1](#sources) | 0.971013 | -3.36 | 0.972742 | +0.8% | +2.3% | +3.1% | 1.7e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [untied twin of the chain-end loop: 18 layers at width 1056, 88 experts in each of the first 3 and last 3 layers, 44 in each of the middle 12, expert hidden size 384, middle residual ×0.5](../../recipes/phase1/m4-muon.json) | [1](#sources) | 0.971043 | -3.33 | 0.972988 | +0.8% | +2.3% | +3.1% | 1.7e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [shared experts: 12 layers at width 1056, 88 experts per layer, expert hidden size 384; the middle 6 looped twice, residual ×0.5; attention, router and norms per visit](../../recipes/phase1/m7-muon.json) | [1](#sources) | 0.971189 | -3.19 | 0.972990 | +0.9% | +2.4% | +3.1% | 1.7e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [shared experts and router: 12 layers at width 1056, 88 experts per layer, expert hidden size 384; the middle 6 looped twice, residual ×0.5; attention and norms per visit](../../recipes/phase1/m5-muon.json) | [1](#sources) | 0.969498 | -4.88 | 0.971184 | +0.8% | +2.4% | +3.1% | 1.7e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [chain-end loop: 12 layers at width 1056, 88 experts per layer, expert hidden size 384; the middle 6 looped twice, residual ×0.5; every part shared across visits](../../recipes/phase1/m3-muon.json) | [1](#sources) | 0.970676 | -3.70 | 0.972425 | -0.3% | +2.4% | +3.1% | 1.7e-3 | [#180](https://github.com/datamllab/LoopTitan/issues/180) |

### Matched budgets: configurations in `recipes/methods/`

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [10 layers at width 1056, 88 experts per outer layer and 134 per middle layer, expert hidden size 384; the middle 4 looped 3 times, residual ×1/3](../../recipes/methods/smelt-v3-muon.json) | [1](#sources) | 0.970858 | -3.52 | 0.972628 | +0.1% | +2.5% | +3.1% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [9 layers at width 1056, 88 experts per outer layer and 179 per middle layer, expert hidden size 384; the middle 3 looped 4 times, residual ×1/4](../../recipes/methods/smelt-v4-muon.json) | [1](#sources) | 0.972644 | -1.73 | 0.974200 | +0.0% | +2.7% | +3.1% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [12 layers at width 1056, 88 experts per layer, expert hidden size 384; the middle 6 looped twice, residual ×0.25 on the first visit and ×1 on the second](../../recipes/methods/latentloop-residual-muon.json) | [1](#sources), [2](#sources) | 0.970283 | -4.09 | 0.972178 | -0.3% | +2.4% | +3.1% | 1.7e-3 | [#254](https://github.com/datamllab/LoopTitan/issues/254) |
| [6 layers at width 1280, 130 experts per layer, expert hidden size 432; the middle 2 looped 4 times](../../recipes/methods/meta-muon.json) | [3](#sources) | 0.978252 | +3.88 | 0.980126 | -0.1% | +0.2% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [8 layers at width 1280, 64 experts per outer layer and 128 per middle layer, expert hidden size 432; the middle 4 looped twice, attention per visit](../../recipes/methods/foil-muon.json) | [4](#sources) | 0.973550 | -0.83 | 0.975301 | +0.0% | +0.1% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [12 layers at width 1280, expert hidden size 432; each 4 consecutive layers share one bank of 256 experts; attention, router and norms per layer](../../recipes/methods/expert-tying-muon.json) | [5](#sources) | 0.982451 | +8.07 | 0.984287 | +0.2% | +0.6% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [4 layers at width 1280, 86 experts per layer, expert hidden size 984; the middle 2 looped with linear input re-injection; training samples the loop count per micro-batch (Poisson-lognormal, mean 4) from a random start, with gradients through the last 2 loops; evaluated at 4 loops from a zero start](../../recipes/methods/huginn-muon.json) | [6](#sources) | 0.985060 | +10.68 | 0.986725 | +0.0% | -0.1% | +4.2% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [5 layers at width 1280, 39 experts per layer, expert hidden size 1728, top-2; the middle 1 looped 8 times with diagonal-decay input injection](../../recipes/methods/sandwich-muon.json) | [6](#sources), [7](#sources) | 0.990709 | +16.33 | 0.992557 | -0.3% | +2.4% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [4 layers at width 1280, 196 experts per layer, expert hidden size 432; all 4 looped 3 times with a router per visit and EMA state updates](../../recipes/methods/loom-muon.json) | [8](#sources) | 0.991848 | +17.47 | 0.993528 | +0.0% | +0.4% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [4 layers at width 640, 562 experts per layer, expert hidden size 304; all 4 looped 3 times; trained on the expected loss over an exit distribution (entropy weight 0.1), evaluated after 3 loops](../../recipes/methods/ouro-muon.json) | [9](#sources) | 1.013632 | +39.26 | 1.015111 | -0.1% | +0.0% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [6 layers at width 1280, 129 experts per layer, expert hidden size 432; all 6 looped twice with per-visit LoRA (rank 160) on attention](../../recipes/methods/relaxed-muon.json) | [10](#sources) | 0.978307 | +3.93 | 0.980033 | +0.3% | +3.1% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |

### Outside matched budgets

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [ordinary MoE looped: 12 layers at width 1280, 64 experts per layer, expert hidden size 432; the middle 6 looped twice, residual ×0.5](../../recipes/methods/m0-loop-muon.json) | [1](#sources) | 0.962134 | -12.24 | 0.964069 | +0.0% | +31.5% | +50.0% | 1.7e-3 | [#247](https://github.com/datamllab/LoopTitan/issues/247) |
| [ordinary MoE with twice the experts: 12 layers at width 1280, 128 experts per layer, expert hidden size 432](../../recipes/phase1/m0-2e-muon.json) |  | 0.964525 | -9.85 | 0.966379 | +96.7% | +0.2% | +0.0% | 1.7e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |

## Sources

1. [SMELT v3](https://arxiv.org/html/2609.01343v3)
2. [LatentLoop v1](https://arxiv.org/html/2610.19975v1)
3. [Meta Looped MoE v1](https://arxiv.org/html/2609.40316v1)
4. [Foil v1](https://arxiv.org/html/2609.35751v1)
5. [Tying v1](https://arxiv.org/html/2606.16825v1)
6. [Huginn-inspired MoE v2](https://arxiv.org/html/2502.05171v2)
7. [Looped Models Done Right, Part I v1](https://arxiv.org/html/2610.23036v1)
8. [LOOM v1](https://arxiv.org/html/2610.01153v1)
9. [Ouro v5](https://arxiv.org/html/2510.25741v5)
10. [Relaxed Recursive Transformers v3](https://arxiv.org/html/2410.20672v3)

What each configuration keeps from its source and what it adapts: [method map](../method-map.md).
