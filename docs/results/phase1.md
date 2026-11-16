# Phase 1 results

<!-- Written by scripts/results_table.py from phase1.json; edit the data, not this file. -->

Protocol: every row is trained on the same data (5.0B tokens of the P2 track's DCLM manifest) with the same schedule (4,769 updates of 1,048,576 tokens, warmup-stable-decay) and seed 42; each optimizer uses one learning rate (AdamW 1.2e-3, Muon 1.7e-3), chosen on the ordinary MoE and the looped MoE ([#180](https://github.com/datamllab/LoopTitan/issues/180)).

- Test, Validation: bits per byte (BPB) on the complete held-out test and validation splits after the last update.
- Δ test: test BPB minus the ordinary MoE's with the same optimizer, in mBPB (0.001 BPB).
- Parameters, FLOPs, KV: unique non-embedding parameters, training FLOPs per token and logical KV cache relative to `recipes/phase1/m0-<optimizer>.json`, from `moeloop.comparison.compare_recipes` with the P2 track's accounting in `eval/bench/tracks.json`. Matched budgets are within 5% on all three.
- Source: the entry of the method map (see [Sources](#sources)) that the design follows.
- Run: the issue whose results comment holds the values. The design links to its frozen configuration.

## AdamW

### Matched budgets: from the ordinary MoE to the looped MoE, one change per row

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [ordinary MoE: 12 layers, 64 experts each, top-8](../../recipes/phase1/m0-adamw.json) |  | 0.999533 | +0.00 | 1.001293 | +0.0% | +0.0% | +0.0% | 1.2e-3 | [#180](https://github.com/datamllab/LoopTitan/issues/180) |
| [finer experts: 72 per layer at width 384, top-9](../../recipes/shape/g-adamw.json) |  | 0.995309 | -4.22 | 0.997202 | +0.0% | +0.0% | +0.0% | 1.2e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [deeper and narrower: 18 layers at width 1056, 58 experts each](../../recipes/shape/u-adamw.json) |  | 0.992619 | -6.91 | 0.994388 | -0.3% | +2.3% | +3.1% | 1.2e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [18 layers: 88 experts in the outer 6, 44 in the middle 12](../../recipes/shape/r-adamw.json) |  | 0.993624 | -5.91 | 0.995442 | +0.8% | +2.3% | +3.1% | 1.2e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [18 layers, 88/44/88 experts, middle residual ×0.5 (untied twin)](../../recipes/phase1/m4-adamw.json) |  | 0.990645 | -8.89 | 0.992400 | +0.8% | +2.3% | +3.1% | 1.2e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [loop the middle 6 of 12 layers twice, residual ×0.5, share experts only](../../recipes/phase1/m7-adamw.json) |  | 0.994194 | -5.34 | 0.996015 | +0.9% | +2.4% | +3.1% | 1.2e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [loop the middle 6 of 12 layers twice, residual ×0.5, share experts and router](../../recipes/phase1/m5-adamw.json) |  | 0.990329 | -9.20 | 0.992225 | +0.8% | +2.4% | +3.1% | 1.2e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [loop the middle 6 of 12 layers twice, residual ×0.5](../../recipes/phase1/m3-adamw.json) | [1](#sources) | 0.990544 | -8.99 | 0.992211 | -0.3% | +2.4% | +3.1% | 1.2e-3 | [#180](https://github.com/datamllab/LoopTitan/issues/180) |

### Matched budgets: configurations in `recipes/methods/`

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [loop the middle 4 of 10 layers 3 times, residual ×1/3](../../recipes/methods/smelt-v3-adamw.json) | [1](#sources) | 0.992698 | -6.83 | 0.994245 | +0.1% | +2.5% | +3.1% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop the middle 3 of 9 layers 4 times, residual ×1/4](../../recipes/methods/smelt-v4-adamw.json) | [1](#sources) | 0.992454 | -7.08 | 0.994377 | +0.0% | +2.7% | +3.1% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop the middle 6 of 12 layers twice, residual ×0.25 then ×1](../../recipes/methods/latentloop-residual-adamw.json) | [2](#sources) | 0.992019 | -7.51 | 0.993853 | -0.3% | +2.4% | +3.1% | 1.2e-3 | [#254](https://github.com/datamllab/LoopTitan/issues/254) |
| [loop the middle 2 of 6 layers 4 times](../../recipes/methods/meta-adamw.json) | [3](#sources) | 1.002254 | +2.72 | 1.003940 | -0.1% | +0.2% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop the middle 4 of 8 layers twice, attention per visit](../../recipes/methods/foil-adamw.json) | [4](#sources) | 0.997246 | -2.29 | 0.999034 | +0.0% | +0.1% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [12 layers, every 4 consecutive layers share one expert bank](../../recipes/methods/expert-tying-adamw.json) | [5](#sources) | 1.003949 | +4.42 | 1.005800 | +0.2% | +0.6% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop the middle 2 of 4 layers a random number of times (mean 4), input re-injected](../../recipes/methods/huginn-adamw.json) | [6](#sources) | 1.435139 | +435.61 | 1.438455 | +0.0% | -0.1% | +4.2% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop the middle 1 of 5 layers 8 times, decaying input injection, top-2](../../recipes/methods/sandwich-adamw.json) | [7](#sources) | 1.015433 | +15.90 | 1.017309 | -0.3% | +2.4% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop all 4 layers 3 times, router per visit, EMA memories](../../recipes/methods/loom-adamw.json) | [8](#sources) | 1.007497 | +7.96 | 1.009361 | +0.0% | +0.4% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop all 4 layers 3 times at width 640, learned exit](../../recipes/methods/ouro-adamw.json) | [9](#sources) | 1.033283 | +33.75 | 1.035037 | -0.1% | +0.0% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop all 6 layers twice, per-visit LoRA on attention](../../recipes/methods/relaxed-adamw.json) | [10](#sources) | 0.997343 | -2.19 | 0.999244 | +0.3% | +3.1% | +0.0% | 1.2e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |

### Outside the matched budgets

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [ordinary MoE, middle 6 of 12 layers looped twice, residual ×0.5](../../recipes/methods/m0-loop-adamw.json) | [1](#sources) | 0.984730 | -14.80 | 0.986424 | +0.0% | +31.5% | +50.0% | 1.2e-3 | [#247](https://github.com/datamllab/LoopTitan/issues/247) |
| [ordinary MoE with twice the experts: 128 per layer](../../recipes/phase1/m0-2e-adamw.json) |  | 0.985241 | -14.29 | 0.986998 | +96.7% | +0.2% | +0.0% | 1.2e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |

## Muon

### Matched budgets: from the ordinary MoE to the looped MoE, one change per row

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [ordinary MoE: 12 layers, 64 experts each, top-8](../../recipes/phase1/m0-muon.json) |  | 0.974376 | +0.00 | 0.976141 | +0.0% | +0.0% | +0.0% | 1.7e-3 | [#180](https://github.com/datamllab/LoopTitan/issues/180) |
| [finer experts: 72 per layer at width 384, top-9](../../recipes/shape/g-muon.json) |  | 0.973517 | -0.86 | 0.975193 | +0.0% | +0.0% | +0.0% | 1.7e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [deeper and narrower: 18 layers at width 1056, 58 experts each](../../recipes/shape/u-muon.json) |  | 0.970165 | -4.21 | 0.971959 | -0.3% | +2.3% | +3.1% | 1.7e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [18 layers: 88 experts in the outer 6, 44 in the middle 12](../../recipes/shape/r-muon.json) |  | 0.971013 | -3.36 | 0.972742 | +0.8% | +2.3% | +3.1% | 1.7e-3 | [#218](https://github.com/datamllab/LoopTitan/issues/218) |
| [18 layers, 88/44/88 experts, middle residual ×0.5 (untied twin)](../../recipes/phase1/m4-muon.json) |  | 0.971043 | -3.33 | 0.972988 | +0.8% | +2.3% | +3.1% | 1.7e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [loop the middle 6 of 12 layers twice, residual ×0.5, share experts only](../../recipes/phase1/m7-muon.json) |  | 0.971189 | -3.19 | 0.972990 | +0.9% | +2.4% | +3.1% | 1.7e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [loop the middle 6 of 12 layers twice, residual ×0.5, share experts and router](../../recipes/phase1/m5-muon.json) |  | 0.969498 | -4.88 | 0.971184 | +0.8% | +2.4% | +3.1% | 1.7e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |
| [loop the middle 6 of 12 layers twice, residual ×0.5](../../recipes/phase1/m3-muon.json) | [1](#sources) | 0.970676 | -3.70 | 0.972425 | -0.3% | +2.4% | +3.1% | 1.7e-3 | [#180](https://github.com/datamllab/LoopTitan/issues/180) |

### Matched budgets: configurations in `recipes/methods/`

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [loop the middle 4 of 10 layers 3 times, residual ×1/3](../../recipes/methods/smelt-v3-muon.json) | [1](#sources) | 0.970858 | -3.52 | 0.972628 | +0.1% | +2.5% | +3.1% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop the middle 3 of 9 layers 4 times, residual ×1/4](../../recipes/methods/smelt-v4-muon.json) | [1](#sources) | 0.972644 | -1.73 | 0.974200 | +0.0% | +2.7% | +3.1% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop the middle 6 of 12 layers twice, residual ×0.25 then ×1](../../recipes/methods/latentloop-residual-muon.json) | [2](#sources) | 0.970283 | -4.09 | 0.972178 | -0.3% | +2.4% | +3.1% | 1.7e-3 | [#254](https://github.com/datamllab/LoopTitan/issues/254) |
| [loop the middle 2 of 6 layers 4 times](../../recipes/methods/meta-muon.json) | [3](#sources) | 0.978252 | +3.88 | 0.980126 | -0.1% | +0.2% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop the middle 4 of 8 layers twice, attention per visit](../../recipes/methods/foil-muon.json) | [4](#sources) | 0.973550 | -0.83 | 0.975301 | +0.0% | +0.1% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [12 layers, every 4 consecutive layers share one expert bank](../../recipes/methods/expert-tying-muon.json) | [5](#sources) | 0.982451 | +8.07 | 0.984287 | +0.2% | +0.6% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop the middle 2 of 4 layers a random number of times (mean 4), input re-injected](../../recipes/methods/huginn-muon.json) | [6](#sources) | 0.985060 | +10.68 | 0.986725 | +0.0% | -0.1% | +4.2% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop the middle 1 of 5 layers 8 times, decaying input injection, top-2](../../recipes/methods/sandwich-muon.json) | [7](#sources) | 0.990709 | +16.33 | 0.992557 | -0.3% | +2.4% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop all 4 layers 3 times, router per visit, EMA memories](../../recipes/methods/loom-muon.json) | [8](#sources) | 0.991848 | +17.47 | 0.993528 | +0.0% | +0.4% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop all 4 layers 3 times at width 640, learned exit](../../recipes/methods/ouro-muon.json) | [9](#sources) | 1.013632 | +39.26 | 1.015111 | -0.1% | +0.0% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |
| [loop all 6 layers twice, per-visit LoRA on attention](../../recipes/methods/relaxed-muon.json) | [10](#sources) | 0.978307 | +3.93 | 0.980033 | +0.3% | +3.1% | +0.0% | 1.7e-3 | [#232](https://github.com/datamllab/LoopTitan/issues/232) |

### Outside the matched budgets

| Design | Source | Test | Δ test (mBPB) | Validation | Parameters | FLOPs | KV | LR | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [ordinary MoE, middle 6 of 12 layers looped twice, residual ×0.5](../../recipes/methods/m0-loop-muon.json) | [1](#sources) | 0.962134 | -12.24 | 0.964069 | +0.0% | +31.5% | +50.0% | 1.7e-3 | [#247](https://github.com/datamllab/LoopTitan/issues/247) |
| [ordinary MoE with twice the experts: 128 per layer](../../recipes/phase1/m0-2e-muon.json) |  | 0.964525 | -9.85 | 0.966379 | +96.7% | +0.2% | +0.0% | 1.7e-3 | [#200](https://github.com/datamllab/LoopTitan/issues/200) |

## Sources

1. [SMELT v3](../method-map.md)
2. [LatentLoop](../method-map.md)
3. [Meta Looped MoE v1](../method-map.md)
4. [Foil v1](../method-map.md)
5. [Tying v1](../method-map.md)
6. [Huginn v2](../method-map.md)
7. [Looped Models Done Right, Part I](../method-map.md)
8. [LOOM v1](../method-map.md)
9. [Ouro v5](../method-map.md)
10. [Relaxed Recursive Transformers v3](../method-map.md)
