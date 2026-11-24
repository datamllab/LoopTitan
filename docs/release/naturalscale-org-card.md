---
title: README
emoji: 🐚
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
---

<!-- One line about Natural Scale, the lab, goes here above its releases; LoopTitan is the first. -->

<!-- banner -->

<!-- capsules -->

Open models and open records for looped mixture-of-experts language models, built and compared with [LoopTitan](https://github.com/datamllab/LoopTitan). Every model here comes with its configuration, its budgets against a standard MoE, its held-out results and the issue that records its run.

## LoopTitan Phase 1

From a standard MoE to a looped MoE one change at a time, each design trained with AdamW and with Muon on the same data, with the same schedule and seed, at matched parameters, training FLOPs and KV cache. Each model page shows where the model stands among the others.

<!-- models -->

Two designs are outside matched budgets on purpose: one more visit, and twice the experts. The study and its record: [results](https://github.com/datamllab/LoopTitan/blob/main/docs/results/phase1.md).
