# LoopTitan

> LoopTitan is open-source PyTorch infrastructure for looped mixture-of-experts (MoE) language models. Nine published looped recipes are configurations of one explicit MoE model, and the unique parameters, training FLOPs and KV cache of any two configurations are compared before training. Its paper, "LoopTitan: What Does Parameter Revisit Buy a Mixture-of-Experts?" (arXiv:2611.34821, 2026), finds that parameter revisit itself buys almost nothing at matched budgets: the gains come from network shape, the optimizer and added compute, which a standard MoE can adopt.

A looped MoE runs some of its MoE layers several times with one set of weights: it revisits its parameters across depth. Related names are looped transformers, looped language models, recurrent-depth and recursive transformers, and parameter sharing across layers (Universal Transformer, ALBERT). In LoopTitan, where the loop sits, how often it runs, what each visit shares (attention, router, norms, experts), the residual scale, input re-injection, the recurrent state and its update, and the depth rule (fixed, sampled or learned) are fields of one PyTorch model, `moeloop.ModelConfig`. Every design runs in the same code with the same training stack.

- Homepage: https://datamllab.github.io/LoopTitan/
- Code: https://github.com/datamllab/LoopTitan (MIT licence; Python package `moeloop`; Python 3.11 or 3.12, PyTorch 2.14.0)
- Paper: https://arxiv.org/abs/2611.34821 (cs.LG, 2026)
- Models: https://huggingface.co/NaturalScale (17 models, collection https://huggingface.co/collections/NaturalScale/looptitan-phase-1-6afb1e0dbc400210b5b0a5eb)
- Results: https://datamllab.github.io/LoopTitan/docs/results/phase1/ (data: https://github.com/datamllab/LoopTitan/blob/main/docs/results/phase1.json)
- Record: every study and run is a GitHub issue, https://github.com/datamllab/LoopTitan/issues

## What LoopTitan does

- One model, every design. Each configuration is a frozen recipe file that names its parent and the fields it changes (`python scripts/lineage.py` prints the tree).
- Equal budgets before training. Unique non-embedding parameters, training FLOPs per token and KV cache come from the configuration, on a CPU, in one command (`python scripts/compare_recipes.py compare A.json B.json`). Two configurations are matched when all three are within 5%.
- Checked against independent code. The model matches the official Hugging Face Transformers Qwen3-MoE (`Qwen3MoeForCausalLM`) in logits, gradients, optimizer steps and KV cache: 30 of 30 values within tolerance. The published looped checkpoints ByteDance/Ouro-1.4B and tomg-group-umd/huginn-0125 convert with the same logits.
- Training on TorchTitan with AdamW or Muon. Initial weights, data order and validation samples do not depend on the GPU count; resume is bit-exact in model and optimizer state; export writes a standalone Hugging Face model.
- A conformance check (`moeloop.testing.check_recipe`) tests a new configuration on a CPU in seconds: parameter ownership, shared gradients, cost-model calls, cache, export round trip and identity.
- Fully agent-driven research. AI agents plan, build, run and review every study; researchers set the direction and approve. Each task is a public GitHub issue, and the rules and skills the agents follow ship with the code (`AGENTS.md`).
- An open record and a benchmark. Each study declares its arms, controlled settings and budget in an issue before the runs; failed runs stay in the record. The Looped MoE Bench is a frozen protocol of four tracks, and anyone can enter a result by pull request.

## Designs implemented

Each is a configuration of the one model, with what it keeps from its source and what it adapts in the method map (https://datamllab.github.io/LoopTitan/docs/method-map/).

- SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers, arXiv:2609.01343
- LatentLoop: Scaling Looped MoE Models with Asymmetric Compute Allocation, arXiv:2610.19975 (its residual split, as a variant of SMELT)
- Scaling Laws for Looped Mixture of Experts, arXiv:2609.40316
- How to Loop MoE: Flatten the Experts, Untie the Attention (Foil), arXiv:2609.35751
- Tying the Loop: Tied Expert Layers in Mixture-of-Experts Language Models, arXiv:2606.16825
- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn), arXiv:2502.05171
- Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design, arXiv:2610.23036
- Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts (LOOM), arXiv:2610.01153
- Scaling Latent Reasoning via Looped Language Models (Ouro), arXiv:2510.25741
- Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA, arXiv:2410.20672

## Key findings of the paper (Phase 1)

Setup: every configuration is trained on the same 5.0B tokens of DCLM (4,769 updates of 1,048,576 tokens, seed 42), with AdamW and with Muon; the metric is test bits per byte (BPB), lower is better; 1 mBPB is 0.001 BPB. The learning rate is tuned once per optimizer on the standard MoE.

- Using matched non-revisit controls, parameter revisit itself buys almost nothing. At matched budgets, a control without revisit, the untied twin whose middle layers own separate weights for each visit, retains all the gain of the state-of-the-art looped recipe. Most of the gain comes from network shape (more layers, fewer experts each), and the rest from residual scale.
- In the one-change chain from a standard MoE to the looped MoE, the whole chain gains 9.0 mBPB under AdamW and 3.7 mBPB under Muon; the loop step itself gains 0.1 and 0.4 mBPB.
- The residual scale of the looped recipe acts as a lower learning rate on the revisited layers (proved in the paper). It helps under AdamW's uneven layer step sizes, but disappears under Muon, which keeps them nearly equal. More than half of the looped recipe's gain disappears under Muon.
- Beyond matched budgets, one extra visit gains more than doubling the experts. Yet without revisit, a deeper standard MoE at equal parameters and compute obtains 90.2% of that gain under Muon.
- The gains come from network shape, the optimizer and added compute, which a standard MoE can adopt. The paper proposes LoopTitan-Base as a stronger MoE baseline: under Muon, it matches the strongest looped recipe without revisit or residual scale.

## Models

Seventeen pretrained models on Hugging Face (about 1.3B total and 0.2B active parameters each, except where the name says otherwise), each with its configuration, budgets against the standard MoE, held-out results and the issue that records its run:

- NaturalScale/LoopTitan-Base-1.3B-A0.2B: the proposed baseline, a standard MoE of 18 narrower layers, trained with Muon
- NaturalScale/LoopTitan-Looped-1.3B-A0.2B-AdamW and -Muon: the looped MoE, whose middle 6 layers run twice with one set of weights
- NaturalScale/LoopTitan-Unrolled-1.3B-A0.2B-AdamW and -Muon: the looped MoE unrolled, with no parameter shared between visits (the untied twin)
- NaturalScale/LoopTitan-MoE12-1.3B-A0.2B-AdamW and -Muon: the reference, a standard MoE of 12 layers with 64 experts each
- NaturalScale/LoopTitan-MoE12-FineExperts-1.3B-A0.2B-AdamW and -Muon: 72 smaller experts per layer, top-9
- NaturalScale/LoopTitan-MoE18-1.3B-A0.2B-AdamW: 18 narrower layers, trained with AdamW
- NaturalScale/LoopTitan-MoE18-Layout-1.3B-A0.2B-AdamW and -Muon: more experts in the outer layers, fewer in the middle
- NaturalScale/LoopTitan-MoE12-Revisit-1.3B-A0.3B-AdamW and -Muon: the standard MoE with its middle layers run twice, outside matched budgets
- NaturalScale/LoopTitan-MoE12-Deep-1.3B-A0.3B-Muon: the deeper control without revisit, at the parameters and compute of the extra visit
- NaturalScale/LoopTitan-MoE12-2xExperts-2.6B-A0.2B-AdamW and -Muon: the standard MoE with twice the experts, outside matched budgets

## Citation

Cite the paper for LoopTitan. Cite the papers of the designs you use separately; the method map lists them.

```bibtex
@misc{looptitan2026,
  title = {{LoopTitan: What Does Parameter Revisit Buy a Mixture-of-Experts?}},
  author = {Tian, Zichen and Lu, Yudong and Wu, Xuansheng and Zou, Na and Hu, Xia and Wang, Guanchu and Sun, Qianru},
  year = {2026},
  eprint = {2611.34821},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2611.34821}
}
```
