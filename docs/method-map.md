# 方法到机制的映射

发布路径采用下表九项，论文名只进入 recipe/source；通用 core 按实际状态更新命名。每项的 `source`、`preserved`、`adaptations` 可从 [`recipes.py`](../src/moeloop/recipes.py) 直接导出。它们共用公开文本、softmax top-k dropless MoE 和 AdamW（默认；Muon 可选，见[训练](training.md#修改研究机制)）；例外是两条稠密路径 `sandwich_spec(dense=True)` 与 `relaxed_spec(dense=True)`，它们的 FFN 全是稠密 SwiGLU，没有 MoE 和 aux；这些替换不建立原论文完整训练复现。`LoopedLM` 是模型类的公开名称，`RecursiveLM` 是同一个类对象的兼容别名，类在 `model.py` 中就定义为 `LoopedLM`，下表链接指向该文件。

| 方法与冻结原始来源 | 实际 forward / objective | 保留与明确适配 | 可核验证据 |
|---|---|---|---|
| [SMELT v3](https://arxiv.org/html/2609.01343v3) §3/4、附录A/B | `smelt_spec` → [`LoopedLM.forward` / `Cell.forward`](../src/moeloop/model.py)、[`objective`](../src/moeloop/training.py) | 中段两遍、core residual 1/2；公开数据/初始化/优化栈，P2 E88 为预算适配 | [P1](evidence/p1.json)、[V1四法资格](evidence/recipes-v1.json)；[P2配对](evidence/recipes-v3.json) |
| [Meta Looped MoE v1](https://arxiv.org/html/2609.40316v1) C.1/C.2 | `meta_spec` → [`LoopedLM.forward`](../src/moeloop/model.py)、terminal CE | 独立 prelude/coda、共享 middle、full BPTT；共同 routing/dispatch/AdamW、独立 embedding/head | [V1资格](evidence/recipes-v1.json)、[`recipe回归`](../tests/model/test_recipes.py) |
| [Foil v1](https://arxiv.org/html/2609.35751v1) §2.1/2.2 | `foil_spec` → [`Cell.forward`](../src/moeloop/model.py) | `(E,D,L)→(2E,D/2,2L)`、逐 visit attention；共同训练底座 | [V1资格](evidence/recipes-v1.json)、[`tests/reference`](../tests/reference) |
| [Tying v1](https://arxiv.org/html/2606.16825v1) §3/附录B | `expert_tying_spec` → layer-order [`LoopedLM.forward`](../src/moeloop/model.py)、[`titan.py`](../src/moeloop/titan.py) expert LR | 连续 experts 共享、其他组件独立、expert LR `/sqrt(g)`；recipe 默认 AdamW（原文 Muon），`optimizer.name` 可改为 Muon | [V1资格](evidence/recipes-v1.json)、[`tests/training`](../tests/training) |
| [Huginn v2](https://arxiv.org/html/2502.05171v2) §3/4.1 | `huginn_spec` → [`initial_latent` / `sampled_depth`](../src/moeloop/latent.py)、[`LoopedLM.forward`](../src/moeloop/model.py) | live anchor、随机 latent、microbatch深度、末段梯度；稀疏pre-norm替换原dense布局；zero-init固定深度评测 | [V2资格](evidence/recipes-v2.json)、[`tests/model`](../tests/model)；HF无cache |
| [LOOM v1](https://arxiv.org/html/2610.01153v1) | `loom_spec` → [`MemoryCell.forward`](../src/moeloop/model.py)、[`ema_update`](../src/moeloop/memory.py) | 双EMA、先写后读、逐轮router、重注入；full BPTT选定路径，无段间optimizer step | [V2及LOOM×attention资格](evidence/recipes-v2.json)、[`tests/model`](../tests/model)；HF无cache |
| [Ouro v5](https://arxiv.org/html/2510.25741v5) | `ouro_spec` → [`expected_lm_loss`](../src/moeloop/halting.py)、[`objective`](../src/moeloop/training.py) | Stage I逐token expected CE减entropy、gate/head/backbone联合训练；terminal CE评测，未交付Stage II/early exit | [V2资格](evidence/recipes-v2.json)、[`tests/hf`](../tests/hf) |
| [Looped Models Done Right, Part I](https://arxiv.org/html/2610.23036v1) §2.1/3.1/3.4.2/4.1/附录A | `sandwich_spec` → [`InputAdapter`](../src/moeloop/model.py)、[`LoopedLM.forward`](../src/moeloop/model.py) | prelude/core/coda、每遍 core 前 `diag_decay` 注入、`z_0 = e`、full BPTT、terminal CE；MoE 路径取发布代码的 input-initialized 变体，`dense=True` 为 §3.4.2 稠密路径 | [`tests/model/test_sandwich_recipes.py`](../tests/model/test_sandwich_recipes.py)；暂无 GPU 运行证据 |
| [Relaxed Recursive Transformers v3](https://arxiv.org/html/2410.20672v3) §2.2/2.3/3.5/附录C | `relaxed_spec` → [`LoRA` / `Cell.forward`](../src/moeloop/model.py)、terminal CE | CYCLE 共享 block、每层每轮一份 LoRA（`W'x + BAx`，无 alpha）、`B = 0` 零初始化、full BPTT；MoE 路径只放松 attention（路由专家不加 LoRA），`dense=True` 为论文的全线性层路径；从随机权重训练而非 uptraining | [`tests/model/test_relaxed.py`](../tests/model/test_relaxed.py)；暂无 GPU 运行证据 |

数值/梯度、原生训练/恢复与 HF 证明各有不同边界，见[验证矩阵](validation.md)和[参考baseline](baselines.md)。P1/P2只比较普通MoE/SMELT，C只比较固定expert pool的attention/router owner；前七条的工程资格不等于七条效果结论，sandwich 与 relaxed 尚无工程资格。P1/P2各两臂已完成相同20.486B labels终点与完整downstream；C四臂已完成有界共同终点。ARC方向混合、C参数增量及单seed限制见[验证记录](validation.md#v3第二尺度组合与交付)；不按结果宣布最终recipe。公开证据引用其原始source snapshot，不为统一发布commit重新训练。

逐轮行为用 [`moeloop.diagnostics`](../src/moeloop/diagnostics.py) 读（logit lens、残差更新、路由重叠、有效深度）；跨方法比较的冻结协议是 Looped MoE Bench，见[成本与比较](comparison.md#looped-moe-bench)。

## 相邻方法的调研入口

下表47项包含基线、相邻范式和边界例，仅记录文献阅读/原代码审阅深度，不是47项实现或训练资格。九项发布路径的实际运行证据以上表为准，其余条目不自动进入支持集合。

每项工作同时关联多个机制轴：状态/布局、调用顺序、组件共享、条件选择、训练目标/梯度、推理缓存。下表提供入口。

| 方法/工作 | 抽取的核心机制 | 实际证据深度 | 详细记录 |
|---|---|---|---|
| [ACT](https://arxiv.org/abs/1603.08983) | 加法累计停机质量、余量与加权输出 | 论文方法 | loop 轨静态审阅 |
| [Universal Transformer](https://arxiv.org/abs/1807.03819) | 共享深度变换、time signal、ACT变体 | 论文+官方核心代码 | loop 轨静态审阅 |
| [ALBERT](https://arxiv.org/abs/1909.11942) | 分组/组件级参数共享 | 论文+官方核心代码 | loop 轨静态审阅 |
| [PonderNet](https://arxiv.org/abs/2107.05407) | hazard退出分布、expected loss、prior KL | 论文方法 | loop 轨静态审阅 |
| [Huginn](https://arxiv.org/abs/2502.05171) | prelude/core/coda、anchor注入、随机初态、suffix梯度 | 论文+官方核心代码 | loop 轨静态审阅 |
| [Ouro / LoopLM](https://arxiv.org/abs/2510.25741) | full-loop、hazard目标、分阶段gate训练 | 论文+官方HF代码；训练源码未确认 | loop 轨静态审阅 |
| [HRM](https://arxiv.org/abs/2506.21734) | H/L双状态、独立bank、双时钟、carry | 论文+官方核心代码 | loop 轨静态审阅 |
| [TRM](https://arxiv.org/abs/2510.04871) | answer/latent双状态、共享bank、末周期梯度 | 论文+官方核心代码 | loop 轨静态审阅 |
| [LoopFormer](https://arxiv.org/abs/2602.11451) | t/dt条件调制、完整与shortcut轨迹一致性 | 论文+官方核心代码 | loop 轨静态审阅 |
| [T-LoopFormer](https://arxiv.org/abs/2609.15160) | token深度预算、稀疏轨迹、带孔cache | 论文+官方核心代码 | loop 轨静态审阅 |
| [xLLM-Loop Part II](https://arxiv.org/abs/2610.06833) | 全局深度prior、注入、TBPTT、terminal-KV | 论文+官方核心代码 | loop 轨静态审阅 |
| [RingFormer](https://arxiv.org/abs/2502.13181) | 共享base加depth-specific低秩投影 | 论文方法；代码未审 | loop 轨静态审阅 |
| [Latent Recurrent Transformer](https://arxiv.org/abs/2605.26797) | token-time memory；depth recurrence的边界例 | 论文方法 | loop 轨静态审阅 |
| [Universal Transformers Need Memory](https://arxiv.org/abs/2604.21999) | memory/register tokens、ACT | 论文方法；puzzle设定 | loop 轨静态审阅 |
| [RecurTrace](https://arxiv.org/abs/2609.03379) | 同token位置的历史loop状态窗口 | 论文方法 | loop 轨静态审阅 |
| [One Model to Loop Them All / ELR](https://arxiv.org/abs/2610.12582) | gated delta update、预算调制、route/cache复用 | 论文方法；未审代码 | loop 轨静态审阅 |
| [Coconut](https://arxiv.org/abs/2412.06769) | 新latent token位置、hidden回灌、KV carry | 论文+官方核心代码 | loop 轨静态审阅 |
| [PonderLM](https://arxiv.org/abs/2505.20674) | 词表概率加权embedding回灌 | 论文方法 | loop 轨静态审阅 |
| [PonderLM-2](https://arxiv.org/abs/2509.23184) | 交错latent slots与Jacobi refinement | 论文方法 | loop 轨静态审阅 |
| [Foil](https://arxiv.org/abs/2609.35751) | 减物理层/扩大专家bank/增visits、独立attention | 论文+官方核心代码 | moe 轨静态审阅 |
| [LOOM](https://arxiv.org/abs/2610.01153) | per-loop router、残差/注入、段间优化 | 论文+官方核心代码 | moe 轨静态审阅 |
| [Tying the Loop](https://arxiv.org/abs/2606.16825) | 相邻逻辑层共享专家参数，其他组件独立 | 论文+官方核心代码 | moe 轨静态审阅 |
| [SMELT](https://arxiv.org/abs/2609.01343) | 中段循环与宽度/专家/GQA/残差配方 | 论文方法；代码未审 | moe 轨静态审阅 |
| [LoopMoE](https://arxiv.org/abs/2606.04438) | iteration+state条件归一化与容量配方 | 论文方法；官方代码未定位 | moe 轨静态审阅 |
| [LatentLoop](https://arxiv.org/abs/2610.19975) | 压缩与完整token的不对称两遍计算 | 论文+官方核心代码 | moe 轨静态审阅 |
| [Sparse Layers are Critical](https://arxiv.org/abs/2605.09165) | 稀疏router的跨loop路径多样性 | 论文方法；代码未审 | moe 轨静态审阅 |
| [MoEUT](https://arxiv.org/abs/2405.16039) | SigmaMoE+SwitchHead、selector-only norm | 论文+官方核心代码 | moe 轨静态审阅 |
| [Chain-of-Experts](https://arxiv.org/abs/2506.18945) | attention一次、FFN专家链、多轮router | 论文+官方核心代码 | moe 轨静态审阅 |
| [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) | token/recursion-choice、attention域与cache | 论文+官方核心代码 | moe 轨静态审阅 |
| [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | fine-grained与always-on shared experts | 摘要；相关V3算子代码审计 | moe 轨静态审阅 |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | selection bias、group限制、原分数combine | 官方inference核心代码 | moe 轨静态审阅 |
| [Qwen3 MoE](https://arxiv.org/abs/2505.09388) | 普通token-choice top-k基线 | 官方Transformers代码；论文元信息 | moe 轨静态审阅 |
| [OLMoE](https://github.com/allenai/OLMoE) | 普通细粒度MoE与可配置prob renorm | 官方Transformers核心代码 | moe 轨静态审阅 |
| [Expert-Choice Routing](https://arxiv.org/abs/2202.09368) | expert维度固定容量选择token | 论文方法；代码未审 | moe 轨静态审阅 |
| [Loop the Loopies](https://arxiv.org/abs/2607.16051) | layer-loop与实测吞吐配方 | 论文方法；代码未审 | moe 轨静态审阅 |
| [Closing the Loop](https://arxiv.org/abs/2610.00673) | warmup/exit entropy/data阶段与norm消融 | 论文方法/消融；代码未确认 | root 轨静态审阅 |
| [Loop-μP](https://arxiv.org/abs/2606.18524v3) | loop与unique-depth分别参数化 | 最新版论文方法/实验范围；代码未审 | root 轨静态审阅 |
| [Parcae](https://arxiv.org/abs/2604.12946) | 受约束对角注入、prelude norm、per-sequence采样 | 论文+官方核心代码 | root 轨静态审阅 |
| [STARS](https://arxiv.org/abs/2605.26733) | Jacobian相关正则与随机深度 | 论文+官方核心代码 | root 轨静态审阅 |
| [Readout Blind Spot](https://arxiv.org/abs/2606.24898) | 读出可见性、状态尺度、loss位置 | 论文+官方核心代码 | root 轨静态审阅 |
| [Growth / Boundary Operators](https://arxiv.org/abs/2609.19107) | 边界norm+anchor、共享/独立growth | 论文+官方核心代码 | root 轨静态审阅 |
| [What Limits Recursive Reasoning](https://arxiv.org/abs/2609.39967) | 梯度位置、bounded/gated update、物理batch | 论文方法/算法任务消融 | root 轨静态审阅 |
| [MixerLoop](https://arxiv.org/abs/2608.18230) | 只重复mixer、保留一次FFN | 论文方法；代码未审 | root 轨静态审阅 |
| [Scaling Laws for Looped MoE](https://arxiv.org/abs/2609.40316) | recurrence/sparsity与预算的分离 | 论文部分章节；非实现审计 | root 轨静态审阅 |
| [LoopRPT](https://arxiv.org/abs/2603.19714v2) | EMA教师、hard-token、噪声rollout与多目标 | 最新版论文方法/消融；代码未审 | root 轨静态审阅 |
| [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) | root solve与implicit differentiation | 论文方法/定理；代码未审 | root 轨静态审阅 |
| [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) | token轴容量选择；与MoR连接 | 论文基础机制；深层代码证据来自MoR | root 轨静态审阅 |

Runtime另行核查 TorchTitan pinned/current、Transformers、flame，并复核 Foil/LOOM/xLLM 的训练与交付路径，见 [TorchTitan](https://github.com/pytorch/torchtitan) 与各项目的公开实现。

效果证据按原论文任务、规模、预算和消融范围保留；只读摘要或未找到代码的条目不会被升级为已验证机制。
