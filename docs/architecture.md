# LoopTitan 架构

本文说明当前模型、recipe、训练与交付的边界。[配方](recipes.md)记录选定原文路径，[验证记录](validation.md)记录实际通过的配置。

## 1. 产品只围绕一个中心

**Open-source code for training and scaling recursive sparse models.**

研究者应能从一份短 recipe 看懂模型如何循环、共享哪些参数、梯度如何流动；修改一个机制后，沿同一条路径完成数据训练、DP/FSDP 放大、中断恢复、评测和模型导出。

#6 已交付四方法工程资格与普通 MoE/SMELT 的完整预训练对照；#7 增加随机初态、双 memory 与逐 token expected CE。新增四条路径已完成512步真实文本、两rank FSDP、完整恢复和独立HF，完整v2阶段验收通过；精确配置及范围见[验证记录](validation.md#新增训练动态与组合资格)。方法效果和 scaling 结论须由相应实验支持。

核心阅读路径固定为：

```text
*_spec() → 模型构造 → 显式 forward → 显式 objective
                           │
                 成本/比较 → TorchTitan 训练接入
                           │
                  checkpoint / eval / HF
```

概念树用来理解研究；代码用浅层组合实现。论文名出现在 recipe 与来源说明中。真正不同的更新方程采用短的专用 `forward`，不必强行变成基础模型的开关。

## 2. 最关键的设计：物理 block 与执行次数分开

一个 `Cell` 是一个实际持有参数的 Transformer/MoE block。重复调用同一个 Cell 就是参数共享。需要每轮独立的 attention/router/norm 时，把相应模块列表放在这个 Cell 里面；专家 bank 仍只有一个。

```text
LoopedLM
├── embedding
├── prefix: Cell 列表，执行一次
├── core: Cell 列表，循环执行
│   └── Cell
│       ├── attentions       一套，或每个 visit 一套
│       ├── attention_norms  一套，或每个 visit 一套
│       ├── ffn_norms        一套，或每个 visit 一套
│       ├── routers          一套，或每个 visit 一套
│       ├── ffn_output_norm  MemoryCell 的独立输出 norm
│       └── experts          一份实际参数
├── suffix: Cell 列表，每次读出执行一次
├── input_adapter           可选的明确输入注入
├── final_norm
├── lm_head                 唯一词表投影
└── exit_gate               可选的跨轮共享标量 gate
```

公开名称是 `LoopedLM`（`from moeloop import LoopedLM`）；`RecursiveLM` 是同一个类对象的兼容别名，旧脚本照常运行，不发警告。类在 [`model.py`](../src/moeloop/model.py) 中就定义为 `LoopedLM`，`RecursiveLM` 只是它的别名。

Dense 对照把稀疏分支替换为普通 FFN；always-on FFN 是独立的可选加法分支，不与“跨轮共享专家 bank”混用名称。首版 embedding/head 独立，不引入跨 FSDP 单元的词表权重别名。

`ModelConfig` schema 4 的 `prefix/core/suffix: RegionConfig` 只覆盖区域 expert/FFN 几何、top-k、always-on FFN 和 residual scale，保持共同 residual width 与 attention 几何。相同配置对象创建独立模块；共享来自对同一实际 owner 的重复调用。旧 schema 1/2/3 以原有默认行为读入，读入后按 schema 4 表示；新增状态/gate/注入默认关闭，旧权重 owner 不变；`diag_decay`/`orthogonal` 注入属于 schema 4，更低 schema 声明它们会被拒绝。配置兼容不等于跨源码的完整训练状态可以 strict resume；未知 schema 拒绝加载。

两种顺序直接保留为短循环：

```python
# stack: A → B → A → B
for visit in range(num_visits):
    for cell in self.core:
        state = cell(state, visit=visit)

# layer: A → A → B → B
for cell in self.core:
    for visit in range(num_visits):
        state = cell(state, visit=visit)
```

上面省略 mask、cache、统计返回，只展示真实执行顺序；实现中这些参数显式传递。不要先展开一张执行图再解释执行。

每份参数只有一个 canonical owner，初始化、optimizer、checkpoint、导出都按物理 owner 管理。每次调用的状态和 KV 则按逻辑位置管理。**共享参数不意味着共享运行状态。**

这是一项有意的表达边界：首版不提供任意跨远距离 Cell 的参数连线、任意共享图或自动分区规划。真实需求超出后，以一个明确的新模型解决，而不是提前加入 owner-map 语言。

## 3. 代表方法怎样落到同一套代码

| recipe | 构造与执行差别 | 选定语义 |
|---|---|---|
| 普通 Dense / MoE | `num_visits=1` | 非循环基线，资源匹配另算 |
| SMELT | 中间一半物理层执行两遍 | core 两 residual 各乘 1/2，外围为 1 |
| Meta middle loop | 独立 prefix/suffix、共享 core stack | 固定深度、full BPTT、terminal CE |
| Foil | core `(E,D,L)→(2E,D/2,2L)` | Base/Foil 都逐 visit 独立 attention，bank 内共享 router/norm/experts |
| Tying | 连续组用 layer loop，仅共享 expert bank | attention/router/norm 独立，expert LR=`base_lr/sqrt(g)` |
| Huginn-inspired MoE | live anchor、独立随机 latent、每 microbatch 采样深度 | 最后 `min(k,r)` 轮反传；固定深度 zero-init 评测是显式适配 |
| LOOM full-BPTT | `MemoryCell`，global/local normalized EMA，每轮独立 router | global 每 forward 清零，local 每轮清零；保持 memory 梯度 |
| Ouro Stage I | 共享 gate、每轮读出、逐 token expected CE−entropy | 所有轮都执行；HF/heldout 使用 terminal logits |
| Sandwich（Looped Models Part I） | 独立 prelude/coda 包围共享 core stack；每遍 core 前 `diag_decay` 注入 prelude 输出 | z_0 = e，固定深度、full BPTT、terminal CE；MoE 与 dense 两条路径 |
| Relaxed Recursive | 一个共享层 block 循环执行，无 prefix/suffix；每个 core Cell 每轮在共享权重上加一份 LoRA（`lora_attention_rank` / `lora_ffn_rank`） | `B = 0` 起步，初始等于普通循环模型；固定深度、full BPTT、terminal CE；MoE 路径只放松 attention，`dense=True` 另放松 FFN |

上表九个命名 recipe 的原文路径、共同路由/优化器/数据替换和 Foil Base 区别见[配方](recipes.md)。`per_visit_router` 与 `input_anchor` 仍是组件，不等于完整方法；`loom()` 从此版本指完整 memory 配方，旧组件用户应使用 `per_visit_router()`。bank 大小、初始化、归一化、路由、loss 与数据逐项声明。

结构 recipe 也不自动构成公平对照：`num_visits=1` 只说明非循环执行，不能证明参数量或计算量已匹配。具体对比在普通配置中写清物理 Cell/bank 数、执行调用数、bank/FFN 宽度和训练预算；不因此建立新的实验比较平台。

P1 的普通 MoE 为10个独立 Cell、d672、expert hidden544、E64/top8、heads7/KV1；SMELT 为10个物理 Cell（2 prefix、5 core执行两遍、3 suffix）、d576、E96/top8、heads9/KV1，共15次逻辑调用。实际 SMELT hidden440 由完整20.486B-label训练 prefix 的文档 packing 统计选定；最终数据、评价、源码和配对已冻结，两臂各完成19,537更新。精确[recipe JSON](recipes.md#p1-普通-moe--smelt)与[P1证据](evidence/p1.json)保留本次来源和结果。两臂共享151669词表和4096ctx；“100M active”指普通基线约98.5M的单次遍历 active body，独立 embedding/head 计入后模型总量约916.5M。选形与 document-mask 预算见[成本与比较](comparison.md)。

新的机制有三种合理接入：

1. **已有构件的新组合**：增加一个普通 recipe 函数。
2. **局部算子改变**：增加一个小模块，保持调用和参数归属明确。
3. **状态更新本身改变**：写短的专用 Cell/model forward，复用 attention、experts、训练与交付部件。

例如 CoE 的“attention 一次、FFN norm 一次、专家链多次”不应伪装成普通 block 重复；内部 `expert_rounds` 与外部 `num_visits` 分别命名。固定专家链可以复用 full-BPTT/terminal-CE，不能自动继承 stack 尾段反传的含义。新 forward 改变参数布局、cache 调用次数或状态结构时，补一小段对应的 parallelize/export 接入；不承诺任意模型零适配获得全部交付能力。多状态方法也不强塞进单张量状态。允许小量局部重复，等实际共同部分稳定后再抽取。

## 4. 核心模型保持普通 PyTorch

基础 Cell 的数学可以直接读：

```python
x = x + residual_scale * attention(attention_norm(x), positions, mask, cache_slot)
z = ffn_norm(x)
ids, weights, aux = router(z, valid_inputs)
update = experts(z, ids, weights)
return x + residual_scale * update, aux
```

具体签名可以在编码时调整；以下语义应稳定：

- attention：GQA、RoPE、pre-norm；头数/位置/归一化规则写入配置。先选一套明确默认值，QK-norm 等小变体由 recipe 显式选择。
- sparse：首版默认 softmax top-k、选中权重归一化、dropless、可选 always-on FFN。selection IDs、combine weights、有效 token 统计职责分开；dispatch counts 与去 padding 后的 aux counts 不混用。
- stage input injection：无注入默认。可选 `add`、`linear`（两个线性投影 `W_h h + W_e anchor`），以及 xllm-loop Part II 的 `diag_decay`（`exp(-dt·a)·h + dt·W e`，逐通道 `dt`、`a`）和 `orthogonal`（OrthoInj：先把衰减后的 `h` 投影到与 `q = dt·W e` 正交的子空间，再加 `q`）。公式与来源在 [`InputAdapter`](../src/moeloop/model.py) 的 docstring。在每个完整 stack visit之前应用，包括第一轮；初始state明确为input、zero或独立random latent。norm/residual次序不隐藏在万能boundary层。后两种默认从 `W = I`、`a = 1`、`dt = ln(5)/2` 起步（`ModelConfig` 的默认值，recipe 可以改写，sandwich 设 `dt = 1`）；参考模型的 `LoopedLM.initialize_weights` 按声明的值初始化，TorchTitan 路径的 `TitanModel.init_states` 直接调用它。
- 逐轮 LoRA（Relaxed Recursive）：core Cell 每个 visit 在共享的 attention 投影（`q/k/v/out`）和 dense 或 always-on FFN（`w1/w3/w2`）上各有一份 `B A x`，无 alpha 缩放，rank 分别由 `lora_attention_rank`、`lora_ffn_rank` 设定；路由专家、router 与 norm 不加。LoRA 是逐轮 owner，与 per-visit 模块一样按 `visit_capacity` 分配；prefix/suffix 解析后 rank 为 0。`A` 正态、`B = 0` 在其他参数之后初始化，同一 seed 下初始模型等于共享权重模型；TorchTitan 路径的 `TitanModel.init_states` 调用同一函数。第三方 `attention_factory` 只在开启 attention LoRA 时收到 `lora=`，约定见[扩展](extensions.md#替换-attention-与逐轮-lora)。
- default depth 是执行默认值；每轮独立模块的 capacity 决定权重形状。全共享结构不因更改执行深度重建权重。独立模块不得越界，cached generation 不得中途换深度。
- forward 不决定训练深度、不调用 optimizer、不更新持久 router 计数。随机 latent 在递归调用与 AC 重放之前产生一次；输出可微 aux sum/count 和 detached 诊断统计。普通公开调用返回 terminal logits；expected CE 训练显式返回每轮 hidden/gate，不常驻每轮的大词表 logits。
- training 不使用 KV cache；推理时每个 `(prefix/core/suffix, cell, visit)` 各有逻辑 KV slot。attention 权重即使共享，slot 也不合并。

不要把每个小操作都升级成公共插件协议。首发稀疏路由范围收敛在上述路径；bias balancing、group routing、capacity dropping 等有需要时增加明确实现和对应目标定义。

**循环主体与词表 head 必须能分开。** 大规模训练不应被迫物化完整 `[B,S,V]` logits。公开 forward/HF 正常输出 logits；TT 训练通过同一个整体 module 调用得到 final hidden 与 aux，接原生分块 head/CE。head 保持唯一参数 owner 和正常 FSDP 调用边界；aux 在完整 microbatch 的外层加一次，不按 logits chunk 重复添加。这是一个有限的训练接缝，不需要另造 loss/backend 框架。[原生 ChunkedLossWrapper](https://github.com/pytorch/torchtitan/blob/086bf6c166ec85c1298eb5596fa9bf95f6a2d840/torchtitan/components/loss.py#L567)。

`MemoryCell.forward` 直接写不同的状态更新：scaled attention 先写入两条 normalized EMA，再令 `h_tilde=h+r_global+r_local`，最后加 scaled、output-normalized FFN。不再额外加一次 attention，也不平均两条 memory。carry 是函数间显式传递的临时 Tensor/标量，不是模块 buffer。

Ouro 的每轮读出使用同一个 suffix/final norm/head，读出结果不写回循环 state。末轮接受剩余概率质量。训练入口使用 `expected_lm_loss` 的普通 head 调用与分块重算，保留 `p*dCE` 与 `CE*dp` 两条梯度；不能在旧 chunked CE 返回的标量外乘概率，因为旧 bridge 忽略外部 `grad_output`。

## 5. 训练动态必须看得见

默认训练仍是固定深度、完整 BPTT、最终输出 CE、AdamW；优化器可在 recipe 的 `optimizer` 中改为 Muon（见下文“优化器分组”）。训练规则在 `TrainingRecipe` 中明确选择：

- **训练深度采样**：从有限整数集合每个 optimizer update 采样一次；所有 DP ranks 和 accumulation microbatches 使用同一深度，依据 seed 和已完成 update 确定。固定深度始终是普通路径。
- **尾段反传**：stack loop 的前 `D-K` 个完整 visit 放在 `no_grad` 内，最后 K 个 visit 反传。anchor 若被后续注入使用，保留其梯度路径。无后续注入时，截断会切断最终 CE 到前缀/embedding 的路径；不偷偷添加直通梯度。
- **随机初态与深度**：Huginn-inspired 路径每 microbatch 使用 shifted Poisson-lognormal 深度（默认均值4、log std 0.5，无硬cap），各 ranks 同步，最后 `min(k,r)` 轮反传。独立 truncated-normal 初态与权重初始化分开；评测明确采用 fixed-depth、zero-init。
- **expected CE**：每 token、每轮 CE 按可微 exit probability 加权，并减去 exit entropy，再加独立 aux。训练完整执行所有轮，没有 early-exit 加速声明。

Layer loop 使用完整 BPTT。有限集合的 update-level 采样要求 `K <= min(depths)`；Poisson-lognormal 路径按已声明定义使用 `min(k,r)`。逐 Cell 截断与完整 stack 的尾段截断不是同一个算法。

为控制首版运行组合，采用以下清楚的支持集合：

| 训练结构 | 首版支持目标 |
|---|---|
| 普通 Dense/MoE、完全共享 stack loop | 固定深度、完整反传 |
| 完全共享 stack loop | 可选 update-level 深度采样、尾段反传 |
| 每轮独立 attention/router/norm，或逐轮 LoRA | 固定深度、完整反传 |
| layer loop / 连续 expert tying | 固定深度、完整反传 |
| live-anchor 随机 latent | microbatch-level Poisson-lognormal、尾段反传 |
| normalized EMA state | 固定深度、完整反传；支持独立 attention 的显式组合 |
| exit distribution | 固定深度、完整反传、逐 token expected CE/entropy |

独立 owner 结合随机深度在数学上可以成立；首版不把 unused 参数、变化的调用轨迹和截断同时加入资格范围。core 的容量语义保持清楚，扩展支持由后续实际需求驱动，不把“暂未资格验证”写成方法无效或 PyTorch 不可能支持。

损失只保留一个显式函数：

- 主 CE 按**整个 optimizer update 的全局有效 label 数**归一。
- 默认 MoE aux 按**单个 microbatch 的可微 router 调用**先取平均，再对整个 update 的逻辑 microbatch（global batch ÷ local batch）取平均。系数不会因循环次数增加而自动倍增；它也不等价于全局 pooled routing loss。
- input/router mask 和 label/loss mask 分开。零有效输入的局部调用贡献零 aux；全局无有效 label 的 update 拒绝执行。
- prefix 无梯度调用的专家使用量可以记录，但不会混进可微 aux 分母。模型重算不更新持久统计，训练器只在实际调用边界汇总一次。

这个选择让归约范围可读，也明确了公平比较的一个限制：改变 microbatch geometry 可能改变非加性 aux，即使全局 batch 相同。global batch 与 local batch 都固定、且训练深度规则也与卡数无关（固定深度或 `uniform_update_v1`；`poisson_lognormal_micro_v1` 不在此列，其深度序列取决于累积步数）、初始状态也不按 rank 抽取（随机 latent 初态用各 rank 自己的 RNG，目前只有 `huginn_spec` 如此，它已被其深度规则排除）时，aux 与卡数无关（数据流同样与卡数无关，见第 7 节）；这一点的 GPU 验证见 #49。不能只记录一个 balance coefficient 就宣称训练目标完全对齐。

Huginn的尾段梯度与全轮expected loss、EMA与段间optimizer step等未定义组合会明确拒绝。Ouro Stage II/token early exit和额外多状态机制仍是后续独立扩展，不预建通用引擎。

**优化器分组按物理 owner。** [`optim.py`](../src/moeloop/optim.py) 的 `optimizer_config` 把 recipe 的 `optimizer` 映射成 native param groups，每个物理参数恰属一个 group；共享 owner 的梯度已是各次调用之和，每步只更新一次。AdamW 是“声明的 zero-decay 角色”加“其余参数”两组。Muon 时，由 `MUON_PATTERN` 这一条规则决定哪些参数交给 TorchTitan 的 DistMuon（简言之：隐藏矩阵（见 `MUON_PATTERN`）；embedding、lm_head、router、exit gate、norm 和 1D 参数仍由 AdamW 更新），规则的准确内容以 `optim.MUON_PATTERN` 为准。每个专家矩阵单独正交化。core 专家带 `expert_lr_multiplier`，所以单独用第二个 DistMuon 实例，权重衰减不补偿。新增矩阵类 owner 必须加入 `MUON_PATTERN`，`tests/training/test_muon.py` 对每个 recipe 构造函数核对这条规则。

Muon 的限制：DistMuon 只接受 CUDA 上的 DTensor，Muon recipe 因此只走原生 FSDP 的 GPU 路径；每个 Muon 参数每步都要有梯度，未被调用的 owner 会直接报错。CPU 测试只在单 rank 上执行真实的一步；#49 在 4 卡与 8 卡 FSDP 上完成了 Muon 的训练、完整 checkpoint 保存与 WSD 分叉；HSDP `dp_replicate>1`、world 大于专家数的分片边界、空 token 专家和 Muon checkpoint 续训未在该次运行中覆盖。

成本区分唯一参数、实际调用与逻辑KV。固定expected CE使用 `fixed_expected_matrix_v1`（有document pairs时为document版本），计每轮head/coda和head checkpoint重算；采样深度与尾段梯度使用 `sampled_depth_matrix_v1`（有document pairs时为document版本）：训练矩阵 FLOPs 是声明深度分布下的期望，`flops.training_depth` 另给 p50/p95 深度下单次的成本；它们不是某次运行的实际深度或实测工作量，默认深度的nominal分解只是一次固定前向。详见[成本与比较](comparison.md)。

## 6. TorchTitan 接入：保留原生训练，把必要差异集中起来

首版继续使用已核查的 TorchTitan `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`；`train` extra 现在从 PyPI 安装 `torchtitan==0.3.0`，即该 commit 的 tag v0.3.0。换上游版本可以后续单独做；不同时追赶新训练引擎 API。

| TorchTitan 继续负责 | 本项目明确负责 |
|---|---|
| 分布式环境、DP/FSDP、梯度 accumulation | LoopedLM 构造、重复调用与梯度窗口 |
| clipping、optimizer（含 DistMuon）、scheduler、训练循环 | 纯路由、loss/aux 定义、按物理 owner 的优化器分组、来源清楚的 recipe |
| checkpoint 调度与 DCP 保存 | 完整 data/RNG/实际进度状态、严格恢复契约 |
| profiling、原生日志和指标输出 | 必要的深度/路由/有效 token 诊断 |

`titan.py` 提供一个 ModelSpec、薄 BaseModel/Trainer/Validator 接入和有限 parallelize 函数。需要覆盖的钩子仅限已定位的真实差异：meta/materialize/init、传入本批 D/K 和 mask、分块 head/CE 与 aux/metrics、完成进度与恢复状态。保留一小段清楚的本地覆写，比多层 callback 更易维护；这些覆写都要对上游 pin 做审查。

首版显式选择 `spmd_backend="partial_dtensor"`，TP/EP/CP/PP 均为 1；激活保持普通 Tensor，参数使用原生 DP/FSDP。避免把 plain PyTorch core 升级成整棵 SPMD annotation 类型树。DP 也采用该 pin 的原生 Replicate/FSDP 归约契约；任意 stock DDP 不是可以不改 loss scaling 就互换的入口。

昂贵专家执行复用选定的原生 `RoutedExperts`，我们的 Cell/路由/目标语义保持不变。普通 PyTorch 参考实现供 CPU、HF 与语义对照。两种执行路径之间只有有限权重映射与数值契约，不建立可任意替换的 backend 平台。

这个接缝有源码依据：原生 `MoE.forward` 会在训练中更新 token counters，即使关闭 bias balancing；直接复用整个 MoE 会把 activation-checkpoint 重算混进统计。原生 Trainer 也会丢弃 loss 返回的 metrics，所以不能只换一个 loss 类就声称接入完整。[原生 MoE](https://github.com/pytorch/torchtitan/blob/086bf6c166ec85c1298eb5596fa9bf95f6a2d840/torchtitan/models/common/moe.py)，[原生 Trainer](https://github.com/pytorch/torchtitan/blob/086bf6c166ec85c1298eb5596fa9bf95f6a2d840/torchtitan/trainer.py)。

FSDP 以明确的参数所有者为单位。全共享 Cell 是自然边界；存在每轮独立 attention 时，将这些子模块单独划分，避免每次调用都收集所有轮的 attention。词表 head 也有独立调用边界，以服务分块 loss。初始化只由 seed 决定：`TitanModel.init_states` 调用 `LoopedLM.initialize_weights`，用 run seed（`debug.seed`）播种的 CPU 生成器按 canonical 顺序逐个整块抽取参数，每个 rank 只写入自己的分片，不通信，一次只物化一个完整参数；所以第 0 步权重与卡数和 GPU 型号无关。具体 wrapping/activation-checkpoint 的配合需在实现中审查并在统一批次资格验证。

首版支持目标为 DP/FSDP。expert execution 边界、canonical 参数身份和局部 parallelize 给以后 EP/TP 留下合理接入位置；今天不设计通用并行规划器，也不声称支持尚未实现的轴。

## 7. 数据、恢复、导出是产品主干

**数据**：普通配置明确 tokenizer、不可变来源、source weights、packing、位置与 document mask。每个 source 先形成等长 block，再按权重选择。所有 rank 打包同一条逻辑 block 流，它只依赖数据、seed、权重和 `seq_len`；每个 rank 走完整条流，只为自己的 block 读 token。一次 update 的 global batch 个 block 按 rank 连续切片（rank r 取其中第 `[r·K, (r+1)·K)` 个，K 为每 rank 的 block 数），所以每次 update 的数据和顺序与卡数无关。原生验证也按 global batch 计：每次读固定数量的 global batch（默认 2；设置 `validation_label_tokens` 时向上取整到完整 global batch），按同一规则切分，每 rank 的步数是该数乘梯度累积步数，所以任意卡数验证同一批 block。实现见 [`data.py`](../src/moeloop/data.py) 的 `PackedDataset`。P1 公开 DCLM 路径先冻结原始 Parquet 的 revision、有序路径及 size/SHA，再准备默认24B train、10M validation、10M test有效 labels；实际供给由完成 manifest 证明。历史预分词 DCLM 的300M/1M/1M fixture 保留为工程 smoke，以 parent manifest/chunk SHA 标识，保留原 EOS、前加本库 BOS；它不覆盖 P1 终点。WikiText2 是安装 demo，FineWeb-Edu 为显式可选路径。准备命令、tokenizer 适配与许可见[训练与数据](training.md)。

**恢复**：在 optimizer update 边界保存模型、optimizer、scheduler、每 rank RNG、数据 cursor/部分 packing buffer、深度采样规则与已完成 update/token 计数。首版严格恢复限定相同拓扑和数据身份：逻辑流本身与卡数无关，但数据游标记录 rank、world size 和每 rank block 数，换卡数会被拒绝。评测 RNG 与训练隔离；durable checkpoint 才能作为恢复点。更换拓扑或数据属于明确的新分支，不假装原运行继续。

**评测**：heldout 给出 CE 和字节精确的 BPB（[`evaluate.py`](../src/moeloop/evaluate.py)）。BPB 的字节表来自 manifest 的 tokenizer 资产（[`tokens.py`](../src/moeloop/tokens.py)），特殊 token 同时退出分子和分母；manifest 没有 tokenizer 资产或 tokenizer 不是 byte-level BPE 时，BPB 为空并在 `bpb_unavailable` 里写明原因，不猜字节数。CORE（DCLM v2 口径，22 个任务的 centered accuracy 均值）由 [`core.py`](../src/moeloop/core.py) 对本地 HF 导出目录运行。lm-evaluation-harness 保留为现成的下游工具。记录模型深度、tokenizer、任务版本和 mask。复用现成评测生态，不造 benchmark 平台。使用方法见[训练与数据](training.md#导出评测和生成)。

**HF 导出**：输出物理权重、config、tokenizer、generation config、必要源码与来源元数据。轻量 wrapper 使用同一 core；只有 PyTorch/Transformers 的离线进程通过 AutoClass 加载，无需 TorchTitan。full-prefix、固定深度生成与已验证 cache 路径分别声明资格；不能从旧模型外推新增 recipe 已通过。加载自定义源码使用 `trust_remote_code=True`。cache 归逻辑调用并固定 session 深度；近似 KV 方法另行命名。[Transformers 自定义模型机制](https://huggingface.co/docs/transformers/en/custom_models)。

Core 只知道存放 Tensor 的具体 `DecodeCache`；HF Cache/Generation 类型停留在 wrapper。wrapper 主动创建与本次逻辑调用数一致的 cache，避免 HF 按物理层数误分配。它只是同一份 KV 的接口转换，不复制一套推理模型或 KV 存储。

Huginn-inspired 与LOOM首条交付路径固定深度、无cache；Huginn评测使用确定性零初态。Ouro的HF `labels` 仍表示标准next-token terminal CE，不是Stage I训练目标。新增路径的实际native/恢复/HF资格以验证页为准。

训练事实使用几份普通文件记录 resolved config、metrics、outcome；CVSM/MLflow 需要时消费它们。公共模型/训练不需要 GitHub Issue、MLflow 服务或 CVSM 状态机。

## 8. 文件保持扁平，阅读保持局部

```text
src/moeloop/
  recipes.py        方法 *_spec 和有限 native trainer factory
  recipe_spec.py    普通可序列化方法/训练/来源记录，不导入 TT
  costs.py          owner、矩阵 FLOPs（含采样深度的期望）、逻辑 KV 估计
  comparison.py     显式比较条件、差异和有限配对
  model.py          LoopedLM（`RecursiveLM` 是它的别名）、Cell、attention、显式 recurrence
  moe.py            纯 router、参考 experts、清楚的路由统计
  training.py       深度选择、梯度规则、objective
  latent.py         独立 latent 初态、按 microbatch 定位的深度采样
  memory.py         无持久 buffer 的 normalized EMA 与缩放系数
  halting.py        exit probabilities、分块 expected CE/entropy
  titan.py          版本固定的 TT 生命周期、parallelize、原生专家绑定
  data.py           source/packing/mixture/stateful data 接口
  prepare.py        显式、不可覆盖的公开/本地数据准备
  train.py          薄训练入口
  hf.py             HF config/model/cache 接入
  export.py         native checkpoint → portable HF
  evaluate.py       heldout CE/BPB 与既有评测工具接入
  tokens.py         byte-level BPE 的逐 token 字节长度
  core.py           CORE（DCLM v2 口径）评测
  optim.py          Muon/AdamW 按物理 owner 的参数分组
  diagnostics.py    逐轮 logit lens、残差、路由重叠、有效深度
  testing.py        第三方 recipe 的一致性测试 check_recipe
  bench.py          Looped MoE Bench 的 track 与提交核对器
recipes/
  ...               普通 Python 函数，组合模型/训练/数据/原生 TT 配置
eval/bench/
  tracks.json       Bench 的冻结 track（见比较文档）
src/moeloop_lab/
  protocol.py       可选内部实验协议，独立于公共 import 路径
```

这表达职责，不强制准确文件数。attention/cache 若明显膨胀，可拆一个具体文件；不要为了“扁平”把几个职责塞成巨文件。配置使用普通 dataclass 加 TT 原生 Python config；不再做 YAML/Hydra DSL、自动插件发现或全局配置注册服务。函数的模块路径已经足够作为 recipe 入口。

## 9. 实现边界

当前实现保持以下边界：

1. 小 core、显式循环、nested Cell 参数所有权、论文 recipe。
2. 命名方法均指定具体原文路径与公开适配；代码接口和实际GPU资格分开，旧冻结实验保留原身份。
3. 原生 TorchTitan DP/FSDP 的完整训练路径，加严格恢复与独立 HF 交付。
4. 同一 repo，公共 `moeloop` 与可选 `moeloop_lab.protocol` 分离。

局部实现、文件拆分和优化服务于真实方法语义。新参数 owner、运行状态、梯度或目标必须经过相应语义检查与实际训练/恢复/导出验证；不能用配置可构造代替运行资格。

当前实现与贡献方式见[开发指南](development.md)，不同状态更新的例子见[FFN-chain 扩展](extensions.md)。
