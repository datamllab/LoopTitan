# 方法配方与阅读入口

[`moeloop.recipes`](../src/moeloop/recipes.py) 的 `smelt_spec()` 等函数返回普通 [`RecipeSpec`](../src/moeloop/recipe_spec.py)，不导入 TorchTitan；公开 `trainer_config(spec)` 或去掉 `_spec` 的内置 factory 才构造 native trainer 配置。阅读顺序是 spec → [`LoopedLM.forward` / `Cell.forward`](../src/moeloop/model.py) → [`training.objective`](../src/moeloop/training.py)。`LoopedLM` 是公开名称，`RecursiveLM` 是同一个类对象的兼容别名（不发警告），类在 `model.py` 中就定义为 `LoopedLM`。

```python
import json
from moeloop.recipes import foil_spec
from moeloop.recipe_spec import RecipeSpec

spec = foil_spec(num_experts=4, core_layers=2, passes=2,
                 vocab_size=128, dim=32, hidden_dim=64,
                 num_heads=4, num_kv_heads=2, max_seq_len=64)
restored = RecipeSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
print(restored.source["selected_E_D_L"])  # [8, 1, 4]
print(restored.model.core.num_experts)  # 8
```

`RecipeSpec` schema 1 中，`model` 描述参数和执行结构，`training` 描述目标/梯度/深度；`source`、`preserved`、`adaptations` 说明原文与适配，`optimizer`、`inference`、`run` 记录运行默认值，`evidence` 留给真实产物身份。空 evidence 和 `run.successful_updates=512` 都不是完成证据。

## 九条选定路径

前四项已有历史工程资格；SMELT另有下文完整预训练对照。新增Huginn-inspired、LOOM、Ouro及LOOM×attention均已通过512步两卡FSDP、各自完整恢复和独立HF；工程轨迹及范围见[验证记录](validation.md#新增训练动态与组合资格)和[机器可读证据](evidence/recipes-v2.json)。最后的 sandwich 与 relaxed 只有 CPU 上的 recipe 与一致性测试，没有 GPU 运行或证据记录。

| 方法与入口 | 保留语义 | 共同底座适配 |
|---|---|---|
| [SMELT v3 §3/附录 A](https://arxiv.org/html/2609.01343v3)，`smelt_spec(physical_layers=8)` | prefix/core/suffix 物理层数 2/4/2；core 完整执行两遍；core 的 attention、FFN residual 各乘 1/2，外围为 1 | 公开文本、共同 sparse cells 与 AdamW；普通 MoE 预算配对另定义 |
| [Meta Looped MoE v1 C.1–C.2](https://arxiv.org/html/2609.40316v1)，`meta_spec(core_layers=2, passes=4)` | 独立 prelude/coda，中段共享循环；固定深度、完整 BPTT、terminal CE | 共同 softmax top-k、选中权重归一化、dropless、AdamW 和独立 embedding/head；不声称验证 scaling law |
| [Foil v1 §2.1–2.2](https://arxiv.org/html/2609.35751v1)，`foil_spec(num_experts=4, core_layers=4, passes=1)` | core `(E,D,L)→(2E,D/2,2L)`，D 是物理 bank 数；每逻辑 visit 独立 attention，bank 内共享 experts/router/norm | 共同 cell、公开数据和优化器；保留独立 prelude/coda，外围 expert 数为原 E |
| [Tying v1 §3/附录 B](https://arxiv.org/html/2606.16825v1)，`expert_tying_spec(groups=2, group_size=4)` | 每组 g 个连续逻辑层仅共享 experts，attention/router/norm 逐层独立；expert LR=`base_lr/sqrt(g)`，其他 LR 不变，WD 系数不补偿 | recipe 默认用 AdamW，原文背景是 Muon；`optimizer.name` 改为 `"Muon"` 时 core 专家仍按 `expert_lr_multiplier` 缩放 LR，WD 不补偿；按实际 canonical core expert owner 分组 |
| [Huginn v2 §3/4.1](https://arxiv.org/html/2502.05171v2)，`huginn_spec()` | 独立随机latent、live prelude anchor、microbatch深度采样、最后 `min(k,r)` 轮反传、独立coda | 公共pre-norm MoE替代原dense/sandwich背景；小模型采样参数显式；fixed-depth zero-init评测、无cache |
| [LOOM v1 §3/4.2.1](https://arxiv.org/html/2610.01153v1)，`loom_spec()` | 两条normalized EMA、embedding重注入、残差缩放、每轮独立router；full BPTT | 首条路径固定H=3；无段间detach/optimizer step；无cache |
| [Ouro v5 §3.2–3.4](https://arxiv.org/html/2510.25741v5)，`ouro_spec()` | 逐token逐轮expected CE−entropy；共享gate、末轮接收剩余概率、完整反传 | 共享gate零初始化、entropy系数0.1为公开栈选择；HF只提供固定深度terminal读出 |
| [Looped Models Done Right, Part I (v1) §2.1/3.1/3.4.2/4.1/附录 A](https://arxiv.org/html/2610.23036v1)，`sandwich_spec()`；稠密路径 `sandwich_spec(dense=True)` | 独立 prelude/coda 包围一个共享 core stack，执行 `passes` 遍；每遍 core 前状态变为 `exp(-dt·a)·z + dt·W e`（`diag_decay`，e 是 prelude 输出，不加 RMSNorm，`W=I`、`a=1`、`dt=1` 起步），`z_0 = e`；固定深度、full BPTT、terminal CE。默认 P2-R3x8-C2 保持论文 P8-R12x8-C8 的 2:3:2 比例 | MoE 路径每个 FFN 是 top-2 专家 bank、无 shared experts，balance 系数 0.001，取发布代码的 input-initialized 路径（论文 MoE 表报告的是 random-state Huginn MoE）；`dense=True` 是 §3.4.2 的稠密路径，无 aux；softmax top-k 替代 sigmoid top-2，论文的 1536 宽 cell、序列长度、词表与优化器设置不复现 |
| [Relaxed Recursive Transformers v3 §2.2/2.3/3.5/附录 C](https://arxiv.org/html/2410.20672v3)，`relaxed_spec()`；稠密路径 `relaxed_spec(dense=True)` | CYCLE 共享：一个共享层 block（含 norm）按 stack 顺序执行 `passes` 遍，无独立 prelude/coda；每个共享层每轮一份 LoRA，`h = W'x + BAx`，无 alpha 缩放，第 l 轮的 LoRA 只在第 l 轮使用；零初始化（`B = 0`、`A` 高斯），训练从普通 Recursive Transformer 出发，共享参数与 LoRA 都训练；固定深度、full BPTT、terminal CE。默认 4 层 × 2 轮、rank 16 | MoE 路径：每个 FFN 是共享的 top-k 专家 bank，router 与路由专家不加 LoRA，只放松 q/k/v/out，balance 系数取仓库默认 0.01（论文没有 MoE）；`dense=True` 是论文路径，q/k/v/out 与 SwiGLU w1/w3/w2 都加 LoRA，无 aux；从随机权重训练而非 uptraining Gemma/TinyLlama/Pythia，因此 Stepwise/Average/Lower 与截断 SVD 初始化不适用（`LoRA.svd_init` 提供公式 4，用于转换全尺寸权重）；attention 与 FFN 各用一个 rank，不复现按组件的 rank 消融；早退与按深度批处理不在此路径 |

前四条、sandwich 和 relaxed 使用固定深度/full BPTT/terminal CE；Huginn、LOOM、Ouro 三条的训练差异不能被这些默认值覆盖。所有路径保留显式router aux归约（dense 的 sandwich 与 relaxed 无 aux）。九个 `*_spec()` 的 `optimizer` 默认都是 AdamW；改用 Muon 只改 `optimizer`，分组规则见[架构](architecture.md#5-训练动态必须看得见)和[训练文档](training.md#修改研究机制)。RMSNorm/RoPE/SwiGLU、独立embedding/head、数据和运行环境是共同背景，应记录完整身份。这些是选定核心路径的公开适配，不是原论文全部工程与效果的复现。

历史四方法工程资格使用300M/1M/1M train/validation/test的预分词 DCLM fixture，继续保留为 smoke；旧副本上游下载 commit 未知，以 parent manifest/chunk SHA 标识。导入保留原 tokens/EOS、不 decode/reencode，只前加本库 BOS=EOS=PAD151645。P1 另用可公开重建的原始 DCLM，已完成24,000,067,249 /10,001,497 /10,009,889 train/validation/test有效 labels。Qwen3-30B-A3B tokenizer pin 为 `ae659febe817e4b3ebd7355f47792725801204c9`、vocab151669；普通 factory 应显式读取 manifest 词表再构造 spec。数据身份、四源文件 tokenizer 和 `--resume` 入口见[训练文档](training.md)。

## P1 普通 MoE / SMELT

实际训练配置已原样发布为 [`p1_moe.json`](../recipes/p1_moe.json) 与 [`p1_smelt.json`](../recipes/p1_smelt.json)。直接加载普通 `RecipeSpec`，保留模型、训练、优化器、完整 schedule 和冻结身份：

```python
import json
from pathlib import Path
from moeloop.recipe_spec import RecipeSpec

spec = RecipeSpec.from_dict(json.loads(Path("recipes/p1_smelt.json").read_text()))
print(spec.model.hidden_dim)  # 440
print(spec.run["schedule_total_steps"])  # 19537
```

| 几何 | 普通 MoE | SMELT |
|---|---:|---:|
| residual width | 672 | 576 |
| 单 expert hidden | 544 | 440 |
| experts / top-k | 64 / 8 | 96 / 8 |
| 物理 prefix/core/suffix | 0 / 10 / 0 | 2 / 5 / 3 |
| core visits / 逻辑 Cell calls | 1 / 10 | 2 / 15 |
| heads / KV heads / head dim | 7 / 1 / 96 | 9 / 1 / 64 |

两臂均为151669词表、独立 embedding/head、4096ctx、固定深度/full BPTT/terminal CE；SMELT 只对 core 两条 residual 各乘1/2。100M-active档指普通基线单次遍历约98.5M active body 参数，排除 embedding/head；其 non-embedding unique约712.7M、模型总量约916.5M，三者不能混称。

F440 来自完整拟训练 prefix 的4096ctx文档 packing：epoch0/seed42/world16、5,001,472个 blocks，每序列平均4,180,216.9229844734个文档内 causal attention pairs。预声明12个 FFN widths 的分析筛选仅按 unique non-embedding 参数、training matrix FLOPs 和逻辑 KV 三预算选形，没有看模型质量。`p1_moe_spec()` 与 `p1_smelt_spec()` 重建这两个训练过的臂（F440，与 P2 同为实测 pairs 口径）；不给 pairs 的默认 dense-S² 分析会选出从未训练过的 F424。

两臂实际完成19,537更新、各20,486,029,312 fresh labels、唯一 seed42；采用16×H200双机 HSDP（replicate2×shard8）、local16/GA1/global256序列。完整 WSD 为20步 warmup、stable至10000、9537步 cosine decay，peak LR3e-4，终点降至.1倍。实际训练源码为 `1bea4d6d2c539e3ae02517aaeede5f9ab68271a7`；发布后的 CLI/文档接线不改变该来源。

完整终点评价、原生曲线和实际成本见 [P1 证据](evidence/p1.json)，分析预算与科学结论的边界见[成本与比较](comparison.md#p1-与文档内-attention-预算)，数据重建及多机训练命令见[训练文档](training.md#p1-实际配方与双机训练)。这是一组公开 DCLM/共同模型栈适配，不声称复现原论文语料或把一个规模的收益外推到所有 looped MoE 方法。

## Foil Base 与 Tying topology

`foil_base_spec()` 和 `foil_spec()` 接收同一组 Base E,D,L。二者都为每个逻辑 core visit 分配独立 attention，所以 flatten 前后 D×L 个 attention owner 不变，hidden width 和单 expert FFN width 也不变。仅独立 attention 不等于完整 Foil。

E=4,D=2,L=2 时，Base 有两个 4-expert bank、每 bank 两次调用；Foil 有一个 8-expert bank、四次调用。外围各用原 E。D 必须为正偶数。Base-tied 的 shared attention 是另一消融，应另命名和记录差异。

Tying `groups=2, group_size=4` 对应连续组 `[0,1,2,3]`、`[4,5,6,7]`：第一份 bank 连续执行四次后进入下一份 bank。native adapter 在 optimizer 构建后识别 canonical core expert 参数，单独应用学习率倍率；attention/router/norm 不缩放，不用含混的参数名正则代替 owner 判断。

## 修改区域

`ModelConfig` schema 4 的 `prefix/core/suffix: RegionConfig` 可覆盖各区域的 `hidden_dim`、`num_experts`、`top_k`、`always_on_hidden_dim`、`residual_scale`。区域共享 residual width 和 attention 几何；复用配置值不会产生模块/参数别名。旧schema1/2/3按原默认行为读取，新状态/gate/注入默认关闭；配置可读不等于跨源码strict continuation。

```python
from dataclasses import replace
from moeloop import LoopedLM
from moeloop.model import RegionConfig
from moeloop.recipes import meta_spec

spec = meta_spec(vocab_size=128, dim=32, hidden_dim=64,
                 num_heads=4, num_kv_heads=2, num_experts=4,
                 max_seq_len=64)
variant = replace(spec, name="meta_core8",
                  model=replace(spec.model, core=RegionConfig(num_experts=8)))
model = LoopedLM(variant.model)
print(model.config.region_config("core").num_experts)  # 8
```

研究变体应更新来源/适配、比较契约与运行证据。改变参数 owner、梯度或目标后，旧运行不能自动证明新路径通过。

## 三种新增训练语义

`huginn_spec()` 默认使用shifted Poisson-lognormal：`r=1+Poisson(exp(tau))`，`tau`的参数使平均深度为4，log std为0.5，反传尾段k=2。深度按seed、成功update与microbatch定位、跨ranks同步，无硬cap。训练latent为独立truncated normal，std=`sqrt(2/5)`、边界±3std；评测显式改用zero-init和`eval_passes=4`。这两个初态分布不能混为同一实验条件。

`loom_spec(core_layers=2, passes=3)` 使用 `gamma=lambda/(H*sqrt(M))`，后续轮以 `g_t=lambda/(t*sqrt(M))` 重注入embedding，默认lambda=beta=0.5。global EMA每forward清零、local EMA每轮清零；scaled attention写入后取两个normalized read直接相加，再接output-normalized MoE残差。状态临时且可微，不是跨样本buffer。

`ouro_spec(passes=3, entropy_coeff=.1)` 的共享gate产生逐token条件停止概率。前R−1轮按hazard分配质量，末轮接收剩余质量；目标为 `sum_r p_r*CE_r - beta*H(p)`，加独立aux。训练执行全部轮，保留概率、head与backbone梯度。原生expected CE按chunk重算head以限制词表中间量；HF的`labels`与标准heldout仍为terminal CE，不是此训练目标。

下面只演示可读配置和固定深度参考前向，不启动训练或声称方法资格：

```python
import json
import torch
from moeloop import LoopedLM
from moeloop.recipe_spec import RecipeSpec
from moeloop.recipes import huginn_spec, loom_spec, ouro_spec

geometry = dict(vocab_size=23, dim=16, hidden_dim=32, num_heads=4,
                num_kv_heads=2, num_experts=4, top_k=2, max_seq_len=32)
for make in (huginn_spec, loom_spec, ouro_spec):
    spec = RecipeSpec.from_dict(json.loads(json.dumps(make(**geometry).to_dict())))
    model = LoopedLM(spec.model).eval()
    with torch.no_grad():
        output = model(torch.tensor([[1, 2, 3]]), collect_readouts=False)
    print(spec.name, spec.training.objective, tuple(output.logits.shape))
```

## 组合与组件边界

attention×router四格由四份普通配置表达：固定expert bank、调用图、norm、目标和训练规则，只改变两个owner开关。它检验组件组合，不宣称总参匹配或完整Foil×LOOM：

```python
from dataclasses import replace
from moeloop.recipes import recursive_moe_spec

base = recursive_moe_spec(**geometry)
grid = {
    f"a{a}_r{r}": replace(base, name=f"a{a}_r{r}",
        model=replace(base.model, attention_per_visit=bool(a), router_per_visit=bool(r)))
    for a in (0, 1) for r in (0, 1)
}
```

`loom_spec(attention_per_visit=True)` 是另一条完整状态×结构组合：保留两EMA、重注入、缩放、逐轮router和full BPTT，只增加逐轮attention owner。其方法资格单独记录，不能替代LOOM基础路径。

| 组件入口 | 含义 |
|---|---|
| `dense_spec()` / `moe_spec()` | 普通非循环基线，资源对齐另算 |
| `recursive_moe_spec()` | 共享的 stack 循环体 |
| `per_visit_router_spec()` | 每 visit 独立 router，单独使用不含LOOM memory/状态 |
| `input_anchor_spec()` | live anchor、update-level深度采样和尾段梯度；单独使用不含Huginn随机初态/Poisson-lognormal过程 |
| `ModelConfig(input_injection="diag_decay" 或 "orthogonal")` | xllm-loop Part II 的两种注入算子（schema 4）；`diag_decay` 由 `sandwich_spec` 使用，`orthogonal` 目前没有命名 recipe；初始化见[架构](architecture.md#4-核心模型保持普通-pytorch) |
| `ModelConfig(lora_attention_rank=r, lora_ffn_rank=r)` | 逐轮 LoRA（新字段，默认 0，缺这两个字段的旧记录按 0 读取）：只作用于 core，路由专家不加；与 EMA 状态互斥，要求固定深度与 full BPTT；`relaxed_spec` 使用 |
| `tiny_spec()` | 安装和工程开发的小配置 |

从此版本开始，`loom()` 构造完整memory方法；旧router组件应使用 `per_visit_router()`。`injected()` 仍是 `input_anchor()` 的deprecated别名。旧checkpoint应读取其实际配置/来源，不能凭旧factory名字替换为新方法后声称exact resume。

Huginn尾段梯度与全轮expected CE等缺少本阶段梯度定义的组合明确拒绝；Ouro Stage II、真实token早退及其KV策略不在此选定路径。新状态更新继续使用短的明确forward，复用数据/训练/HF接缝。继续阅读[架构](architecture.md)、[成本与比较](comparison.md)和[训练与恢复](training.md)。
