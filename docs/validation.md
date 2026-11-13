# 验证记录与支持边界

2026-10-25。V1/P1与V2按原始来源继承；V3的P2两臂、C四格、完整四模型downstream、真实installed-source旅程和完整HF端点均已执行。**最终assembler与独立`check_release.py --stage v3`均实际exit0，报告内容核对一致，scope=`v3_full_acceptance`。** 原始数据、配置、曲线、进程与产物身份以[公开V3投影](evidence/recipes-v3.json)、[可重画分析](evidence/v3-analysis.json)及[#8](https://github.com/datamllab/LoopTitan/issues/8)的实际记录为准。以下将工程资格、科学结果和硬件性能分别声明。

## V3：第二尺度、组合与交付

### 配对、数据与执行来源

P1/P2使用同一公开DCLM前缀、tokenizer、文档隔离与顺序、4096ctx、local16/global256/GA1、seed42、BF16计算/FP32 master与reduce、相同terminal CE/aux规则与完整WSD。每臂 **19,537更新 /20,486,029,312新鲜有效labels**，无padding；停止/恢复不改变原schedule horizon。不同shape不声称初始权重逐元素一致。

| P2配置 /主分析预算 | 普通MoE | SMELT |
|---|---:|---:|
| width /expert FFN width | 1280 /432 | 1056 /384 |
| 物理层 /逻辑调用 | 12 /12 | 12（3+6+3）/18 |
| experts /top-k | 64 /8 | 88 /8 |
| query heads /KV heads /head dim | 10 /1 /128 | 12 /1 /88 |
| unique non-embedding参数 | 1,318,288,640 | 1,314,771,744（−0.2668%）|
| 总参数，含独立embedding/head | 1,706,561,280 | 1,635,096,672 |
| document-aware training FLOPs /padded position | 2,573,863,361.534 | 2,515,805,029.899（−2.2557%）|
| 4096ctx、batch1/BF16逻辑KV bytes | 25,165,824 | 25,952,256（+3.1250%）|

baseline的physical active non-embedding为203.52M；候选为146.92M，按逻辑调用累计为220.38M，不能声称两臂active相等。E88是公共栈适配；最多32个纯成本候选在质量结果出现前按三预算残差选形。工程各5%上限不消除残差对解释的限制。

P2两臂实际各用两节点 **16 H200 /HSDP replicate2×shard8**，两组16卡并发；不是32-rank模型训练。两条主线都有真实durable full DCP恢复，保存model、optimizer、scheduler、rank RNG、cursor/progress；完整曲线及独立HF按实际来源保留。32卡collective连通检查不作为32-rank模型训练或效率证据，EP/TP/PP与弹性拓扑恢复未验。

| 记录 | 冻结源码 |
|---|---|
| P1科学训练 | `1bea4d6d2c539e3ae02517aaeede5f9ab68271a7` |
| V2新方法/组合资格 | `78d799dfa11c6be4130be63b252de9b91ae5aa8b` |
| P2与C原生训练 | `17d51e4df9e807f81c5d31301441daba9ce527b2` |
| 真实source→package→installed旅程 | `97f4348678ae0213965c60d25a280777c878a51c` |
| 最终只读assembler/checker | `d94c268dd5d74c668f079e1b8d468c7a7989822f` |

最终gate与文档有各自的后继来源，不冒充上述训练commit。旧P1 schema2 HF没有`labels_objective`，按已知无exit gate的terminal语义只读消费；原config/权重/provenance未迁移。测试使用的序列化config快照与原始JSON字典相同，字节SHA分别标识各自文件，不混用。

### 原生曲线、完整HF与实际成本

| P2固定终点 | 普通MoE | SMELT | 候选−普通 |
|---|---:|---:|---:|
| native固定验证切片CE，1,048,576 labels | 2.82716393 | 2.81225824 | −0.01490569 |
| 完整4096ctx HF validation CE，10,001,497 labels /8167docs | 2.89231853 | 2.87802890 | −0.01428963 |
| 完整4096ctx HF test CE，10,009,889 labels /8212docs | 2.88725790 | 2.87307719 | −0.01418072 |
| 全保留native进程GPUHr | 135.01837 | 148.03375 | +9.6397% |

完整HF为FP32、terminal深度1/2、无`max_batches`、`complete_split=true`，分母和逐文档身份对应冻结heldout；native曲线共46点，每点约1M labels，不能替代完整10M端点。P1完整test为3.00927801/3.00338042（Δ−0.00589759）。这两个同D配对观察到较低候选CE，P2差值更大；不据两尺度、单seed推断scaling law、普适Loop优势或严格原论文复现。

主compute坐标是 **document-aware analytical model compute**：`fixed_terminal_document_matrix_v1`、4096ctx、冻结attention pairs均值4,180,216.9229844734，乘各点实际padded_positions。dense-S²另作secondary估计；逻辑计算不是GPU实测FLOPs。共同compute仅在覆盖区间做局部插值。P1的6.0311%等CE解析节省来自原生约1M验证切片，不是完整HF或墙时节省。

GPUHr统计原始train目录全部保留native进程wall×本地GPU数，含初始化、validation/checkpoint和失败；P2各12份收据，main-run子集133.23214/146.16926不替代135.01837/148.03375。foreground reservation与它重叠，不相加；独立HF/downstream及环境闲置另列，不解释为GPU busy time。P1实际78.02286/103.43271，候选高32.5672%；P2候选高9.6397%，没有硬件加速结论。

### 连续性能窗口

下表为同硬件/软件、4096ctx/global256、无Cell AC的真实P2 update窗口：warmup steps5–9，连续测量10–29。有效labels/s为全局值，峰值为global rank0观测的本地allocator值，不推定全rank最大值，完整进程成本另计。

| P2 | median step（min–max）s | median有效labels/s | peak allocated /reserved GiB |
|---|---:|---:|---:|
| MoE | 1.57763（1.52090–1.74038）| 664,652 | 102.46 /111.93 |
| SMELT | 1.69898（1.66362–2.06691）| 617,182 | 123.71 /133.68 |

部分owner代表窗口为C的B+A（a1_r0）：4H200、2048ctx/global16、无AC，steps1–5 warmup、6–25测量；median step **0.127723s**（0.123826–0.334440），median有效labels/s **256,555**、aggregate **237,596**，rank0 peak allocated/reserved **4.997071/6.181641GiB**。实际窗口与source/metrics身份见[V3分析投影](evidence/v3-analysis.json)；不以全共享C-B代替。上述是已测几何，不外推EP/TP、32卡模型效率或任意更大模型；更大预训练需先核对activation/head显存、数据容量、保存恢复和吞吐，再决定AC/并行取舍。

### C：固定expert pool的owner四格

共同d512/F1024/E8/top2、2 banks×4 visits、heads8/kv2、2048ctx/global32768 positions、LR3e-4、warmup64/cosine至0.1peak；每臂seed42、2048更新、67,108,864有效labels。这里的CE是共同原生终点口径，不是完整10M HF评价。

| 四格 | 末点CE | unique non-embedding | 相对B的core增量 | native GPUHr |
|---|---:|---:|---:|---:|
| B | 5.16462517 | 26,487,296 | 0 | 0.44051 |
| B+R | 5.15252876 | 26,511,872 | +24,576（0.0928%）| 0.43174 |
| B+A | 5.10578394 | 30,419,456 | +3,932,160（14.8457%）| 0.45228 |
| B+A+R | 5.10317230 | 30,444,032 | +3,956,736（14.9385%）| 0.45974 |

百分比以B的core non-embedding 26,486,784为分母；embedding/head共155,309,056，不用于稀释增量。`I_AR=CE_AR−CE_A−CE_R+CE_B=+0.00948477`，combined仅比A再低0.00261164。四格非iso-parameter，不是完整Foil×LOOM；只描述此短面板，不宣布最终recipe。C的dense matrix成本单列，不与P1/P2 document-aware坐标混用。

### 四模型完整downstream与安装

`lm_eval[hf]==0.4.13`，0-shot、FP32、terminal深度MoE1/SMELT2，完整ARC-Easy test2376 /HellaSwag validation10042。`auto:4`实际选到batch64；四模型原生`task_hashes`和harness源码身份相等，任务YAML/helper、数据revision与sample logs均保留（sample logs未公开），没有`--limit`替代：

| 模型 | ARC acc /acc_norm (%) | HellaSwag acc /acc_norm (%) |
|---|---:|---:|
| P1 MoE | 54.76 /50.08 | 33.52 /39.63 |
| P1 SMELT | 56.23 /49.96 | 34.03 /40.43 |
| P2 MoE | 58.63 /52.15 | 36.39 /45.32 |
| P2 SMELT | 58.38 /52.53 | 37.11 /46.39 |

ARC在P1 raw acc改善、norm略降，P2相反；HellaSwag两尺度两指标改善。保留全部指标，单seed/文档stderr不提供训练seed方差。实际评分前台预留合计0.238624 GPUHr，只读合并0.007671另计。

真实source97→sdist→由sdist重建wheel、checkout外无Torch base import后，完成公开WikiText2小子集prepare、实际train4→full-state resume8、full DCP→HF、完整2057-label demo validation及独立离线确定性生成。训练4096有效labels；这是实际用户旅程，不是科学P1/P2/C。原uv路径缺失、native父环境没有TT metadata、HF-only依赖版本误判三次失败保留；最后用正常构建的pin086 TT0.3.0和真实5.20.2依赖环境走通，未伪造dist-info或构造替代权重。当前安装按公开pin5.20.2，历史native/HF-only记录的5.19.0保留原身份，不笼统宣称相同环境。

最终验收工具来源为`d94c268dd5d74c668f079e1b8d468c7a7989822f`，实际assembler与独立CLI均exit0；公开分析绑定原始analysis SHA `4283eaf816c93ea87a4c65a825c0013d46d476f8b2d77692d561636e43699a7f`。公共JSON/CSV/图是保留数值与身份的投影，重新核验仍需原始数据/DCP/HF或自行重建的完整运行，不以投影替代raw证据。旧严格全网络BF16梯度失败仍见下文[公开实现性能与数学资格](#公开实现性能与数学资格)，阶段通过不等于全网络parity。

## 新增训练动态与组合资格

2026-10-24。**实际CLI完整通过，scope=`v2_full_acceptance`。** [机器可读证据](evidence/recipes-v2.json)和其中四条CSV保留执行源码、recipe/运行时、实际进度、native验证、独立HF、完整恢复和实际成本。冻结训练/验收源码为`78d799dfa11c6be4130be63b252de9b91ae5aa8b`；后续发布提交不能冒充训练提交。#6的完整P1按其原始source/验收文件SHA继承，没有重训。

四条路径均使用真实公开DCLM、Qwen3词表151669、dim128/expert hidden256、E4/top2，H200两卡sharded FSDP（replicate1/shard2）、context2048、global16/local8/GA1、BF16计算/FP32 master与reduce、无Cell AC。每条**512次成功更新、16,777,216有效labels**，唯一训练seed42。

| 路径 | 实际不可约语义 | native末点CE | HF小样本CE | 恢复模型/optimizer最大误差 |
|---|---|---:|---:|---|
| huginn | 随机latent；512个microbatch深度1–19，tail2且live anchor | 7.093894 | 6.550015 | 0 / 0 |
| loom | M2/H3双EMA、重注入、独立router、full BPTT | 6.772820 | 6.237102 | 0 / 0 |
| ouro | R3逐token expected CE−entropy；共享gate/head真实更新 | 6.853518 | 6.302900 | 0 / 0 |
| loom_attention | 完整LOOM状态与每轮独立attention，M2/H3 | 6.761012 | 6.233232 | 0 / 0 |

这些是不同目标/结构的**工程学习与交付轨迹，不是公平质量排行榜或原论文效果复现**。native验证为固定2048ctx、每次65536有效labels，共9点；HF为256ctx、1024有效labels、`complete_split=false`，两组数值不作为native/HF loss parity。

- 四条路径都从真实full DCP导出，独立`python -I`离线消费者加载。Huginn/LOOM/hybrid使用无cache固定深度生成；Huginn的zero-init评测是公开适配。Ouro固定terminal生成的cache/重算通过，HF labels仅计算标准terminal CE。
- 各自连续8步与4+恢复4步，模型/optimizer最大误差为0；两rank RNG、数据cursor、已完成计数及scheduler等离散状态一致。Ouro另读取真实DCP Adam moments确认gate/head/backbone均更新；分布熵与每轮概率/CE真实记录。
- attention×router四格各8次真实更新，共用expert pool/调用图/归一化规则，只改owner开关；配置和成本可重建。这条8步记录是运行资格；#8正式C效果见上文。
- 正式四条512步主训练原生进程合计0.44319 GPUHr；加其恢复分叉与四格为**1.38247 GPUHr（20个native receipts，无失败）**。口径是原生进程wall×实际GPU数，包含初始化/验证/存盘，不含HF后处理、早期smoke和环境闲置，不等于GPU busy time。

新方程有独立FP32/闭式梯度oracle，实际两进程CPU FSDP覆盖expected head、EMA+FullAC及随机latent尾段反传；GPU证据覆盖上述两rank执行配置，不外推新增方法已通过多机/任意并行矩阵。公共比较器对随机深度训练总FLOPs保留unknown；expected CE单列head checkpoint重算，不能宣称hazard带来forward早退。

### 参数调整的语义回归

独立复审后补充以下CPU验证，比较完整logits、CE/aux及所有参数梯度；它们与上面的GPU训练证据分别记录：

| 路径 | 已覆盖的非默认条件 | 独立参照 |
|---|---|---|
| LOOM | M1/H1、M3/H4；beta为0/.25/.8、不同lambda；独立attention、GQA/QK norm、always-on FFN、异构外围及padding | 闭式加权历史，覆盖双memory、重注入和reset；长历史组合另用FP64排除求和顺序舍入，其余FP32 |
| Huginn-inspired | M3、实际r=1/3/5、tail=1/2/full；非默认latent分布、injection scale和异构外围 | 显式展开后切断计算图，保留live anchor，独立于生产的no-grad执行段 |
| 实际两rank FSDP | Ouro expected CE×FullAC；LOOM/Ouro M3/H4、独立attention、E1<shard2的另一expert分片轴 | 全参数梯度对照，两个microbatch、两个update，覆盖重复调用/重算 |
| HF词表 | 扩容、对齐到倍数、收缩；普通递归/LOOM/Ouro；原权重保持与新行初始化 | save/load后配置与logits一致，另有`python -I`离线AutoClass消费者 |

复审发现并修复HF标准`resize_token_embeddings()`只更新外层词表而遗漏core/嵌套配置的问题，以及`mean_resizing=False`新增行未使用`init_std`的问题。修复只触及HF消费接口；旧训练未做词表resize，原训练方程、配置、DCP及当时导出的HF保持原始身份，不为此重复GPU训练。新增回归能拒绝双memory取平均、anchor被detach、beta被写死为默认值等实际临时变异。

这些检查支持在既有机制内调整参数，不代表所有开关笛卡尔积都已资格。改变状态/梯度/目标需保留或增加对应独立oracle；改变目标训练规模、精度或拓扑时，先做目标几何的短GPU运行，再铺长程比较。

## P1：实际配置、数据与完整训练

直接重放本次配置应读取 [MoE JSON](../recipes/p1_moe.json) 和 [SMELT JSON](../recipes/p1_smelt.json)，而非重新从通用 factory 的默认值生成。训练源码冻结为 `1bea4d6d2c539e3ae02517aaeede5f9ab68271a7`；后续文档和 CLI 发布提交不能改写这次训练来源。TorchTitan 固定 `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`，实际环境为 Python3.12.15、Torch2.14.0+cu130、CUDA13.0、Transformers5.19.0。

| P1 配置或分析预算 | 普通 MoE | SMELT |
|---|---:|---:|
| width / expert FFN width | 672 / 544 | 576 / 440 |
| 物理层 / 逻辑 Cell 调用 | 10 / 10 | 10（2+5+3）/ 15 |
| experts / top-k | 64 / 8 | 96 / 8 |
| attention heads / KV heads / head dim | 7 / 1 / 96 | 9 / 1 / 64 |
| 唯一 non-embedding 参数 | 712,656,672 | 737,845,056（+3.5344%）|
| 总参数，含独立 embedding/head | 916,499,808 | 912,567,744 |
| training FLOPs / padded position | 1,284,757,348.671 | 1,248,742,044.863（−2.8033%）|
| 逻辑 KV bytes / 4096ctx，batch1/BF16 | 15,728,640 | 15,728,640 |

“100M-active 档”指普通 MoE 的约98.5M active non-embedding body，并非总参数量。SMELT 中间5层整 Cell 共享并执行两遍，core 两条 residual 各乘1/2，外围不缩放。三预算均在事前5%配对范围内；FFN width 从12个事前候选按预算残差选取，未按质量选形。这些是 `fixed_terminal_document_matrix_v1` 解析预算，实际 kernel 时间、显存与 GPU-hours 另报，不能用分析 compute 代替实测资源。

[公开源清单](evidence/p1-dclm-source.json)固定 DCLM revision `817d6752765f6a41261085171dd546b104f60626` 的1024份有序 metadata；准备达到预算时实际读取前198份，并非下载或训练完整数据集。[实际 manifest](evidence/p1-data-manifest.json)标识 train **24,000,067,249**、validation **10,001,497**、test **10,009,889** 个有效 labels，fingerprint 为 `e67a956d2fe51dd1784505a6f58d7790f397a09624330700c0ebaffcd344f152`。数据按原文档身份与精确内容全局去重、整篇分配 split；Qwen3 tokenizer 固定 `ae659febe817e4b3ebd7355f47792725801204c9`，词表151669。源 tokenizer 与规范化准备资产的 hash 分开保留。

[完整 prefix 统计](evidence/p1-data-statistics.json)覆盖实际拟训练的5,001,472个 blocks：20,486,029,312 labels、无 padding，各 rank 新鲜数据容量充足。4096ctx、world16 的 document-causal query/key pairs 均值为 **4,180,216.9229844734 / sequence**，包含对角线。该实测统计参与预算配对，不使用 dense-S² 或早期20M-label小样本代替。

正式训练在两节点各8张 H200 上执行 **HSDP replicate2 × shard8**，local batch16、GA1、global256 sequences/update、4096ctx；TP/EP/CP/PP=1。计算 BF16、master/reduce FP32，native grouped experts/FlexAttention、head chunks8，无 AC、无 compile。两臂同一 AdamW/LR3e-4、clip1，完整 WSD 为20步 warmup、至10000步 stable、9537步 cosine decay至0.1倍。有限 train epoch0、shuffle、`num_workers=0`；实际 stop/resume 不重置 horizon、数据或 optimizer，不追加训练 seed。

每臂真实经过8、95、954、4769、10000、19537的共同停止点，保留实际恢复 lineage。原生有限 validation 共46点，每点 **1,048,576 labels**；成功终止时补非周期末点，已在周期验证的停止步不重复。其末点 CE 为 MoE **2.9485197**、SMELT **2.9438329**；这是固定约1M切片，与下面独立 HF 的完整约10M validation/test 分开。same-D 与共同解析 compute 的曲线分别报告，共同 compute 只在实际覆盖区间内插值。

### P1 结果与解释

[公开 P1 证据](evidence/p1.json)汇集冻结配置、[完整曲线](evidence/p1-native-curves.svg)、[末段放大](evidence/p1-native-late-curves.svg)、评价与资源口径。完整 HF 终点评价使用4096ctx、无截断 split，两臂逐文档身份和有效 label 数一致：

| 完整 HF 评价 | labels / 文档 | 普通 MoE CE | SMELT CE | ΔCE（SMELT−MoE）|
|---|---:|---:|---:|---:|
| validation | 10,001,497 / 8,167 | 3.01446893 | 3.00836268 | −0.00610625 |
| test | 10,009,889 / 8,212 | 3.00927801 | 3.00338042 | −0.00589759 |

从6,002,049,024个训练 labels 起，31个连续同数据 native 验证点的 SMELT CE 更低。共同解析 compute 区间末端为 **2.5581766134191145×10¹⁹ FLOPs**；该处普通 MoE 在相邻实际测点间插值得到 CE **2.95114454**，SMELT为 **2.94383287**，差 **−0.00731166**。只在两臂共同观测区间内比较达到相同 native CE 所需计算，解析 compute 节省为 **6.0311%**；不使用越出共同区间的各自终点估计作为主结果，不把约1M native CE 与完整 HF CE 混合。

本次结果分类为 **`clear_positive`，仅限这组公开 DCLM、冻结模型配对与 seed42**：完整 val/test 与末段曲线方向一致。它不移除 SMELT 唯一 non-embedding 参数多3.5344%的预算残差，不估计训练 seed 方差，也不证明所有 MoE、规模或数据上的因果收益。每文档配对不是训练随机性的显著性检验。

原生训练实际资源为普通 MoE **78.02286 GPU-hours**、SMELT **103.43271 GPU-hours**，合计 **181.45557 GPU-hours**，按真实进程墙钟×本地卡数统计，含初始化、原生验证与存盘，不含独立 HF 后处理或环境空闲。SMELT实际训练成本高 **32.5672%**，本轮**没有 GPU-time 收益**；上述6.0311%是解析 compute 下的质量比较，不是加速比。

本次是 SMELT 核心共享结构与公平比较思想在公开 DCLM/共同模型栈上的适配，不声称复现原文语料、完整工程或绝对 loss/缩放前沿节省。

## 分布训练、完整状态恢复与模型交付

选定 P1 源码/数据/HSDP16配置的真实连续8步与4+4恢复，核对 **103个模型张量、309个 optimizer 张量（1545个 state keys），最大误差均为0**；16个 training ranks 与16个 data ranks 的 RNG、progress、cursor/source、scheduler/scalar/hyperparameters 精确一致。两条20.486B主线也实际从各自保护的 full DCP 恢复到终点，不只验证 model-only warm start。

历史目标形状的8-rank FSDP恢复（源码 `33f246b`）及两节点16-rank全 FSDP恢复（`bae23d7`）也比较了相同103/309个张量，模型、optimizer和各 rank continuation 状态误差为0。它们使用各自数据、batch和源码，作为已验证执行路径复用；不把不同 local batch 的吞吐称为严格 strong scaling。更早的四方法资格范围见下文。

最终交付要求从真实19,537步 DCP 导出 HF，并在 `python -I`、离线、未导入训练库的独立消费者中核对固定深度 cache/greedy。完整 validation/test 评价使用4096ctx、完整 split，并保存每文档 loss sum/count，以相同文档身份和有效 labels 配对；约1M原生验证、历史256ctx小样本和完整约10M HF 评价各报自身分母。两臂均完成最终 HF 导出、独立离线 cache/greedy 以及完整 validation/test，命令均返回0；完整 heldout 的 label/document counts 与逐文档 sidecar 已核对。

最终阶段 checker 使用 `ad1a21f` 对 JSON tuple/list round-trip 身份边界的修正，训练源码仍冻结于 `1bea4d6`；修正不改变训练、模型数学或原数值容差，相关真实 CPU DCP 回归为39 passed。 发布前还修复了非日志步复用上一条梯度范数/吞吐/显存观测的记录问题：未采样字段按步省略，CE/aux/progress保持逐步记录；P1实际使用`log_freq=1`，冻结证据不受影响，来源仍为`1bea4d6`。 通用 trainer adapter 还统一消费声明的AC与零WD角色，并拒绝未支持的共同栈设置；既有P1配置为无AC、四项零WD角色及FP32 master/reduce，原冻结产物保持原样。

数据重建、两节点原生训练、严格恢复、DCP→HF与完整评价均使用[公开训练入口](training.md)，不依赖特定调度器或额外 helper。公开 metadata 标识实际产物，不包含96GB token payload或训练 checkpoint；自行重建时应核对 manifest/content hash，并生成自己的实际运行记录。

这些证据覆盖已记录模型档、执行设置和拓扑。更大模型/更多节点应先用同一源与数据语义做短 smoke、保存/恢复和吞吐检查，再进入完整训练；不能由16卡通过推断任意未来规模无需适配。EP/TP/CP/PP、弹性拓扑恢复和任意 compile 组合仍不属于已验证范围。

## 公开实现性能与数学资格

[公开参考脚本](../scripts/recipe_benchmark.py)在单张 H200 上，以普通 MoE 目标形状、BF16、batch2/context4096、head chunks8、无 AC，核对相同参数/数学配置和自然路由；5步 warmup 后测量20次包含 chunked head/CE、反传和相同 AdamW 的完整 step。本库中位数 **147.04ms**、公开 TorchTitan Qwen3 路径 **141.43ms**，比率 **1.0397**，未触发预设 `>1.20` 调查线。初始化、数据 I/O、checkpoint/eval 与分布式通信不在该测量内。

该记录的原始完整 BF16 深层梯度检查仍是 **`correctness.status="failed"`**，误差与3%门槛原样保留，并记录自然 top-k 分岔；显式 `--component-math` 的同输入 Flex/SDPA、FP32 router、官方 HF experts 及其全部组件导数检查通过，另有官方 HF FP32 小模型完整更新 oracle。资格范围是 `component_equivalence_with_observed_end_to_end_bf16_route_divergence`，不宣称完整网络梯度 parity。性能测量未固定路由。

[8卡性能摘要](evidence/performance-fsdp8.json)来自单节点8张 H200 的真实 FSDP 对照，公开[213行测量脚本](../scripts/recipe_benchmark_fsdp8.py)与执行文件逐字节相同。两臂使用普通 P1 几何、相同初始权重、local batch16/context4096、accumulation2/global batch256、head chunks8、无 AC、FP32 master/BF16 FSDP/FP32 reduce，以及相同 fused AdamW 和 clip1。1次安全更新与3次 warmup 后各测10次完整更新；每步取8个 ranks 的最大同步墙钟时间，barrier 在计时外。

| 单节点8卡 FSDP | step 中位数 | 全局有效 labels/s | 峰值 allocated GiB |
|---|---:|---:|---:|
| 本库 | 1.650406s | 635,344 | 77.61 |
| 固定 pin 的官方 TorchTitan Qwen3 | 1.618800s | 647,749 | 76.81 |

step 比率 **1.01952**，未触发原 `>1.20` 调查线。计时使用 GPU 常驻合成 token、每行一个无 padding 的因果文档和自然路由；数据 I/O、mask 构造、编译/warmup、checkpoint/eval 不计时。首个非计时完整更新中，全部参数的梯度与更新值有限，CE 为12.06318665/12.06318092；10个计时更新的 loss/grad norm 有限。

两侧保留各自原生 FSDP 分组/reshard、Flex autotune 和 router 实现。实际 hook 记录两侧 router 输入 FP32、materialized gate 权重 BF16，官方 scores 为 FP32；输出 dtype 本身不能证明逐算子精度一致。数学依据仍是上述单卡 components 与 HF FP32 oracle，原 strict BF16 失败保留；本次增加的是分布计时及运行安全检查。该比率只覆盖单节点8卡，不是双机官方对照、完整分布 BF16 梯度等价或完整 P1 科学验收。

在摘要所列软件版本及固定 TorchTitan checkout 的训练环境中，可按下列命令复跑。先从公开摘要提取本次两份模型配置。该性能测量采用普通 P1 几何，但其合成数据、8卡 FSDP/GA2 与正式 P1 的真实数据、16卡 HSDP/GA1 分别记录。

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

## 历史工程资格：四方法 DCLM 短训

[历史机器可读证据](evidence/recipes-v1.json)保存当时的 request/spec、源码和产物 hash、进度、heldout、恢复与独立 HF 结果；当时 `check_release.py --stage v1` CLI 返回0、verifier为 passed。该批训练冻结在 `1aad50570c58c57ddb6f8770d30f78637bc75865`，其中20个源码/训练文件的 hash 仅标识那个快照。当前模型、数据准备和训练代码已有变化，不能再声明与其逐字节相同；旧 JSON 原样留存为历史工程证据，具体复用范围按所触及路径另行判断。最终 CI、独立 review 和当前 P1 验收由现行 PR/证据记录。 后续源码 `79bbc5f` 另在2-rank上实际完成Foil的8次更新、Full AC、DCP→独立HF/cache/greedy，补充较新共享算子的执行证据；它不追认为四方法各自都在当前源码重跑512步。

四方法均在 **2张 H200、2 ranks sharded FSDP** 上完成 **512 successful updates、context2048、global batch16（32768 positions/update）**。每法实际累计 packed positions、有效 inputs、有效 labels 均为 **16,777,216**。`dp_replicate=1/dp_shard=2`，TP/EP/CP/PP=1；Torch 2.14.0+cu130、CUDA13.0、Python3.12.15，TorchTitan pin为 `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`。计算参数为BF16，master parameters/gradient reduce为FP32；Foil 使用 Full AC，其他三法无 AC。

### 支持矩阵与评价口径

每法 native validation 在 step1、64、128、192、256、320、384、448、512 共 **9点**，context2048、每点全局8192有效 labels。独立离线 HF heldout 是 **context256、1024 labels 的固定 validation 小样本**，`complete_split=false`。两组 CE 的上下文和覆盖范围不同，不作为 native/HF loss parity，也不据此给方法做科学排名。

| 方法 | core passes | AC | native heldout CE：step1 → 512 | HF CE：256ctx/1024labels | 本轮额外8 vs 4+4恢复分叉 |
|---|---:|---|---|---:|---|
| SMELT | 2 | 无 | 11.927265 → 7.219949 | 6.691771 | 支持共同恢复入口；未单独做新增分叉 |
| Meta | 4 | 无 | 11.952620 → 7.269093 | 6.750134 | 支持共同恢复入口；未单独做新增分叉 |
| Foil | 2 | Full | 11.934988 → 7.192571 | 6.657285 | 通过 |
| Tying | 4 | 无 | 11.928846 → 7.300015 | 6.786260 | 通过 |

四法均从各自真实 durable full DCP 导出，在 **`python -I`、离线、未导入训练仓库包**的独立进程加载 HF；固定深度 heldout、cache 与 cached/uncached greedy 均通过。每法核对3个增量 cache 位置和4个 greedy 新 tokens，cache logits 最大绝对误差≤`3.338e-6`，检查门槛 `atol=1e-5/rtol=5e-5`；greedy token IDs 相同。这是已运行配置与有限输入的资格，不承诺任意规模、上下文和 cache 组合。

实际新增分叉是 Foil/Tying 的连续8步与4步+恢复4步，保持同一 schedule horizon、源码、数据与拓扑：Foil 比较51个模型张量/153个 optimizer 张量，Tying 比较65个模型张量/195个 optimizer 张量，模型和 optimizer 最大绝对差均为0。optimizer scalar/hyperparameters、两个 training ranks 的 RNG/progress、两个 data ranks 的 cursor/source 状态和 scheduler 离散状态精确一致；每分叉最终262,144 positions/labels。Tying 实际 expert LR multiplier为0.5。共同恢复接口与这两条新增分叉证据分别声明，不写成四法各自都做过分叉。

### 预算配对与数据范围

普通 MoE/SMELT 另完成 **每臂8步真实 pilot**，每臂262,144 positions/labels、context2048/global batch16、同一 DCLM fixture。固定五候选中选中 12物理层×8-expert MoE 对 8物理层×12-expert SMELT，均12次逻辑 Cell 调用；vocab151669、dim128、expert hidden256、top-k2、heads4/2。以下为 `fixed_terminal_matrix_v1`、无 AC、batch1/BF16 KV 的三个分析预算：

| 预算 | 普通 MoE | SMELT | 相对差异 |
|---|---:|---:|---:|
| 唯一 non-embedding 参数 | 10,042,496 | 9,844,864 | -1.968% |
| 矩阵 training FLOPs/position | 171,998,976 | 172,035,840 | +0.0214% |
| 2048ctx逻辑 KV bytes | 6,291,456 | 6,291,456 | 0 |

三项在预声明5%内，真实 pilot 已运行；这不等于 wall-time/峰值显存匹配，也不外推为四条512步 recipe 互相资源匹配或谁质量更好。[比较定义](comparison.md)保留 unknown 和估计范围。

数据 manifest fingerprint 为 `1dd1c249342678f5d758bd30aa7e5944e151b12f76844dcfb45505202e7d446b`，来源、Qwen3 tokenizer pin、300M/1M/1M 实际 split 计数和隔离见[DCLM证据](evidence/dclm-v1.json)。原 token/EOS 保留，前加本库 BOS；上游下载 commit 未知，精确预分词 parent chunks 尚未公开发布。公开报告给出内容身份，实际数据/checkpoint 产物另行保存；这些限制不因短训完成而消失。

该历史批次的 CPU 模型回归为 **137 passed**；Torch-free 控制 suite 为 **286 passed、19 skipped**，不表示当前 head 已重跑相同结果。它们与 GPU 资格分开，skips 不计为方法检查通过。四方法 loss 轨迹、短生成和8步 pair pilot 是工程学习/交付证据，不是论文效果复现、长期稳定性或 scaling 结论。

## 0.1 工程基线历史（2026-10-21）

以下保留原0.1记录及原始失败事实；其中2–4步工程 fixture、旧CPU计数和多机结果不能当成本轮四方法 DCLM 资格。

2026-10-21。单机四卡、双机各一卡资格与原生 BF16 算子对照已通过；CPU、GPU、多机、科学训练分别记录。

### 已取得的证据

| 路径 | 实际检查 | 当前事实 |
|---|---|---|
| 控制协议 | frozen 无 Torch 环境的完整 suite | 286 passed，15 skipped；模型可选测试正确跳过 |
| 模型/数据/训练/HF/交付 | 完整 CPU suite | 108 passed，包含宿主 RNG seed、导出许可证、相对路径、FFN-chain 更新与精确路由负对照 |
| 社区 baseline | 官方 Qwen3-MoE 六配置、全部梯度、4 次 AdamW、exact cache | 已实际执行；数值与容差见 [baseline](baselines.md) |
| 原生分布式语义 | 真实两 rank CPU FSDP、accumulation、Full/选定 Selective AC、chunked head + aux | 通过；Gloo 使用 SUM 通信，不能据此推定 CUDA/NCCL |
| 数据 | 实际固定 revision WikiText-2/GPT-2 prepare、mmap loader、精确恢复 | 通过；显式 special roles 与 manifest 一致 |
| HF 交付 | 真实 DCP selective model load、离线新进程 AutoClass、生成 | 通过；FP32/BF16 dtype 与缓存有独立审阅 |
| 下游接入 | lm-eval 0.4.13 ARC-Easy 2 题及 HFLM likelihood/generation | 工程链路通过；随机权重和小 limit 不代表模型质量 |
| 原生 CUDA | H200 × 4，Torch 2.14.0+cu130、NCCL 2.30.7 | 六种代表配置、GA2、BF16/FSDP、Full AC、validation/full DCP、严格恢复和实际模型交付全部通过 |
| CUDA baseline | 官方 Qwen3-MoE 六配置各 2 次 AdamW | forward、全部梯度、更新与 cache 通过；与 CPU 四步检查分别记录 |
| 原生 BF16 算子 | H200 单卡，grouped experts/FlexAttention 对 plain expert loop/SDPA | 共 80 个物理参数张量；forward/辅助标量完全一致，每 token、每逻辑调用的 expert 集合精确相同，梯度通过首轮失败后、重跑前声明的双约束 |
| 安装产物 | 由 sdist 构建 wheel，checkout 外干净安装 | 无 Torch base import、模型 CLI、prepare → DCP → HF → heldout/generate → AutoClass 数值往返通过；最终 installed-wheel 导出 LICENSE 字节/hash 与 metadata 一致 |
| 多机 | 两台 H200 主机各一张 GPU，原生 NCCL/FSDP | 实际 TCP collective、GA2、validation/full DCP、严格 4 vs 2+2 恢复及 partial attention + Full AC 通过 |

CPU 环境：Python 3.12.3、Torch 2.14.0+cpu、Transformers 5.20.2。原生训练固定 TorchTitan `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`。GPU 实际环境与 source snapshot 单独记录，不用 CPU 环境的版本替代 GPU 环境的版本。

GPU 实测快照为 `6f5893daa5d5b337af1818a7862cb21312885d1d`，Python 3.12.15、Transformers 5.19.0、CUDA 13.0。训练参数使用 BF16，master parameters 与 gradient reduce 使用 FP32。六种配置为 shared、partial attention、partial router、layer expert tying、input injection 与 dense；其中 partial attention/injection 使用 Full AC，injection 实际经历深度 3、2 和一轮 suffix 梯度。各配置均完成 2 updates，shared 连续路径完成 4 updates。

严格恢复比较连续 4 步和 2+恢复后 2 步：27 个模型张量、230272 个元素的最大绝对差为 **0**；四个 ranks 的 Python/NumPy/Torch CPU/CUDA RNG、数据 cursor/source counts、progress 与 LR scheduler 离散状态完全一致。累计有效 input/label tokens 均为 1024。比较仅读取模型和 continuation 状态；optimizer 保留在 full checkpoint 中，未逐元素比较其所有张量。

真实训练 DCP 导出为 FP32 HF 产物后，64 个有效 heldout targets 的 CE 为 3.910506；cached generation 和脱离项目/TorchTitan 路径的独立 HF 加载通过。该 fixture 由本地文本生成，不代表预训练质量。完整机器可读结果见[四卡资格](evidence/gpu-single-node.json)、[CPU baseline](evidence/qwen3-cpu.json)与[CUDA baseline](evidence/qwen3-cuda.json)。

双机资格使用固定源码 `2fac1074fcd1e92cd3f08865e7b98851374611c2`，训练库与上述四卡快照一致，GPU 软件版本相同。原生节点身份核对确认两台不同物理主机，每台实际可见一张 H200；NCCL 两 rank 的 BF16 all-reduce 均得到 `[3, 5]`，原生日志确认使用 NET/Socket TCP。训练使用 `dp_shard=2`、GA2、全局 batch 4、序列长度 32，分别完成 shared 连续 4 步、stop2/resume4 和 partial attention + Full AC 2 步，均有逐步 validation 和 durable full DCP。

双机连续与恢复路径的 27 个模型张量、230272 个元素最大差为 **0**；两个 ranks 的 Python/NumPy/Torch CPU/CUDA RNG、data cursor、progress/source counts 与 scheduler 离散状态精确一致，恢复后的逐步指标与连续路径尾段相同，累计有效 input/label tokens 均为 512。full optimizer checkpoint 的加载已实际执行，但未直接逐张量比较 optimizer；不增加此项相等性结论。[双机机器可读证据](evidence/gpu-multi-node.json)同时保留真实拓扑检查、源码身份与运行结果。这是小型工程 fixture 的正确性资格，不是 RDMA、吞吐或 scaling 性能证明。

原生 BF16 算子对照使用公开脚本 `scripts/compare_native.py`。早期执行 commit 为 `2fac1074fcd1e92cd3f08865e7b98851374611c2`，脚本 SHA256 为 `bbaa9c02922fd90d0bfb706dd8d787b65d7a28a2d2d7d040c11e5f9adb602e0d`；模型/训练库与上述四卡资格快照一致。shared 和带 prefix/suffix、linear input injection、suffix 反传的两种配置分别核对 27、53 个物理参数张量，均包含 padding/document mask 与 balance/z 辅助目标。两配置的 logits、loss、辅助标量和 router assignment 汇总计数完全一致；全局梯度 relative L2 分别为 `0.00597881`、`0.00528446`，全部张量中最差 relative L2 为 `0.00995642`，最大坐标差除以所在参考张量 peak 的最差值为 `0.013216`。

补充的精确路由资格执行 commit 为 `75543a360c4f398e7c0c836c689a23e0c6141e3f`，脚本 SHA256 为 `de2ccdbf9b3e95aa0280494aafae8cf839fc886dace14fee8d86b4ce5f0c6555`。脚本逐逻辑 Router 调用核对 owner、有效输入 mask 和每个有效 token 的 top-k expert 集合，允许 top-k 返回顺序不同，并纳入 no-grad visits。shared 的 4 次调用、400 个 expert assignments 和 injected suffix 的 8 次调用、800 个 assignments 均精确一致；前向与 80 个物理参数的梯度检查也通过，精度门槛保持不变。[schema-2 完整重测](evidence/native-kernel-routing-qualified.json)保存逐调用 IDs hash 和实际比较结果。旧 schema-1 的 `router_assignments_equal` 字段只比较汇总计数，不能证明逐 token/调用相同；旧记录保留，并由新增的 token/visit 交换负对照约束这一边界。

梯度验收同时要求每个张量及全局向量的 relative L2 ≤ `0.03`，并要求每个张量 `max_abs(g_native - g_ref) ≤ 2 * eps_bf16 * max_abs(g_ref) + 1e-6`，其中 `eps_bf16 = 0.0078125`。这项有限输入资格参考 BF16 精度与张量尺度，不是任意归约、模型或硬件的误差保证。

首轮逐元素 `atol=0.0005, rtol=0.05` 检查中，shared embedding 的 4096 个梯度坐标有 1 个未通过，原始任务与报告仍保留 **Failed**。首轮失败后经数值策略审阅改用上述双约束，并在重跑前声明；随后以相同模型、输入和未改动的训练库重新运行公开脚本并通过；两次完整逐张量结果分别见[首轮失败记录](evidence/native-kernel-initial-failed.json)和[重新验收记录](evidence/native-kernel-qualified.json)，原始记录均保留。

截至 **2026-10-21 10:11:36 UTC**，已完成 GPU 资格批次的累计保守资源计数为 **0.51972 GPU-hours**，包含此前两次失败尝试和本次双机资格的 **0.09056 GPU-hours**。本次按 163 秒、两张 GPU 计数。这是历史资格批次的快照。

[安装产物检查](evidence/package-installation.json)使用另一个独立环境，从 git archive 构建 sdist，再由 sdist 构建 wheel。模型、数据、训练和 HF runtime 均可安装；源码包不包含研究文档和资格脚本，这些从 Git checkout 使用。控制协议的 Git capture 也按设计需要原 Git checkout。

[最终安装包检查](evidence/package-final.json)使用 `2e000a49ff788e1ecdb7d816c4cfbfdf23b966c8` 的全新 archive/sdist/wheel，在 checkout 外通过 `python -I` 导出，实际读取 installed distribution 的许可证；导出的 LICENSE 与 wheel metadata 字节相同，并由 provenance hash 覆盖。14 个公开模块与源码 archive 逐字节相同，另一个未安装项目/TorchTitan 的 HF-only 环境完成本地加载、精确 cache 与 greedy 重算对照，缓存误差为零。

路径修复后的源码 `a19c6fa611f564078a45533d8c92ce096b5564e1` 已重新完成[安装包检查](evidence/package-paths.json)，覆盖新的 archive/sdist/wheel、14 个公开模块的逐字节身份、installed LICENSE、真实 DCP 导出、heldout/generate 和独立 HF-only 消费者。原安装包记录保留。

同一源码另在 H200 上完成[相对路径端到端回归](evidence/relative-path-export.json)：从相对的 manifest、tokenizer 和输出目录启动真实 native BF16 训练，完成 GA2、两次更新、逐步 validation 和 durable full DCP。训练元数据记录绝对路径；切换工作目录后，自动发现与显式指定 tokenizer 的导出均通过，27 个模型张量完全一致，64 个 heldout targets 与 cached generation CLI 通过。首次检查脚本误用不存在的 provenance 字段而报错；修正检查字段后验证通过，原 driver 失败与产品命令成功分别保留。这次只改变训练入口的路径规范化，未改变模型或优化计算；此前四卡/跨机数值证据仍对应各自记录的源码快照。

### 已发现并修正的实际问题

- StatefulDataLoader 创建 iterator 会消耗训练全局 CPU RNG：给 loader 独立 generator，真实 next-batch/RNG 回归通过。
- Selective AC 与每轮独立 attention 的 nested FSDP 重算不兼容：明确拒绝该组合，Full AC/关闭 AC 保持支持。
- 训练 continuation identity 未包含完整 loss config：现已包含 loss、AC/compile、seed policy、source hash、数据/tokenizer 与并行/optimizer/schedule。
- 显式 BOS/EOS/PAD 角色可能在 export 中丢失：准备阶段规范化复制后的 assets，manifest/训练/导出校验同一身份。
- HF eval 普通梯度调用误建 inference cache：自动 cache 仅用于无梯度推理，保留显式配置选择。
- 原生框架初始化 Torch seed 并未初始化 Python/NumPy：显式初始化 rank-specific 宿主 RNG，保持 Torch 生成器不变，真实四卡 continuation 重测通过。
- standalone HF 导出原先遗漏推理源码的许可证：现从源码 archive 或 installed wheel metadata 读取 LICENSE，随产物写入并计算 hash；实际 installed-wheel 路径通过。
- 相对 manifest 路径原先在训练与导出中按不同目录解释：训练入口统一记录启动目录下的绝对 manifest、tokenizer 和输出路径，目录形式的 manifest 输入规范为其中的 `manifest.json`；CPU 回归与真实两步训练到跨目录 HF 导出通过。

### 0.1 支持集合

目标范围是 plain reference core、单状态 causal LM、softmax normalized top-k/dropless、GQA/RoPE、terminal CE、AdamW、同拓扑 continuation、独立 HF exact cache。固定/full 支持 ordinary/stack/layer/典型部分共享；sampled depth 与 suffix 限 fully shared stack。严格数据继续目前要求 `num_workers=0` 与 immutable shared manifests。

CUDA graphs、任意 model compile、EP/TP/CP/PP、弹性拓扑恢复、动态 token halting、近似 KV、多状态模型和其他 optimizer 不属于当前已验证承诺。CPU 构造出某个组合不等于已取得原生机器资格。

最终 GPU 资格执行使用固定源码快照、明确的程序时限与小步数，记录真实成功 updates、durable checkpoint、各 rank 状态及程序退出。

### 历史证据不支持的推断

这些证据不能证明任意模型规模/硬件/全部开关组合、长期稳定性、吞吐领先或论文分数。工程 fixture 的 loss 下降仅用于发现训练更新断点，不代表 data recipe、预训练效果或 scaling 结论。
