# 成本与比较契约

[`moeloop.costs.estimate_cost()`](../src/moeloop/costs.py) 与 [`moeloop.comparison`](../src/moeloop/comparison.py) 返回普通 JSON，不分配 GPU、不启动训练。参数存储、逻辑执行、数据暴露和实测结果分别报告。

```python
from moeloop.recipes import foil_base_spec, foil_spec
from moeloop.costs import estimate_cost
from moeloop.comparison import compare_recipes, diff_recipes

base, foil = foil_base_spec(), foil_spec()
cost = estimate_cost(foil, seq_len=2048,
                     activation_checkpointing="full_cells")
print(cost["formula_version"])  # fixed_terminal_matrix_v1
print(cost["calls"]["forward_cells"])  # 6
comparison = compare_recipes(base, foil, seq_len=2048)
print(sorted(comparison["unknown"]))  # ['data', 'evaluation', 'quality', 'runtime']
print(len(diff_recipes(base, foil)) > 0)  # True
```

采样深度的 recipe 同样返回数字，不再是 `None`：

```python
from moeloop.costs import estimate_cost
from moeloop.recipes import huginn_spec

depth = estimate_cost(huginn_spec(), seq_len=2048)
print(depth["formula_version"])  # sampled_depth_matrix_v1
print(depth["flops"]["training_depth"]["quantile_visits"])  # {'p50': 4, 'p95': 8}
```

## 报告口径

| 字段 | 含义 |
|---|---|
| `parameters.unique_total/unique_trainable` | 每个实际 owner 计一次；trainable 默认未冻结模型 |
| `parameters.unique_non_embedding` | 同时排除独立 input embedding 和 output head，其余参数均保留 |
| `parameters.by_component/by_region` | attention、norm、router、expert、head 及各区域 owner/几何 |
| `calls` | 训练 forward/backward/recompute Cell、router、expert assignments、terminal head，以及独立的 inference_cells |
| `flops` | 每 padded input position 的矩阵 forward/backward/recompute，含独立 head/router 分项 |
| `tokens` | 估算采用的 batch/context/packed positions，实际有效 input/labels 待运行提供 |
| `kv` | 以 spec.inference.num_visits 计算每逻辑 attention 调用的 K+V，指定 context/batch/dtype 下的 bytes |
| `unknown` | 未估计的算子、optimizer、运行内存、实际 tokens 和实测工作量 |

默认公式 `fixed_terminal_matrix_v1` 用一次 MAC=2 FLOPs；attention QK/AV 计两个稠密 S×S 矩阵乘，不扣 causal triangle/document mask。显式传入 `attention_pairs_per_sequence` 时使用 `fixed_terminal_document_matrix_v1`：按实际 input mask/document边界统计每条 packed sequence 允许的 causal query/key pairs（含对角线），传入其均值；不由 label 数或平均文档长度猜测。`estimate_cost()`、`compare_recipes()`、`smelt_p1_pair()` 都接受该参数。

两种口径的 full-BPTT 矩阵 backward 均估为 forward 两倍。`full_cells` 追加 Cell 矩阵的一次 replay，terminal head/input adapter 不在这些 Cell 内。`activation_checkpointing=None` 表示未知 recompute，不当成零。文档 pairs 校正的是分析矩阵工作量，不等于 FlexAttention 实际 block/kernel 工作量或 wall time。

采样深度和尾段梯度（`TrainingRecipe` 的 `sampled_visits`/`depth_rule`、`grad_visits`）使用 `sampled_depth_matrix_v1`（给定 `attention_pairs_per_sequence` 时为 `sampled_depth_document_matrix_v1`）：前向按期望深度计，反向与重算覆盖 prefix、suffix、head 和最后 `min(grad_visits, k)` 个 core visit，对声明的深度分布取期望（工作量对 k 与 `min(grad_visits, k)` 是线性的）。`flops.training_depth` 另给该规则下 p50/p95 深度的单次成本（`quantiles.p50/p95.training_per_token`，每 padded position，对应一次 update 或一个 microbatch），工作量随深度不减，所以它们是典型与偏大的单次代价。这些是声明分布上的估计，不是某次运行抽到的深度或实测工作量，后者看训练记录里每 microbatch 的实际 r。采样深度下 `calls` 的逻辑调用数为 `None`，nominal 深度的值另列在 `calls.nominal`。尾段梯度却没有 input injection（最终损失到前缀的路径被截断），或目标不被支持时，没有可信 training FLOPs，相应值为 `None` 并说明原因。dispatch、norm、softmax、CE、optimizer、通信和 allocator 未计项保留 unknown；矩阵 FLOPs 不替代 wall time、MFU 或峰值显存。参数共享不合并逻辑 KV slot。

固定深度的训练 calls/FLOPs 使用 `model.num_visits`，可信训练估计还要求 `training.num_visits` 与之匹配（采样深度由深度规则决定，`model.num_visits` 只给 nominal 前向）；KV 独立使用 `spec.inference.num_visits`，缺省才回落到 model 深度。输出的 `calls.inference_cells`、`kv.num_visits` 和 `kv.logical_slots` 明示这一差别。因此只改声明的 inference depth 不会改训练 FLOPs，却会显示实际逻辑 KV 差异；per-visit 模块超过 `visit_capacity` 直接拒绝。比较需要分别固定训练和评价深度，不能用训练深度覆盖推理声明。

## CLI 和普通 JSON

```sh
python scripts/compare_recipes.py inspect smelt --seq-len 2048
python scripts/compare_recipes.py diff foil_base foil
python scripts/compare_recipes.py compare moe smelt --seq-len 2048 \
  --activation-checkpointing full_cells
python scripts/compare_recipes.py compare recipes/p1_moe.json recipes/p1_smelt.json \
  --seq-len 4096 --attention-pairs-per-sequence 4180216.9229844734
python scripts/compare_recipes.py pair > /tmp/moeloop-smelt-pair.json
```

在 checkout 根运行，或使用脚本绝对路径；recipe 参数也可用 `RecipeSpec.to_dict()` 保存的 JSON 路径。`inspect` 输出 recipe/cost，`diff` 展示 resolved model/training/optimizer/run/inference 的实际差异，`compare` 生成比较记录。区域继承先 resolve，避免把显式写出继承值当成机制差异。`inspect/compare` 支持显式传入文档 pairs；`diff` 不计算成本，`pair` 保留自己的小型 dense-S² 合约，均不接受该参数。

`compare_recipes(reference, candidate, seq_len=..., data=..., evaluation=...)` 可接收普通数据/评价身份。比较前固定 question/arms、manifest/tokenizer、validation/test 隔离、有效数据暴露、objective、optimizer/schedule、seed、终点、推理深度/读出、共同 heldout 指标（CE 或字节精确的 BPB）和选择规则。CLI 未附加实际身份时，相应项保持 unknown。

四个分类是 `matched`、`controlled`、`measured-different`、`unknown`：前三个分析预算在容差内才 matched，确实相同的配置才 controlled；`measured-different` 保存已算出的预算差异，以 `basis=analytical_estimate` 说明来源，组名不表示做过 GPU 实测。未知不会被判为 matched。

## 历史小型工程配对

`smelt_budget_pair()` 固定枚举 SMELT E∈{8,10,12,14,16} 五个形状，返回选定 E=12 的比较。baseline 是 12 个独立层、每层 8 experts；SMELT 是 8 物理层（2 prefix、4 core×2、2 suffix）、每 bank 12 experts；均有 12 次逻辑 Cell 调用。共同 d=128、单 expert hidden=256、top-k=2、heads=4/2。

API 默认 vocab50257、context2048、batch1、BF16 KV、无 AC 时，unique non-embedding 参数为 10,042,496 / 9,844,864，矩阵 training FLOPs/position 为 94,114,560 / 94,151,424，逻辑 KV 均为 6,291,456 bytes。三项均满足相对 baseline 的绝对差异≤5%；5% 是工程规则，不冒充原文容差。这是默认分析例子；DCLM 实际运行须显式使用 manifest 的 vocab151669，重新计算含 head 的工作量。`pair` 固定其 CLI 假设，其他 AC/dtype/batch/容差用 `compare`。

该 JSON 证明声明的分析预算配对。真实 pair pilot、四方法短训、恢复和 HF 证据由实际产物核验，见[验证记录](validation.md)；它没有给出质量排名。

## P1 与文档内 attention 预算

`smelt_p1_pair()` 固定普通基线 d672/L10/E64/top8/F544/heads7/KV1，SMELT 为 d576、10物理层（2+5+3，中间5层执行两遍）、E96/top8/heads9/KV1；只在12个预声明 FFN widths（384至472、步长8）中按最大分析预算差异选形，不使用 heldout 质量选形。两臂 vocab151669、context4096，SMELT 共15次逻辑 Cell 调用。

普通基线的100M-active档是一次遍历约98.5M active non-embedding body；唯一 non-embedding 参数约712.7M，含两个独立词表 owner 的总参数约916.5M。实际 SMELT F440 约737.8M non-embedding unique、总参数约912.6M；active、unique、逻辑访问工作量分别由输出字段解释，不能用“100M”指总模型大小。

本次数据已完成 train24,000,067,249、validation10,001,497、test10,009,889有效 labels；两臂实际完成各19,537次更新、20,486,029,312 fresh labels。预算采用[完整训练 prefix 统计](evidence/p1-data-statistics.json)：epoch0、seed42、world16下5,001,472个4096-token blocks，无padding，每序列允许的文档内 causal query/key pairs 均值为 **4,180,216.9229844734**，包括对角线。它覆盖完整拟训练 prefix，已用实际 loader 交叉检查，不是20M-label小样本，也不等于 dense-S² 或完整 causal triangle。

直接加载发布的精确 JSON 复算；下例仅运行 CPU 分析，不启动训练：

```python
import json
from pathlib import Path
from moeloop.recipe_spec import RecipeSpec
from moeloop.comparison import compare_recipes

reference = RecipeSpec.from_dict(json.loads(Path("recipes/p1_moe.json").read_text()))
candidate = RecipeSpec.from_dict(json.loads(Path("recipes/p1_smelt.json").read_text()))
comparison = compare_recipes(
    reference, candidate, seq_len=4096, batch_size=1, bytes_per_element=2,
    activation_checkpointing="none", tolerance=.05,
    attention_pairs_per_sequence=4180216.9229844734,
    data={"manifest_fingerprint": "e67a956d2fe51dd1784505a6f58d7790f397a09624330700c0ebaffcd344f152"},
    evaluation={"seq_len": 4096, "readout": "terminal", "full_validation_and_test": True},
)
print(json.dumps(comparison["matched"], indent=2))
```

| 分析预算，按 reference 的相对差 | 普通 MoE | SMELT | 差异 |
|---|---:|---:|---:|
| unique non-embedding 参数 | 712,656,672 | 737,845,056 | +3.5344% |
| training matrix FLOPs / padded position | 1,284,757,348.671 | 1,248,742,044.863 | −2.8033% |
| logical KV bytes，batch1 / 4096ctx / 2 bytes | 15,728,640 | 15,728,640 | 0% |

三个预算都落在预声明5%以内；batch1仅为成本报告单位，实际训练 global batch 为256条序列。固定12个FFN候选的预算筛选选中 F440，未使用质量指标。`p1_smelt_spec()` 给出这个 F440 臂（实测 pairs 口径，与 P2 相同）；无 pairs 的 `smelt_p1_pair()`/`recipe_anchor.py` 按 dense-S² 选出从未训练过的 F424；生成新 provisional JSON 不会生成本次实际冻结身份。数据重建、16卡2×8 HSDP 与完整 WSD 命令见[训练文档](training.md#p1-实际配方与双机训练)。

本次训练源码为 `1bea4d6d2c539e3ae02517aaeede5f9ab68271a7`。两臂相同数据顺序、seed42、batch/microbatch、优化器、终点与读出；训练、数据、运行与评价身份在开跑前冻结。后续公开入口补丁只暴露已有参数和实际工件，不将其新 HEAD 写成训练来源。

### 如何读 P1 结果

[P1 证据](evidence/p1.json)分开报告完整约10M-label validation/test 端点、[原生约1M-label验证曲线](evidence/p1-native-curves.svg)及[末段放大](evidence/p1-native-late-curves.svg)、实际 GPUHr 和三预算。相同数据量 D 的端点回答等数据下的质量；共同解析计算 C 的曲线只在两臂共同观测范围内插值，并保留相邻点/权重，不外推到未训练区域。C 使用冻结矩阵公式乘以真实 padded positions，不能称为实测 GPU FLOPs。完整FP32 HF终点评价与原生BF16固定验证片段不混成同一条曲线。

SMELT−MoE 的 CE 差为负表示前者更好。配对文档评价要求同一数据/上下文/读出与各文档 label counts；它不估计训练 seed 方差。本次只训练 seed42；小幅或没有优势的结果照实保留，不以单点领先声称普适收益或完整复现 SMELT 论文。

运行成本与实现对照也分开：P1 如实报告两种结构的墙钟和 GPUHr。普通 MoE 与官方 TorchTitan 的[同几何8卡 FSDP 对照](evidence/performance-fsdp8.json)测得 median step time 开销1.95%，使用 GPU-resident synthetic 数据、无数据加载/验证/存盘；它不覆盖跨机或全部方法，也不是完整BF16逐梯度等价证明。SMELT 有15次逻辑 Cell 调用，MoE有10次；两种模型的成本差不能直接归因于同一模型实现的性能回退。

## 比较回答的问题

同底座机制消融保持无关机制和评价一致；资源匹配 recipe 竞争允许明示结构/optimizer 差异，记录预算与替换；总训练计算下的数据/深度分配还需要实际工作量和数据暴露。三类问题分别报告。跨方法质量采用共同标准 LM heldout：Bench 和跨 tokenizer 的比较用字节精确的 BPB（`evaluate_heldout` 的 `bpb`），CE 同时保留；训练 objective 单独报告；后续 Ouro expected/entropy 目标不能与 terminal CE 混排。

## Looped MoE Bench

[`eval/bench/tracks.json`](../eval/bench/tracks.json) 冻结了四条 track：S（C-grid 规模，约26M unique non-embedding 参数，单卡入门）、M（P1 几何，5B labels 的筛选）、L（P1 几何和冻结的 P1 schedule，约20.5B labels）、P2（P2 几何，约200M 激活参数，5B labels，Phase 1 研究的主轨道，其余规则同 M）；数值以该文件为准，S 的 label 预算标为 provisional，首次参考 run 前由人重新冻结。每条 track 固定数据 manifest 与 tokenizer、序列长度、global batch、schedule、优化器设置、seed 规则，以及学习率（S/M 档固定学习率网格，L 与 P2 档每个优化器一个学习率），评价是完整 validation/test 上的字节精确 BPB。

预算按本页的 `compare_recipes` 判定：unique non-embedding 参数、训练矩阵 FLOPs/token、逻辑 KV bytes 三项相对 track 参考 recipe 的差异都在 5% 以内；口径是稠密 S²、不计重算，数的是模型工作而不是执行策略。S 和 M 的 LR 在 validation BPB 上按网格选择，L 与 P2 每个优化器只有一个固定 LR、不做试验；排名只看 test BPB；seed 42 必选，其余可选，榜上给均值、范围和 seed 数。GPU 小时与吞吐只报告，不是资格条件。

[`moeloop.bench`](../src/moeloop/bench.py) 读一份提交 JSON，给出 `eligible`、`budget-mismatch`、`protocol-mismatch`、`missing-evidence` 之一并列出全部原因；`leaderboard` 按 test BPB 均值给合格提交排名：

```python
import json
from pathlib import Path
from moeloop.bench import Submission, leaderboard, load_tracks, validate

submission = Submission.from_dict(json.loads(Path("submission.json").read_text()))
report = validate(submission, load_tracks()[submission.track])
print(report.verdict, report.reasons)
print(leaderboard([report])["rows"])
```

核对器只检查提交里声明的内容和证据哈希是否齐全，不重跑训练或评测、不打开被哈希的文件、不判断质量。入口是 `moeloop.bench` 的 Python API；命令行（run、sweep、validate、leaderboard）由 `scripts/bench.py` 提供。

## 状态与多轮目标的成本边界

固定expected CE使用`fixed_expected_matrix_v1`（显式document pairs时为对应document版本）：每轮coda/head/gate前向、矩阵反向，以及chunked CE checkpoint额外的head投影重算分别计量；最终gate的概率不作为额外退出，训练仍执行所有轮。EMA/RMS/entropy等非矩阵工作按现有policy不计入矩阵FLOPs并明示，不把它们当实测GPU成本。

随机深度/尾段梯度的训练work按上文 `sampled_depth_matrix_v1` 取期望，p50/p95的单次成本另列；nominal固定深度示例只是一次前向，不是实际工作。每microbatch实际r/min(k,r)及真实token计数由训练进度另记录。参数owner、固定推理KV和cache资格独立报告，uncached预算不表示实现了缓存。不同训练objective的loss不直接排名，使用有明确推理策略的共同terminal LM heldout。

[新增方法证据](evidence/recipes-v2.json)按冻结估计器的默认dense-S²矩阵口径记录理论成本，和实际native GPUHr分开；#6特定document统计预算保持原身份。该证据早于采样深度的成本模型，其中 Huginn 的 `training_per_token` 仍是 `None`（`fixed_terminal_matrix_v1` 的 nominal 记录）；证据记录不改写，现在重新估计会得到上面的期望值。
