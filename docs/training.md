# 训练、数据与模型交付

研究者只需读取普通 Python recipe、JSON 和明确的 model/objective。TorchTitan 提供外层训练生命周期，公共代码无需 GitHub、MLflow 或内部部署服务。已有 prepared 数据时，可直接从[JSON 运行与恢复](#从-json-运行与恢复)，再[导出、评测和生成](#导出评测和生成)。

## 安装环境

已核对 Python 3.12、Torch 2.14.0、TorchTitan `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`；`train` extra 从 PyPI 安装 `torchtitan==0.3.0`，即该 commit 的 tag v0.3.0。CUDA 环境先按硬件安装 Torch 2.14.0，再安装项目：

```sh
uv venv --python 3.12
. .venv/bin/activate
uv pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu130
uv pip install -e '.[data,hf,train]'
python -m moeloop.train --config tiny --help
```

CPU 参考模型/HF 使用官方 CPU index 和 `.[data,hf]` 即可。训练与控制环境分开；`uv run --frozen` 面向不含 Torch 的实验控制环境，已安装额外训练依赖后直接使用该环境的 `python`。

训练入口检查实际 import 的上游源文件，而非仅检查发行版名称。离线集群可将相同上游源码放在 `PYTHONPATH`，无需修改共享 Python 环境。不同上游版本需要单独审阅 adapter；不会静默降级到另一个模型。

## 数据准备和 mixture

WikiText2 安装/工程示例，不作为方法有效性或最终筛选数据：

```sh
python -m moeloop.prepare --recipe wikitext2_gpt2 --output /data/demo
```

固定 WikiText-2 raw 与 GPT-2 tokenizer revision，只处理原始非空行。每行是一个文档；明确添加 BOS/EOS，文档内做 next-token shift。它是流程示例，不等于外部 WikiText PPL 榜单的上下文协议。

P1 已完成公开原始 DCLM 的数据准备。重放时直接使用已发布的[原始1024份有序 catalog](evidence/p1-dclm-source.json)，不要重新冻结另一份清单：

| 身份 | P1 实际值 |
|---|---|
| 原始数据 | `mlfoundations/dclm-baseline-1.0-parquet@817d6752765f6a41261085171dd546b104f60626` |
| tokenizer | `Qwen/Qwen3-30B-A3B@ae659febe817e4b3ebd7355f47792725801204c9`，vocab151669 |
| 有效 labels | train 24,000,067,249；validation 10,001,497；test 10,009,889 |
| prepared manifest fingerprint | `e67a956d2fe51dd1784505a6f58d7790f397a09624330700c0ebaffcd344f152` |

准备到第198份 Parquet 时达到各 split 预算，扫描19,624,997篇原始文档、去除10,762篇重复文档，无空文档。原始 document ID 与精确 UTF-8 内容 SHA 全局去重；按内容 hash 前8 bytes 的 big-endian 整数 mod100 分割，0=test、1=validation、其余=train，各 split 在完整文档边界达到预算。**保留完整1024份 catalog**：只需下载实际读取的前缀，将 catalog 截成198份会改变 provenance 和 manifest 身份。

先将固定 tokenizer 下载到一个新的目录，**仅取以下四个源文件**；`config.json` 等额外源文件也会进入身份记录，不能混入：

```sh
export QWEN3_TOKENIZER_FOUR_FILES=/assets/qwen3-p1-tokenizer
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-30B-A3B", revision="ae659febe817e4b3ebd7355f47792725801204c9",
    allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    local_dir=os.environ["QWEN3_TOKENIZER_FOUR_FILES"],
)
PY
```

`DCLM_PARQUET` 指向含 catalog 相对路径的本地 Parquet 根目录，`P1_DATA` 为新的 prepared 目录。按本次实际预算和 shard 大小准备：

```sh
export P1_DATA=/data/dclm-p1
python scripts/recipe_data.py --dclm-source docs/evidence/p1-dclm-source.json \
  --source-directory "$DCLM_PARQUET" --tokenizer "$QWEN3_TOKENIZER_FOUR_FILES" \
  --train-label-tokens 24000000000 --validation-label-tokens 10000000 \
  --test-label-tokens 10000000 --shard-tokens 128000000 --output "$P1_DATA"
cmp "$P1_DATA/manifest.json" docs/evidence/p1-data-manifest.json
```

该命令不下载模型权重。省略 `--source-directory` 会按冻结 catalog 使用 Hugging Face cache/下载；其他数据使用者可用 `--freeze-dclm-source` 建立自己的清单，但不再是这次 P1 的原始身份。原始 tokenizer.json SHA 为 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`；准备时显式规范化 BOS=EOS=PAD151645，输出 tokenizer.json SHA 为 `a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794`，完整源/输出资产 hash 见 [manifest](evidence/p1-data-manifest.json)。

中断后在相同源清单、tokenizer、预算和输出目录的原命令上追加 `--resume`，从最后提交的输入 shard 继续。完成后的 token/index shards 与 manifest 保持 immutable；不足时不能循环小集冒充 fresh labels。公开仓库包含元数据和各 payload 的 hash，不包含约96GB的 prepared tokens。

早期工程资格还使用已有 DCLM 预分词文档的 legacy importer。其默认300M/1M/1M fixture 继续用于 smoke，不满足 P1 的完整数据供给；`DCLM_PREPARED` 指向该旧格式的源 manifest 与 payload：

```sh
export MOELOOP_DATA=/data/dclm-qwen3-v1
python scripts/recipe_data.py --packed-source "$DCLM_PREPARED/manifest.json" \
  --tokenizer "$TOKENIZER_DIR" --output "$MOELOOP_DATA"
```

源格式为已有按文档保存的 little-endian uint32 tokens、累计文档末尾 offsets、原文 blake2b64 sidecar。导入固定前两个 train chunks，不静默追加第三块；校验原文件 size/SHA、offsets、token range 和逐文档末尾 EOS。默认至少 train 300M、validation 1M、test 1M 有效 shifted labels，完整文档边界略超，为后续 268M-position horizon 预留同一内容身份。

tokenizer 固定为 `Qwen/Qwen3-30B-A3B@ae659febe817e4b3ebd7355f47792725801204c9`，实际 vocab_size=151669。源序列为 EOS-only（151645），导入不解码、不重新分词、不追加第二个 EOS，只前加一个显式 BOS=151645；PAD 也设为151645。原 token body/EOS 保持不变，这不是原 Qwen 已配置 BOS。源 tokenizer 资产 SHA 与输出规范化 role/config 的 SHA 分开记录，词表 ID 映射不变。

按原 token 文档的 uint32 little-endian 字节（含原 EOS）计算 SHA256、全局去重，以 hash 前8 bytes 的 big-endian 整数 mod100 分配：0=test、1=validation、2..99=train。各 split 在整篇边界达到预算后停止接收。sidecar 用 parent chunk 身份/文档 ordinal 定位，保留 token SHA 和原文 blake2b64；它们不是完整原始 HF/WARC document ID。

来源是公开 [DCLM baseline 家族](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)，其公开卡记录为 CC-BY-4.0。上述 legacy 副本的上游下载 commit 未知，不能拿另一次观测的公开 HEAD 冒充；其内容 revision 由 parent manifest 与 tokens/docs/dochash SHA 标识。新的原始 Parquet 路径使用 `--freeze-dclm-source` 记录的公开 revision 和逐文件身份。完整清单、许可、选择规则、实际 labels、tokenizer 和 packing 进入各自 manifest；新数据不能静默继承旧 fixture 的身份。

历史300M工程 fixture 的[公共证据摘要](evidence/dclm-v1.json)保留 manifest fingerprint、两个 parent chunks 的 SHA、tokenizer 源/规范化资产 hash、实际 split 计数及 token SHA 隔离检查。实际有效 labels 为 train 300,002,777、validation 1,000,401、test 1,000,504；235,913篇选中文档的全局 token SHA 重复/跨 split 重复计数为0。精确预分词二进制 chunks 尚未公开发布，上游下载 commit 仍未知；该摘要标识实际内容身份，训练/恢复/HF 资格另见[验证记录](validation.md)。

FineWeb-Edu/GPT2 是独立的可选示例，不是 DCLM 的前置条件或更优数据声明：

```sh
python scripts/recipe_data.py --fineweb-edu --output "$FINEWEB_OUTPUT"
```

该显式 fallback 固定 FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` 的 `sample/10BT/000_00000.parquet`，下载/缓存并校验 size/SHA；复用本地文件时用 `--source-file "$FINEWEB_SHARD"` 替代 `--fineweb-edu`。不指定任何 source 选项会报错，不再隐式下载 FineWeb。它使用 GPT2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`，许可为 [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) 加 [CommonCrawl terms](https://commoncrawl.org/terms-of-use)，tokenizer 为 MIT。修改源/tokenizer/分割/预算须创建新 fixture；test 独立保留，不用于调参。

本地 JSONL/text 使用同一 writer：

```sh
python -m moeloop.prepare --input /data/train.jsonl --validation /data/valid.jsonl \
  --test /data/test.jsonl \
  --tokenizer /assets/tokenizer --bos-token-id 1 --eos-token-id 2 \
  --output /data/prepared
```

准备阶段输出 immutable token/index shards、tokenizer 与 manifest。显式 special ID 会规范化复制后的 tokenizer 角色，原始 assets 不变；训练、导出和生成保持同一 BOS/EOS/PAD。输出目录必须是新目录。

数据按文档内预测 pair 拼 fixed-length block，位置在每个 block 的文档 segment 内从零开始，跨文档不可 attention。长文档跨 block 不保留前一 block 的 KV，也不重新添加 BOS。尾部 padding 使用独立 input mask，labels=-100；PAD 可以与 EOS 同 ID。

多个 source 先各自形成等长 block，再按权重选择。因此权重描述 input-token mixture，而不是文档数量比例。用普通 Python 数据配置定义 source、split 和权重：

```python
from moeloop.data import PackedDataset
data = PackedDataset('/data/prepared/manifest.json', seq_len=1024,
                     source_weights={'web': 0.7, 'code': 0.3}, seed=42)
```

writer 的 `prepare_dataset` 接受 `{source: {split: texts}}` 迭代器；不会把整个语料的 token 装进 Python list。运行时 mmap 有界缓存。训练启动由 rank 0 校验全量 hash，其他 rank 核对同一 manifest 和文件尺寸；之后复用验证结果。所有 rank 必须看到相同 immutable 数据命名空间，运行中不修改 shards。

## 修改研究机制

`moeloop.recipes` 的九条 `*_spec()` 返回可在CPU上查看和修改的 `RecipeSpec`，包括 `huginn_spec()`、`loom_spec()`、`ouro_spec()`、`sandwich_spec()`、`relaxed_spec()`；其JSON、区域覆盖和方法适配见[配方](recipes.md)。用户的模型配置保持显式；DCLM的实际词表不同于内置小例默认值，普通factory应主动读取manifest并传给spec。

`trainer_config(spec)` 直接消费 `run.activation_checkpointing`：`none` 关闭 AC，`full_cells` 使用原生 Full AC；其他策略明确拒绝。`optimizer.zero_weight_decay_roles` 使用 `norm/embedding/lm_head/router` 的声明子集，空列表表示所有参数采用给定 weight decay。共同 adapter 当前支持 AdamW 与 Muon（`optimizer.name`）、FP32 master/reduce、core 专家 LR 作用域且无 weight-decay 补偿；其他显式设置会报错，不保留与执行不符的 metadata。

选 Muon 时，`optimizer.name` 写 `"Muon"`，并在 `optimizer.muon` 中给出且只给出 `optim.MUON_FIELDS` 列出的字段，多一个或少一个都会报错；最短写法见 `tests/training/test_muon.py` 的 `muon_spec`，Bench 固定的取值见 [`eval/bench/tracks.json`](../eval/bench/tracks.json) 的 `optimizers.fixed.Muon`。隐藏矩阵（见 `MUON_PATTERN`）由 DistMuon 更新，其余参数仍按 `optimizer` 里的 AdamW 设置更新。哪些参数是隐藏矩阵，只在 [`optim.py`](../src/moeloop/optim.py) 的 `MUON_PATTERN` 一处，理由与限制见[架构](architecture.md#5-训练动态必须看得见)：Muon 只走 GPU 上的原生 FSDP，GPU 验证见 #49。

通用方法模板默认 context 2048、global batch 32768 packed positions、512 successful updates、每64 updates 固定 validation，BF16 native experts。它们与下文的实际 P1 JSON 是不同规模的配方；`tiny` 保持四步安装 demo。实际运行资格另见验证页。

例如保存为 `my_recipes.py`，通过公开的短 adapter `trainer_config(spec)` 返回 native Trainer config：

```python
# my_recipes.py
import os
from moeloop.data import PreparedManifest
from moeloop.recipes import smelt_spec, trainer_config

def experiment():
    manifest = PreparedManifest.load(os.environ["MOELOOP_DATA"])
    spec = smelt_spec(vocab_size=manifest.tokenizer["vocab_size"])
    return trainer_config(spec)
```

```sh
export MOELOOP_DATA=/data/dclm-qwen3-v1
torchrun --standalone --nproc-per-node=4 -m moeloop.train \
  --module my_recipes --config experiment --dump-folder /runs/experiment
```

CLI使用原生Python ConfigManager。模型config决定参数形状与固定推理默认值，training recipe决定训练深度、梯度与目标。上例可将 `smelt_spec` 换成任一新增spec，仍通过同一个 `trainer_config(spec)` 接入；内置同名factory是模板，真实数据必须显式匹配词表。

| 训练规则 | 实际行为 |
|---|---|
| 固定/有限集合深度 | 固定深度或每update一次选择，同update各rank/microbatch相同 |
| Huginn-inspired采样 | 每microbatch按Poisson-lognormal采样r，跨rank同步；只对最后 `min(k,r)` 次stack visit反传，anchor保持live |
| LOOM full-BPTT | 固定H>2，global memory每forward清零、local每轮清零；无段间detach/optimizer step |
| Ouro Stage I | 全轮执行，逐token expected CE−entropy；唯一head与gate都训练，标准validation另用terminal CE |

随机初态在递归与AC之前只生成一次；深度以成功update/microbatch坐标定位，validation隔离RNG。`uniform_update_v1` 的深度序列与卡数无关；`poisson_lognormal_micro_v1` 按 microbatch 抽样，训练器取 rank 0 的 microbatch 序号，所以序列取决于累积步数 global batch ÷ (local batch × 卡数)。部分共享、layer loop、EMA和expected CE采用固定深度/full BPTT；缺少明确梯度定义的组合直接报错。

主 CE 使用全局整个 update 的有效 label 分母。aux 是 `local_micro_call_mean`：先按单个 microbatch 的可微 router calls 平均，再对整个 update 的逻辑 microbatch（global batch ÷ local batch）平均。改变 microbatch geometry 可能改变非加性的 aux，即使全局 batch 不变；global batch 与单卡 batch 都固定、且深度规则也与卡数无关（固定深度或 `uniform_update_v1`，不含 `poisson_lognormal_micro_v1`）、初始状态也不按 rank 抽取（随机 latent 初态用各 rank 自己的 RNG，目前只有 `huginn_spec` 如此，它已被其深度规则排除）时，aux 与卡数无关，GPU 验证见 #49。数据流与卡数无关是其前提之一：所有 rank 打包同一条逻辑 block 流，一次 update 的 block 按 rank 连续切片（rank r 取该 update 的第 `[r·K, (r+1)·K)` 个，K 为每 rank 的 block 数），所以不同卡数的运行每次 update 读到相同的 block、相同的顺序，每个逻辑 microbatch 的 block 也由数据流固定（规则名 `logical_blocks_v1`，见 [`data.py`](../src/moeloop/data.py) 的 `PackedDataset`）。forward 不更新持久 router 计数；AC 重算不会重复提交统计。

原生 grouped experts 执行 BF16；FP32 reference 的算术容差与 native GPU 资格分别记录。训练 head/CE 分块，避免强制创建完整 `[B,S,V]` logits。默认关闭 CUDA graphs/model compile，采用 `partial_dtensor` DP/FSDP。每轮独立 attention 可用 Full AC；Selective AC 与该 nested FSDP 组合会明确拒绝。

Ouro的概率/entropy使用稳定精度，忽略label同时排除CE与entropy；expected CE按整个update的全局有效labels归约，aux另行归约。训练可保留每轮hidden，但通过非重入checkpoint分块重算head/CE，避免保存每轮完整词表中间量。这项head重算即使Cell AC关闭也有计算成本；`fixed_expected_matrix_v1`单列它。Huginn随机深度与尾段梯度的训练FLOPs是声明深度分布下的期望（`sampled_depth_matrix_v1`），p50/p95深度的单次成本在 `flops.training_depth`；默认深度的nominal分解只是一次固定前向，二者都不是实测成本。

并行修改代码/recipe 时各用自己的 worktree，每个实验使用独立输出根及自己冻结的 recipe/source 身份。数据 manifest 与 immutable shards 可只读复用；不同 worktree 不共写 checkpoint、metrics 或 attempts。

## 从 JSON 运行与恢复

`scripts/recipe_experiments.py` 读取普通 `RecipeSpec` JSON 和 prepared manifest，在已分配 GPU 上调用同一个原生 Trainer。下面使用上文的 DCLM/Qwen3 数据，用 C 的一份 JSON 演示8步再接续至16步；它保留完整2048步 schedule，不代表正式 C 面板验收。其他 recipe 应显式匹配 manifest 词表与 batch/拓扑。

```sh
export SPEC="$PWD/recipes/c_a0_r0.json"
export RUN=/runs/experiment
python scripts/recipe_experiments.py run --spec "$SPEC" \
  --manifest "$MOELOOP_DATA/manifest.json" --output "$RUN" --gpus 2 --stop 8
python scripts/recipe_experiments.py run --spec "$SPEC" \
  --manifest "$MOELOOP_DATA/manifest.json" --output "$RUN" --gpus 2 --stop 16 \
  --resume "$RUN/checkpoint/step-8"
```

`--gpus` 是所有节点的总 rank 数。单节点默认使用 `--standalone`；多节点在每个节点运行同一命令，并设置 `NNODES`、`NPROC_PER_NODE`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。同一实验各节点共享同一个输出目录和相同输入身份；并发实验分别选输出根与 rendezvous，不互相复用端口。launcher 不申请 GPU。

首次运行保存 `spec.json`、`request.json` 和实际执行的 `source/` 快照，随后追加进程事实、`metrics.jsonl`、`outcome.json` 与 full DCP。已有更新必须显式传 `--resume`；runner 在启动前将完整 checkpoint 保留到 `protected/step-N`，避免 rolling keep2 清理恢复起点。保持原源码、recipe、runtime、数据路径与拓扑；`--stop` 是绝对成功更新坐标，不重置 optimizer 或 schedule。修改代码或实验控制后使用新的输出目录和 lineage。

安装后的公共 factory 是 `moeloop.experiment:experiment`，通过 `torchrun ... -m moeloop.train --module moeloop.experiment --config experiment` 使用。以 `MOELOOP_EXPERIMENT_SPEC`、`MOELOOP_EXPERIMENT_MANIFEST`、`MOELOOP_EXPERIMENT_OUTPUT` 指定普通文件路径，按需设置 `MOELOOP_EXPERIMENT_STOP` 与 `MOELOOP_EXPERIMENT_RESUME`。runner 的 `--installed` 选择已安装包而非源码快照；source97 的完整 wheel 用户旅程已通过 V3 实际验收，精确来源和收据见[验证记录](validation.md)。

可选 CVSM/MLflow 只记录已有命令与有限产物，见[实验协议](protocol.md)；它们不是准备数据、训练、恢复或 HF 消费的依赖。

## 新增方法的阶段资格

完整v2阶段资格已通过，见[实际证据](evidence/recipes-v2.json)。可在 GPU 上重放四条新增路径（Huginn、LOOM、Ouro及LOOM×attention）和attention×router四格；该阶段入口继承#6完整证据，不重复原预训练：

```sh
python scripts/recipe_qualification.py --stage v2 \
  --manifest "$MOELOOP_DATA/manifest.json" \
  --v1-evidence "$V1_EVIDENCE" --v1-source "$V1_SOURCE" \
  --output "$MOELOOP_EVIDENCE" --gpus 2 --sharing-grid
python scripts/check_release.py --stage v2 --evidence "$MOELOOP_EVIDENCE"
```

`V1_EVIDENCE`指接受过验收的完整持久产物目录，`V1_SOURCE`指其原冻结P1源码身份JSON；公开`docs/evidence/p1.json`只是投影，不能代替原始bundle。默认每新增方法512步、2048ctx、global16序列，是工程资格；四格的短更新不是效果排行榜。新增方法的实际恢复、HF范围与最终状态由验证记录列出。

## P1 实际配方与双机训练

本次实际配方为 [`recipes/p1_moe.json`](../recipes/p1_moe.json) 和 [`recipes/p1_smelt.json`](../recipes/p1_smelt.json)。直接加载这些 JSON；`p1_smelt_spec()` 给出实际训练的 F440 臂（实测 pairs 口径，与 P2 相同），不带参数的 `recipe_anchor.py` 仍生成 dense-S² 的 F424 模板，不能替代实际 F440。两臂已经完成19,537次更新，各 **20,486,029,312 有效 labels**，无 padding、无重复 epoch、唯一训练 seed42。模型几何见[配方](recipes.md#p1-普通-moe--smelt)，实际曲线、完整终点评估和 GPUHr 见 [P1 证据](evidence/p1.json)。

训练来源固定为 `1bea4d6d2c539e3ae02517aaeede5f9ab68271a7`，TorchTitan 为上述固定 commit；实际环境 Python3.12.15、Torch2.14.0+cu130、CUDA13.0，HF 消费使用 Transformers5.19.0。后续公开工件与预算 CLI 的接线不改变模型数学，也不代表发布 HEAD 被重新训练。

| 共同运行控制 | 本次实际值 |
|---|---|
| 硬件 / mesh | 两节点、每节点8×H200；DP replicate2 × shard8（HSDP） |
| batch / context | local16、GA1、global256条序列、4096ctx；1,048,576 positions/update |
| 优化 | AdamW，peak LR3e-4，betas(.9,.95)、eps1e-8、WD.1、clip1 |
| 训练算术 | BF16 compute、FP32 masters/reduce、native grouped experts、FlexAttention、head chunks8 |
| 目标 | full BPTT、terminal CE，balance coeff.01，aux=`local_micro_call_mean` |
| WSD | warmup20更新；stable end10000；cosine decay9537更新至峰值LR的.1；完整 horizon19537 |
| 其他 | finite epoch0、shuffle、num_workers0；无 AC/compile/CUDA graphs；TP/EP/CP/PP=1 |

norm、embedding、lm_head、router 的 WD 为0；两臂 expert LR 均不另缩放。1e-4 是未使用的稳定性 fallback，不是一条额外实验。local batch/GA 同样属于比较控制，因为 aux 对 microbatch 几何并非线性。

所有 rank 需能读同一份 prepared 数据与运行目录。两台已分配节点分别在 checkout 根执行以下命令，只将 `NODE_RANK` 设为0或1；`MASTER_ADDR`、`MASTER_PORT` 使用自己的 rendezvous 地址。另一臂将 `P1_SPEC` 改为 `recipes/p1_smelt.json`，使用独立的 `P1_RUN`。首次完整训练前执行 `unset MOELOOP_ANCHOR_STOP MOELOOP_ANCHOR_RESUME`；短训和接续时按下文设置它们，torchrun 命令不变。

```sh
export P1_SPEC=recipes/p1_moe.json
export P1_RUN=/runs/p1-moe
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export MOELOOP_ANCHOR_SPEC="$P1_SPEC"
export MOELOOP_ANCHOR_MANIFEST="$P1_DATA/manifest.json"
export MOELOOP_ANCHOR_OUTPUT="$P1_RUN"
torchrun --nnodes=2 --nproc-per-node=8 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  -m moeloop.train --module recipe_anchor --config anchor \
  --training.local-batch-size 16 \
  --parallelism.data-parallel-replicate-degree 2 \
  --parallelism.data-parallel-shard-degree 8 --comm.init-timeout-seconds 1800
```

先做短程验证时，在启动前设 `MOELOOP_ANCHOR_STOP=8`；它只设置本次绝对停止坐标，保留 JSON 内的完整19537 WSD。完成后按下节保护 full DCP，再移除 stop 或改为较大的诊断坐标接续，两臂保留相同控制，不追加 seeds。每477更新执行一次固定 validation/checkpoint；原生约1M-label validation 实际每点为1,048,576 labels，停点会增加终点评价，均与完整10M终点评测分开记录。

仅探索新形状时可以使用 `recipe_anchor.py --attention-pairs-per-sequence 4180216.9229844734 --output /runs/new-pair`，它不启动训练，也不自动生成本次冻结的 source/runtime/data 身份。该 pairs 数来自完整20.486B训练 prefix 的实际文档 packing；预算定义与复算命令见[成本与比较](comparison.md#p1-与文档内-attention-预算)。

## 继续训练与结果事实

native checkpoint 目录为 `checkpoint/step-N`，最终保存完整训练状态。直接使用 `moeloop.train` 时，相同输出目录再次启动会加载 latest checkpoint；也可显式设置 `--checkpoint.initial-load-path /runs/original/checkpoint/step-N`。上文 runner 始终要求显式 `--resume`。保持模型、source code、数据/tokenizer、world size、workers、optimizer/loss/AC 与 schedule horizon 不变；目标 stop step 可以延长。

P1 入口保留最近两个 rolling checkpoints。要将早期 checkpoint 作为可审阅的恢复起点，**先在所有训练进程退出后完整复制到 rolling 目录之外**，再启动下一段；例如在共享存储上执行一次：

```sh
mkdir -p "$P1_RUN/continuation"
test ! -e "$P1_RUN/continuation/step-8"
cp -a "$P1_RUN/checkpoint/step-8" "$P1_RUN/continuation/step-8"
export MOELOOP_ANCHOR_RESUME="$P1_RUN/continuation/step-8"
unset MOELOOP_ANCHOR_STOP
```

在两节点使用同一个保护路径并重跑上面的 torchrun 命令，接续时保留 `MOELOOP_ANCHOR_RESUME`。完整 DCP 包括 metadata 及所有 rank shards；只拷贝模型或某个 rank 不是 full-state 恢复。公开重放生成自己的 process/路径 lineage，原运行的 source、恢复证明和结果身份保留在证据中。

首版严格继续要求 `num_workers=0` 和相同拓扑：逻辑数据流本身与卡数无关，但保存的数据状态记录 rank、world size 和每 rank block 数，换卡数会被拒绝。保存每 rank RNG、数据游标/部分 buffer、source/mixture 状态、已完成 updates 和实际 tokens。DataLoader 的独立 generator、validation RNG 隔离保证继续过程不额外消耗训练随机序列。换数据阶段、换拓扑或仅模型 warm start 应作为新 lineage 明确执行；不使用 `strict=False` 隐藏不一致。

CE、aux、进度和路由计数按每次成功update记录；随机深度另记每microbatch的实际r/梯度轮数，expected CE另记训练expected CE、entropy与exit分布。每次训练 validation 向输出目录的 `validation.jsonl` 追加一条记录，除 CE 外带 `label_bytes`、`ce_sum_text`、`bpb`、`byte_table_sha256` 和 `bpb_unavailable`，口径与 `moeloop.evaluate` 相同（见下文“导出、评测和生成”）。分子分母都按 rank 求和，所以 BPB 不依赖卡数。BPB 同时以 `validation_metrics/bpb` 记入原生指标，与 TorchTitan 自带的 `validation_metrics/loss` 同组，JSONL 里的字段名仍是 `bpb`；没有字节表时这四个字节字段为 None，`bpb_unavailable` 写明原因，也不记该指标。实现见 [`titan.py`](../src/moeloop/titan.py) 的 `LoopValidator.validate`。

梯度范数、原生吞吐和显存峰值只在TorchTitan实际日志采样步出现；`metrics.log_freq > 1`时，其余行省略未观测字段。P1使用`log_freq=1`。

输出目录包含 model/resolved config、continuation identity、metrics.jsonl、outcome.json 和每次 attempt 的小型事实。successful updates 与原生提前自增的 step 分开；input tokens、有效监督 labels、padding positions 和递归工作量分别计数。数据耗尽或 checkpoint 未落盘不会标为完成。

相对 manifest、tokenizer 和输出路径按训练启动目录解释，训练入口在元数据中保存绝对路径；manifest 也可输入包含 `manifest.json` 的目录。之后可以从其他工作目录导出。自动发现和身份校验仍需要原 prepared 数据与 tokenizer 文件可访问。

## 导出、评测和生成

以下接续上面的8→16步示例；其他运行替换为实际持久保存的 step 和对应 context。导出目录和评测结果文件均使用新路径。

```sh
python -m moeloop.export --checkpoint "$RUN/checkpoint/step-16" \
  --tokenizer "$MOELOOP_DATA/tokenizer" --output "$RUN/hf-step16"
python -m moeloop.evaluate --model "$RUN/hf-step16" \
  --manifest "$MOELOOP_DATA/manifest.json" --seq-len 2048 --device cuda \
  --output "$RUN/heldout-step16.json"
python -m moeloop.generate --model "$RUN/hf-step16" --device cuda \
  --prompt 'The model' --max-new-tokens 32
```

export 仅读取 DCP 的模型张量，不分配 optimizer；在 meta 上构造 owner 后严格载入。保存物理参数一次，附带 config、tokenizer、generation config、相对导入源码、LICENSE 与 hash/provenance。导出目录必须新建。模型配置和 tokenizer 可从训练元数据发现，也可显式指定；错误 tokenizer 会被拒绝。

单一 source 的 heldout 可追加 `--document-loss-output /results/documents.json`，在同一次评测保存按 split 文档序号对齐的 CE sum/count，并合并跨 block 的同一文档。输出包含数据/上下文/读出身份与完整性标志；聚合结果记录其路径和 SHA。

结果同时给出 CE 和字节精确的 BPB，`bpb = ce_sum_text / (ln 2 · label_bytes)`。字节表来自 manifest 的 tokenizer 资产（[`tokens.py`](../src/moeloop/tokens.py)）；字节数为 0 的 label（EOS、PAD 和所有特殊/新增 token）同时不进 `ce_sum_text` 和 `label_bytes`，与 nanochat 的 `evaluate_bpb` 一致，把 EOS 的 CE 留在分子里会得到更大的数。`byte_table_sha256` 标识字节表。manifest 没有 tokenizer 资产或 tokenizer 不是 byte-level BPE 时，字节字段为 None，原因写在 `bpb_unavailable`；有 NFC 规范化的 tokenizer（如 Qwen3）按规范化后的文本计字节。精确定义见 `evaluate_heldout` 的 docstring。

CORE（DCLM v2 口径）对导出的 HF 目录运行。评测包由使用者手动下载并核对哈希，评测器不下载：

```sh
python -m moeloop.core --bundle-info   # 官方 URL、哈希与下载步骤
python -m moeloop.core --model "$RUN/hf-step16" --bundle eval_bundle \
  --device cuda --output "$RUN/core-step16.json"
```

结果的 `core` 是 22 个任务 centered accuracy 的均值，只有跑满全部任务且不设 `--max-per-task` 时才有值，否则为 None；`core_v1` 用旧版 random baseline，用于对照 nanochat 与 2025 年 9 月前的 DCLM 数字。提示词、打分规则和已知限制见 [`core.py`](../src/moeloop/core.py) 的 docstring。

逐轮诊断对同一个导出目录和 prepared split 运行，输出每轮的 logit lens、残差更新、路由负载熵与 visit 间重叠，以及截断到 1..K 轮的损失：

```sh
python -m moeloop.diagnostics --model "$RUN/hf-step16" \
  --manifest "$MOELOOP_DATA/manifest.json" --seq-len 2048 --device cuda \
  --output "$RUN/diagnostics-step16.json" --plot "$RUN/diagnostics-step16.svg"
```

`--plot` 需要 `report` extra（matplotlib）。visit 从 0 计；logit lens 与残差需要整轮状态，即 `order="stack"` 或 core 只有一个 Cell，否则这两项跳过并写明原因。BPB 的口径同 `evaluate`。各量的定义见 [`diagnostics.py`](../src/moeloop/diagnostics.py) 的 docstring。

P1 每臂分别导出真正的 `step-19537`，用 FP32 HF 模型完整评价 validation/test，不传 `--max-batches`：

```sh
python -m moeloop.export --checkpoint "$P1_RUN/checkpoint/step-19537" \
  --tokenizer "$P1_DATA/tokenizer" --output "$P1_RUN/hf"
for split in validation test; do
  python -m moeloop.evaluate --model "$P1_RUN/hf" \
    --manifest "$P1_DATA/manifest.json" --split "$split" \
    --seq-len 4096 --batch-size 4 --device cuda:0 \
    --output "$P1_RUN/full-$split.json" \
    --document-loss-output "$P1_RUN/full-$split-documents.json"
done
```

P1 证据里的终点是 CE，当时的评测还没有 BPB 与 CORE；P1/P2 test 的 BPB 与 CORE 补测见[证据](evidence/p1p2-test-bpb-core.json)。上面的命令不变；输出含 BPB 的条件是 manifest 带 tokenizer 文件，否则 `bpb_unavailable` 写明原因。完整 validation/test 实际各10,001,497 /10,009,889 labels；两臂文档顺序及各文档 label counts 必须相同。配对文档差便于理解 heldout 差异，不估计训练 seed 方差。原生BF16固定验证曲线与该完整FP32终点评估分别报告。

HF Cache 类型停在 wrapper；core 的 KV 按逻辑 occurrence 独立，即使参数共享。一个 cache session 固定深度，不能中途新增缺少历史 KV 的 visit。公开模型 eval full-prefix 可正常求梯度，自动 cache 只在无梯度推理中启用。

Huginn-inspired与LOOM首条路径使用固定深度、`use_cache=False`，每次forward重建初态/memory；Huginn评测采用确定性零latent，属于声明的eval adaptation。Ouro训练过的gate随模型导出，HF生成仍使用terminal logits；HF的`labels`按标准next-token shift计算terminal CE，不计算Stage I expected CE或entropy。新增路径的独立消费者资格仍以验证记录为准。

HF消费端支持标准`model.resize_token_embeddings(len(tokenizer), mean_resizing=False)`；可传`pad_to_multiple_of`。实际词表大小同时更新HF、core与保存配置，原embedding/head行保持，新增行按`init_std`初始化，两个参数owner继续独立。随后将模型和tokenizer一起保存即可离线重载。更换tokenizer后的原生预训练应重建prepared manifest并建立新运行身份。

下游评测复用现有 lm-evaluation-harness：

```sh
uv pip install -e '.[eval]'
python -m lm_eval --model hf \
  --model_args pretrained=/models/experiment,trust_remote_code=True,add_bos_token=True \
  --tasks arc_easy --num_fewshot 0 --device cuda --batch_size 1
```

真实结果应固定任务版本、完整评测范围和 checkpoint 身份。`--limit 2` 只适合工程连通性检查。heldout CE 使用本项目文档/packing 协议，不能直接拿来对比另一套 tokenizer/context 处理的 PPL；BPB 按字节归一，去掉了 tokenizer 切分的差异，但 context、packing 与 mask 仍是本协议。
