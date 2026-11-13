# 实现与贡献

按[架构](architecture.md)开发。多个开发者或 subagents 按不重叠文件分工，由集成者审阅。局部接口、测试和实现细节自主迭代；研究问题、方法交付、资源和正式比较条件的实质变化单独说明。CVSM 服务正式对照实验。

核心保持普通函数、模块和显式循环。已有构件的新组合增加 recipe；局部新算子增加小模块；真正不同的状态更新保留专用短 forward。新 forward 改变参数 owner 或 cache 发生次数时，补有限的 runtime/export 接入。不要为了目录扁平把全部职责塞入巨文件，也不要为未知需求预建 GraphExecutor/Policy/插件平台。

新方法先提供可独立导入的 `*_spec()`，写清来源、原文保留与适配；用 `compare_recipes.py inspect/diff/compare` 检查实际配置。九个命名方法的边界见[配方](recipes.md)，成本估算和正式比较问题见[比较](comparison.md)；写完 `*_spec()` 先跑[一致性测试](#一致性测试与诊断)。`loom/injected` 的旧组件别名不能当成完整方法贡献。

## 开发环境和检查

控制协议环境：

```sh
uv sync --frozen
uv run --frozen python -m pytest -q
```

模型环境按 README 安装 CPU Torch 与 HF/data extras 即可验证纯模型、数据和独立消费者；原生 adapter 检查再安装 train extra 与该 pin 需要的 Triton：

```sh
uv pip install -e '.[data,hf,train,report]' 'triton==3.8.0' 'pytest==8.3.3'
python -m pytest -q tests/model tests/data tests/training tests/hf tests/delivery tests/reference tests/qualification tests/comparison tests/setup
python scripts/compare_baselines.py --output /tmp/qwen3-reference.json
git diff --check
```

CI 分开验证 Torch-free 控制接口和完整 CPU 模型/原生接入；依赖 Torch 的测试在文件顶部用 `pytest.importorskip("torch")`，控制环境跳过它们，由模型环境的 job 覆盖。测试应验证实际语义：shared gradient 等于各次调用之和、不同循环顺序确实不同、no-grad/anchor 路径、packing/cursor、官方 baseline、真实 FSDP/AC/chunked loss、独立 HF 进程。避免逐行复制实现来增加测试数量。

## 一致性测试与诊断

[`moeloop.testing.check_recipe`](../src/moeloop/testing.py) 是给任何 recipe 的一致性测试套件，用小几何实例化后在 CPU 上运行：

```python
from moeloop.recipes import smelt_spec
from moeloop.testing import check_recipe

spec = smelt_spec(vocab_size=128, dim=32, hidden_dim=64, num_heads=4,
                  num_kv_heads=2, num_experts=4, max_seq_len=64)
report = check_recipe(spec)  # 有检查失败时抛 ConformanceError，其 report 列出每项结果
print({name: check["status"] for name, check in report["checks"].items()})
```

六项检查是 `ownership`（参数 owner 唯一、不共享存储）、`shared_gradients`（共享梯度等于各次调用之和）、`cost_calls`（成本模型的调用数和参数数等于一次真实前向）、`cache`（分块 cache 解码等于完整重算，KV 槽位和字节等于成本模型；不用 cache 的 recipe 为 not_applicable）、`hf_roundtrip`（`save_pretrained`/`from_pretrained` 后 logits 相同）和 `identity`（从 JSON 重建的 spec 保持 continuation identity 和初始权重）。它复用 `tests/model` 与 `tests/hf` 的 oracle，检查的是 `LoopedLM(spec.model)` 在 float32 CPU 参考路径上的契约，不说明质量、训练稳定性、原生 GPU kernel 或导出命令行；自写的 forward（见[扩展](extensions.md)）也不在其中。

训练出模型之后，[`moeloop.diagnostics`](../src/moeloop/diagnostics.py) 读逐轮行为（logit lens、残差更新、路由重叠、有效深度），命令见[训练文档](training.md#导出评测和生成)；Looped MoE Bench 的提交用 `moeloop.bench.validate` 自查，见[比较](comparison.md#looped-moe-bench)。诊断用来理解模型，不构成方法资格。

## GPU 开发与调试

GPU 训练只消费已授权、已分配的资源。部署地址、节点和凭证不进入公共 recipe。

在同一可用环境中连续调试，复用数据/cache；调通后固定实际源码身份，以干净进程运行资格。每次 attempt 保留源码 hash、配置、命令、runtime 和原始结果，失败不覆盖。把 durable checkpoint、tokenizer/data manifest、HF 和 evidence 到已核实持久目录；不把临时容器路径当成交接成果。

## 方法资格与工程 probes

四方法的真实文本资格入口消费给定 manifest 和可见 GPU，不管理集群：

```sh
export MOELOOP_DATA=/data/dclm-qwen3-v1
export MOELOOP_EVIDENCE=/runs/v1-evidence
python scripts/recipe_qualification.py --stage v1 \
  --manifest "$MOELOOP_DATA/manifest.json" --output "$MOELOOP_EVIDENCE" --gpus 2
python scripts/check_release.py --stage v1 --evidence "$MOELOOP_EVIDENCE" --engineering-only
```

上述命令只创建并核对四方法工程资格，`--engineering-only` 不代表完整 V1 验收。完整 V1 验收还需要实际 P1、公开性能和8/16卡恢复等原始资产组成完整 evidence；对该完整目录运行 `python scripts/check_release.py --stage v1 --evidence "$MOELOOP_FULL_EVIDENCE"`，不加 engineering-only。已完成范围和原始资产/公开投影的区别见[验证记录](validation.md)。

默认目标为四方法各512 successful updates、context2048、global batch16×2048 positions，并检查 native FSDP、完整恢复与实际 checkpoint 的 HF 独立消费。`--pilot` 是有界开发检查，不能计为方法资格。检查器读取真实产物，缺项或不匹配身份必须失败；命令存在不表示运行已完成，结果见[验证记录](validation.md)。

旧的小型 batch 用于工程接缝回归。`scripts/qualification.py` 不分配资源，只准备离线工程 fixture、核对 durable full checkpoint 和比较 continuation：

```sh
export PYTHONPATH="$PWD/src:$PWD/scripts:$PYTHONPATH"
export MOELOOP_DATA=/runs/qualification/data
python scripts/qualification.py prepare --output "$MOELOOP_DATA"
export MOELOOP_OUTPUT=/runs/qualification/shared
export MOELOOP_QUAL_VARIANT=shared MOELOOP_STEPS=4
torchrun --standalone --nproc-per-node=4 -m moeloop.train \
  --module qualification_recipes --config smoke
python scripts/qualification.py verify --directory "$MOELOOP_OUTPUT" --steps 4
```

`shared`、`partial_attention`、`partial_router`、`expert_tying`、`injected`、`dense` 的 2–4 updates 只验证已有组件，不替代四方法真实文本短训。比较连续 4 步与 stop2/resume4 时保持 fixture、source snapshot、topology、horizon16 和 recipe 一致，仅 stop target/明确 checkpoint 加载位置不同：

```sh
python scripts/qualification.py compare --uninterrupted /runs/qualification/full4 \
  --resumed /runs/qualification/resumed4 --output /results/continuation.json
```

模型张量使用明确数值容差；所有 ranks 的 RNG、cursor、schedule 和离散完成状态严格对应。还要执行实际 native checkpoint 的 export/heldout/cached generation。多机配置必须有真实多节点证据，不能从单机 green 推定。部署地址、节点、凭证和调度资源设置留在使用方，不进入公共 recipe。

原生 grouped experts/FlexAttention 的 BF16 数值对照使用[独立脚本](../scripts/compare_native.py)，需要已分配的 CUDA GPU 和固定 TorchTitan 环境：

```sh
python scripts/compare_native.py --output /results/native-kernel.json
```

脚本固定 seed、document/padding 输入、shared 与 injected suffix 配置，比较模型前向及全部物理 owner 含 balance/z 的梯度。精确路由按每次逻辑调用、每个有效 token 的 expert 集合核对，允许 top-k 返回顺序不同；汇总直方图不能替代此项。梯度必须同时满足张量/全局 relative L2 与 BF16 tensor-peak 的界限；具体阈值、首轮失败和重测的机器记录见[验证记录](validation.md)。保留失败事实与原始结果，修改验收策略后单独重跑，不覆盖旧状态。

## 对外承诺

用[验证记录](validation.md)描述具体通过的硬件、版本和支持组合。源码可读或某项 CPU 检查通过，不能证明所有运行组合、性能或科学效果。发现局部执行问题先保留算法并简化/修正实现；改变核心数学或公开支持范围时再讨论。

Muon 路径同理：分组规则和单 rank CPU 上真实的一步有测试（`tests/training/test_muon.py`），跨卡数一致的数据流在 1/2/4 个 CPU FSDP rank 上测过（`tests/training/test_topology.py`）；#49 在 4 卡与 8 卡 FSDP 上完成了 Muon 的训练、完整 checkpoint 保存与 WSD 分叉；HSDP `dp_replicate>1`、world 大于专家数的分片边界、空 token 专家和 Muon checkpoint 续训未在该次运行中覆盖。
