# TorchTitan 实验记录接入（可选）

本文描述 `moeloop_lab.protocol` 的记录接口。公共 `moeloop` 模型与原生 Trainer 的使用方式见[训练与数据](training.md)，实际机器资格见[验证记录](validation.md)。

接口基于 [TorchTitan v0.3.0 Trainer 的固定源码](https://github.com/pytorch/torchtitan/blob/086bf6c166ec85c1298eb5596fa9bf95f6a2d840/torchtitan/trainer.py)，不是随 main 漂移。`Trainer.Config.to_dict()` 提供配置，`parallel_dims` 的 batch mesh 提供真正 data-parallel degree，`gradient_accumulation_steps` 提供 Trainer 已推导的累积次数。

## 在训练入口插入记录

在现有 GPU 环境单独以 editable 方式安装本库，不向训练环境安装控制侧的 MLflow 依赖。基础包只有标准库依赖；保留 Git checkout 和受保护的 `configs/upstream.json`。

```sh
/training/env/bin/python -m pip install --no-deps -e /path/to/moeloop-lab
```

模型入口负责构造原生 Trainer.Config、数据与模型本身。加入下面的调用，不复制 Trainer 或改变训练数学：

```python
from pathlib import Path
import actual_model_module  # replace with the imported model implementation
from moeloop_lab.protocol import capture_trainer, write_metrics

trainer = config.build()
try:
    capture_trainer(
        trainer,
        data_manifest=Path("/data/packed/manifest.json"),
        tokenizer_files={
            "tokenizer": Path("/assets/tokenizer.json"),
            "tokenizer_config": Path("/assets/tokenizer_config.json"),
        },
        model_module=actual_model_module,
        evaluation_manifest=None,  # supply the actual evaluation identity when enabled
    )
    trainer.train()
    write_metrics({"native.step_counter": trainer.step})
finally:
    trainer.close()
```

在每个 rank 调用；仅全局 `RANK=0` 读取来源/manifest 和写文件。按原训练入口清理分布式进程组，用 torchrun 将任一 rank 的错误传播给其余 worker。适配器不增加 barrier，也不接管模型/进程生命周期；记录失败必须让入口非零退出。

`native.step_counter` 只表示原生状态字段，不能认证已完成的优化步数：该版本在执行 train_step 前递增 step，捕获数据耗尽后会直接退出，因此最后一步未完成时计数仍可能等于目标。`FINISHED` 同样只表示命令退出 0。实际训练入口必须另行验证成功的优化步数、消费 token 和必要 checkpoint；本版配置适配器不提供这项完成认证。

由 CVSM 从干净的 run worktree 包裹**一次完整 allocation 的主命令**。控制环境使用 `uv run --frozen python scripts/run.py --issue N -- <cmd>`；`<cmd>` 必须显式选择已有训练环境的 python/torchrun，不能误用不含 Torch 的控制环境。`RUN_DIR`、`MLFLOW_RUN_ID`、`MLFLOW_TRACKING_URI` 会传给子进程。

首版仅验证本地命令记录接口；不能在每个远程节点分别包一层 launcher 并把多个 run 当成同一 allocation。多节点数据分发、网络、scheduler、GPU 预算和恢复须另行接入。

## batch 与 token 口径

```text
effective batch = local_batch_size × batch_mesh_degree × gradient_accumulation_steps
input positions per step = effective batch × seq_len
target input positions = input positions per step × original configured steps
```

例如 world size 32、batch mesh 16、local batch 1、累积 8、seq_len 4096 时，每步为 524288 个输入位置，而非按 32 个数据副本计算。native `global_batch_size=-1` 的自动模式只能对应一个累积步。与实际累积不一致的声明会拒绝记录，避免貌似相同的 recipe 隐含不同 token 预算。

这里计的是配置中的输入位置，不是非 padding token、独立样本数、唯一 token 或 FLOPs。物理参数、激活计算、吞吐和有效 loss token 需要各自的真实计量；不能从这些乘法推断。

## MLflow 记录端

```sh
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run --frozen python scripts/doctor.py
```

doctor 只读取 `/health` 并通过官方 SDK 有界查询最多一个 experiment，不创建记录、不迁移数据库、不重启服务。认证按 MLflow 标准环境变量提供，凭证不进 URI 或 Git。两个 CVSM 命令使用相同显式 endpoint；项目默认配置拒绝静默创建本地 SQLite。

使用 `MLFLOW_EXPERIMENT_NAME` 为独立 study 指定稳定命名空间。远程 worker 不能直接使用控制机的 `127.0.0.1`；本版不改变服务器监听和认证，也不承诺远程可达。模型只写小型 JSON 产物时无需访问 MLflow，上传由保持运行的控制侧 launcher 完成。
