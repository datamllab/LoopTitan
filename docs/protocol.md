# LoopTitan 实验协议

本文件描述可选的 `moeloop_lab.protocol` 与 CVSM 协作流程。独立使用只需普通 recipe JSON、prepared manifest 和输出目录，命令见[训练与数据](training.md#从-json-运行与恢复)，无需 GitHub 或 MLflow。当前项目协作采用敏捷方式，具体以 `AGENTS.md` 的 Project 和对应 issue 为准。

模型由 TorchTitan 的原生 Trainer 执行；选择 CVSM 时，本库把运行实际使用的配置、身份和有限产物接到已有记录端，不另造训练循环、调度器或任务数据库。

## 状态与决定

| 对象 | 唯一记录位置 | 完成条件 |
| --- | --- | --- |
| 研究问题与比较规格 | GitHub `study`，字段为 Question / Arms / Controlled / Adaptable / Criteria / Budget / Context | 人明确接受结论后才关闭 |
| 工程计划 | GitHub `plan` 和原生 sub-issues / dependencies | 子任务结束后等待分析或人决定 |
| 代码变更 | `code` issue、独立 worktree、PR | guard/test 通过且当前 SHA 独立 review 通过，再按 Merge 字段合并 |
| 一组实验 | `run` issue | 每条命令都有记录，贴出完整结果和规格核对后关闭 |
| 一次尝试 | 使用 CVSM 时对应一个原生 MLflow run | 记录真实终态；失败、被停止和重试都保留 |

保留 CVSM 的 publish / merge / budget / conclude 四个 gate。人已明确授权的范围不重复询问；代码通过检查不等于研究结论成立。MLflow 只保存运行事实，任务状态始终回查 GitHub。

## 从 recipe 筛选到预训练

筛选、确认和长程预训练使用同一套状态机，以不同 study 的 Criteria 和 Budget 表达，不新增阶段状态表。派发新实验前，在 Question / Criteria 写清预期能观察到的主要信号，以及正、负结果各会改变什么重要决定。没有足够信号时收窄结论、停止追加投入；强信号再确认稳定性、解释和迁移范围。

MoE / Loop 的比较必须先选清预算口径。固定输入 token 不代表固定 FLOPs；独立存储参数、激活参数和逻辑 block 调用也不是同一个量。Arms 只放有意改变的设置；Controlled 明确数据/顺序、tokenizer、模型/初始化、优化器/调度、有效 batch、训练长度、精度、调参预算、评测/选 checkpoint 和 seed，并逐项给理由。工程调整仅在 Adaptable 写明的规则内进行，不因 OOM 悄悄缩短训练或改变比较。

已完成的 P1 使用两份冻结的[实际 JSON](training.md#p1-实际配方与双机训练)；V2 方法资格，以及 V3 的 P2、正式 C 面板、完整 downstream 和已安装 wheel 旅程，均已有实际验收，范围与来源见[验证记录](validation.md)。`scripts/smoke.py --print-spec` 只输出 CPU 工程检查规格，其 `record_kind=engineering-fixture` 和 `fixture.completed` 不能用作模型结果。

## 运行记录接口

保留 `<run> --issue N [options] -- <cmd>` 和 `<results> --issue N [N ...] [--metric ...] [--spec FILE]`。命令见 AGENTS.md；`--spec -` 读 study 正文的标准输入，输出仍以 `规格核对：` 结束。

`capture_trainer()` 在 `config.build()` 后、`trainer.train()` 前观察原生对象，并在全局 rank 0 原子写入 `$RUN_DIR/config.json`。CVSM launcher 随后记录不可变的 `cfg.*` params 和原始 artifact，训练进程不必安装 MLflow。

| 配置键前缀 | 来源与含义 |
| --- | --- |
| `native.*` | 实际 `trainer.config.to_dict()` 的全部字段，不手工重建或过滤 |
| `derived.*` | 实际 world size、batch mesh degree、梯度累积、有效 batch、每步/目标输入位置数 |
| `assets.*` | 数据 manifest、tokenizer 文件和可选 evaluation manifest 的内容 SHA-256 |
| `sources.*` | 实际导入的 TorchTitan 和模型模块所在 Git checkout 的 SHA；要求干净，TorchTitan 必须匹配受保护的 pin |
| `environment.*` | 当前 Python、训练包版本、Torch Git/CUDA 版本；无法观察的值为 null |

例如 study 应固定 `derived.global_batch_size` 和 `derived.target_input_tokens`。只有保持这两项且比较允许改变 aux 的 microbatch 几何时，才可按明确规则调整 `native.training.local_batch_size` 与 `derived.gradient_accumulation_steps`。每个正式尺度/面板使用同一个事前固定 seed（42）。命令中使用实际生效的 `--debug.seed`，CVSM 记录为 `native.debug.seed`。

完整 native 配置会保留输出目录等变化。确实与结果无关的路径，逐个以其真实键写入 Adaptable，并给规则，例如 `native.dump_folder  # 每次使用新的独立输出目录`。数据路径只有在相同 manifest/tokenizer 身份下才可适配；checkpoint 加载路径不能被当作普通输出路径消去。

`native-config.json` 保留同一原生配置；`metrics.json` 是 rank 0 写一次的有限数值快照；`log.txt` 来自 launcher。`derived.target_input_tokens` 是配置目标，MLflow `FINISHED` 是命令退出 0，原生 step 字段是状态计数；三者都不证明实际完成的优化步或 token。checkpoint、原始数据和大量日志放在显式外部存储，不能写入会被整体上传的 RUN_DIR。`uv.lock` 冻结控制环境，不能替代训练环境锁；运行环境观察同样不是完整可重建环境证明。

manifest 哈希识别 manifest 本身，不代表重新核验了所有数据 shard；模型模块 SHA 不自动证明所有动态插件的身份。这些条件必须由具体数据/模型入口验收并在 study 中说明。

## 中断与恢复的边界

原生 TorchTitan 在 `train()` 内调用 checkpointer.load，初始化完成时的 `trainer.step` 不能作为恢复成功证据。实际进度由成功 optimizer updates、全 rank 数据/RNG 状态与 durable full DCP 核对；`--stop` 只改变本次绝对停止坐标，保留 recipe 的完整 schedule。

P1 的双机 HSDP16 已完成连续8步与4+恢复4步的模型、optimizer 和各 rank 状态比较，V2 新方法也已完成各自的完整状态恢复，见[P1 证据](evidence/p1.json)、[V2 证据](evidence/recipes-v2.json)。这些验收限于记录的配置；严格继续要求相同源码、数据/tokenizer、拓扑、optimizer/loss 与 schedule，以及 `num_workers=0`，不承诺任意 world size 恢复或 EP/TP。

恢复前记录父尝试、checkpoint 身份、原 schedule、拓扑与恢复检查。`recipe_experiments.py` 要求显式 `--resume`，启动前把完整 DCP 保护到 rolling 目录之外。使用 CVSM 时，每次独立提交（包括失败后的重试）建立新的 MLflow run；一次提交内预定的分段保存/恢复保留各自的进程收据，不覆盖原记录。

`scripts/recipe_experiments.py` 在已经分配的 GPU 上运行，消费标准 torchrun 多节点环境；每个并行 agent/worktree 使用独立输出根，多节点的不同实验也使用独立 rendezvous。`scripts/run.py` 仍是可选的单机、前台命令记录端；`gpus=[]` 只关闭本地 GPU 租约。资源分配、持久作业监督、远端 GPU 小时计量和 walltime 上限由使用者的运行环境负责。
