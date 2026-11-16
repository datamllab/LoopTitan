# LoopTitan experiment protocol

This file describes the optional `moeloop_lab.protocol` and the CVSM collaboration flow. Standalone use needs only a plain recipe JSON, a prepared manifest and an output directory. It does not need GitHub or MLflow. For the commands, see the section on running and resuming from JSON in [Training and data](training.md). The project currently collaborates in an agile way. The Project section of `AGENTS.md` and the matching issue are the authority.

TorchTitan's native Trainer runs the model. When you select CVSM, this library connects the configuration, the identities and the finite artifacts that the run actually used to the existing recorder. It does not make its own training loop, scheduler or task database.

## State and decisions

| Object | Only record location | Completion condition |
| --- | --- | --- |
| Research question and comparison spec | GitHub `study`, with the fields Question / Arms / Controlled / Adaptable / Criteria / Budget / Context | Closed only after a human explicitly accepts the conclusion |
| Engineering plan | GitHub `plan` with native sub-issues / dependencies | When its sub-tasks end, it waits for analysis or a human decision |
| Code change | `code` issue, separate worktree, PR | guard/test pass and an independent review passes for the current SHA; then it is merged as the Merge field says |
| A set of experiments | `run` issue | Closed when every command has a record and the complete results and the spec check are posted |
| One attempt | With CVSM, one native MLflow run | Records the real final state; failures, stops and retries are all kept |

The four CVSM gates (publish / merge / budget / conclude) are kept. Scope that a human has explicitly authorized is not asked again. Code that passes its checks does not mean that a research conclusion holds. MLflow keeps only run facts. Always read task state back from GitHub.

## From recipe screen to pretraining

Screening, confirmation and long pretraining use the same state machine. Different studies express them through their Criteria and Budget. There is no new table of stage states. Before you dispatch a new experiment, write in Question / Criteria the main signal that you expect to observe. Also write which important decision a positive result and a negative result would each change. If the signal is not sufficient, narrow the conclusion and stop further investment. If the signal is strong, then confirm its stability, its explanation and its range of transfer.

A MoE / Loop comparison must first choose its budget definition. A fixed number of input tokens does not mean fixed FLOPs. Separately stored parameters, active parameters and logical block calls are also not the same quantity. Arms contain only the settings that change on purpose. Controlled states the data and order, tokenizer, model and initialization, optimizer and schedule, effective batch, training length, precision, tuning budget, evaluation and checkpoint selection, and seed, each with a reason. Engineering adjustments happen only within the rules written in Adaptable. Do not silently shorten training or change the comparison because of an OOM.

The completed P1 uses two frozen [actual JSON files](training.md) (see the section on the actual P1 recipes and two-node training). V2 method qualification, and for V3 the P2, the formal C panel, the complete downstream evaluation and the installed-wheel journey, all have actual acceptance. For their scope and sources, see the [validation record](validation.md). `scripts/smoke.py --print-spec` prints only a CPU engineering check spec. Its `record_kind=engineering-fixture` and `fixture.completed` must not be used as model results.

## Run record interface

The interfaces `<run> --issue N [options] -- <cmd>` and `<results> --issue N [N ...] [--metric ...] [--spec FILE]` are kept. For the commands, see AGENTS.md. `--spec -` reads the study body from standard input. The output still ends with the spec-check section.

`capture_trainer()` observes the native objects after `config.build()` and before `trainer.train()`. On global rank 0, it writes `$RUN_DIR/config.json` atomically. The CVSM launcher then records immutable `cfg.*` params and the raw artifact. The training process does not need to install MLflow.

| Configuration key prefix | Source and meaning |
| --- | --- |
| `native.*` | All fields of the actual `trainer.config.to_dict()`, not rebuilt or filtered by hand |
| `derived.*` | The actual world size, batch mesh degree, gradient accumulation, effective batch, and input positions per step and in total (target) |
| `assets.*` | Content SHA-256 of the data manifest, the tokenizer files and the optional evaluation manifest |
| `sources.*` | SHA of the Git checkouts that hold the actually imported TorchTitan and model modules. They must be clean, and TorchTitan must match the protected pin. |
| `environment.*` | The current Python, training package versions, and Torch Git/CUDA versions; a value that cannot be observed is null |

For example, a study should fix `derived.global_batch_size` and `derived.target_input_tokens`. Only when these two stay fixed, and the comparison allows the aux microbatch geometry to change, can `native.training.local_batch_size` and `derived.gradient_accumulation_steps` change by an explicit rule. Each formal scale or panel uses the same seed (42), fixed in advance. The command uses the effective `--debug.seed`, and CVSM records it as `native.debug.seed`.

The complete native configuration keeps values that change, such as the output directory. Write each path that truly does not affect results into Adaptable, one by one with its real key, and give a rule. For example: `native.dump_folder  # a new separate output directory for each run`. A data path can adapt only under the same manifest/tokenizer identity. A checkpoint load path must not be removed as if it were a plain output path.

`native-config.json` keeps the same native configuration. `metrics.json` is a snapshot of finite numbers that rank 0 writes once. `log.txt` comes from the launcher. `derived.target_input_tokens` is the configured target. MLflow `FINISHED` means that the command exited with 0. The native step field is a state counter. None of the three proves the optimizer steps or tokens that were actually completed. Checkpoints, raw data and large logs go to explicit external storage. Do not write them into RUN_DIR, because RUN_DIR is uploaded as a whole. `uv.lock` freezes the control environment. It cannot replace a lock of the training environment. The observation of the run environment is also not a complete proof that the environment can be rebuilt.

A manifest hash identifies the manifest itself. It does not mean that all data shards were verified again. The SHA of a model module does not automatically prove the identity of all dynamic plugins. The specific data and model entry points must verify these conditions, and the study must state them.

## Limits of interruption and resume

Native TorchTitan calls checkpointer.load inside `train()`. Thus `trainer.step` at the end of initialization is not evidence of a successful resume. The actual progress is checked against successful optimizer updates, the data and RNG states of all ranks, and a durable full DCP. `--stop` changes only the absolute stop coordinate of this run. It keeps the complete schedule of the recipe.

For P1, two-node HSDP16 compared the model, the optimizer and the state of each rank between 8 continuous steps and 4 steps plus 4 resumed steps. The new V2 methods also completed their own full-state resume. See the [P1 evidence](evidence/p1.json) and the [V2 evidence](evidence/recipes-v2.json). These acceptances are limited to the recorded configurations. A strict continuation requires the same source, data/tokenizer, topology, optimizer/loss and schedule, and `num_workers=0`. Resume at any world size, and EP/TP, are not promised.

Before a resume, record the parent attempt, the checkpoint identity, the original schedule, the topology and the resume checks. `recipe_experiments.py` requires an explicit `--resume`. Before the start, it protects the complete DCP outside the rolling directory. With CVSM, each separate submission (a retry after a failure included) creates a new MLflow run. Planned segment saves and resumes inside one submission keep their own process receipts and do not overwrite the original record.

`scripts/recipe_experiments.py` runs on GPUs that are already allocated, and it uses the standard torchrun multi-node environment. Each parallel agent or worktree uses its own output root. Different multi-node experiments also use their own rendezvous. `scripts/run.py` is still the optional single-machine, foreground command recorder. `gpus=[]` only turns off the local GPU lease. Resource allocation, supervision of persistent jobs, remote GPU-hour accounting and walltime limits are the job of the user's run environment.

## Terms in earlier threads

Most issues, pull requests and comments before plan #274 are in Chinese. The keywords and headings that the tools parse (`### Goal`, `### Arms`, `## Changes`, `Closes #N`, `Verdict:`, `Decision:`, `Blocked:`) were English then and did not change. The table maps the other earlier terms to the English ones that the tools and skills use now.

| Earlier Chinese | English |
| --- | --- |
| 代码任务 / 运行任务 / 研究 (form names) | Code task / Run task / Study |
| `#N 的结果：n 个 run，k 组，按去掉 seed 的命令分组。` | `Results of #N: n runs in k groups, grouped by command with the seed removed.` |
| 命令 / 失败数 / 均值 / 标准差 / 环境锁 / 数据 / 无 | command / n_failed / mean / std / env lock / data manifest / none |
| 标注 | Flags |
| commit 不同 / 环境不同 / 数据不同 | Commit differs / Env lock differs / Data manifest differs |
| 缺环境锁文件 / 工作区不干净 / 成功退出但没有指标 | with missing env lock files / with a dirty worktree / with exit code 0 and no metrics |
| 未完成（还在跑或已被杀）/ 已在 MLflow 中删除，但仍计入上表 | not finished (still running or killed) / deleted in MLflow but still counted in the table |
| 规格核对 | Spec check |
| 没有偏差 / 没有给出规格 | No deviations / No spec given |
| 不成立 / 未记录 | does not hold / not recorded |
| 不属于任何 arm / 属于多个 arm / 没有 run 的 arm | with no matching arm / in more than one arm / Arms with no run |
| arm X 缺 … 的取值 | Arm X is missing values of … |
| 未声明的差异 | Undeclared difference |
| 读不懂的规格行 / Controlled 行没有写理由 / 规格里没有 … 一节 / 出现在多节 | Unreadable spec line / Controlled line without a reason / No … section in the spec / is in more than one section |
| 等人合并 / 不合并 / 已合并 / 已按人的指示合并 | waiting for a human to merge / not merged / merged / merged on the human's instruction |
| `Decision: 发布 …` / `Decision: 合并 PR …` | `Decision: publish …` / `Decision: merge PR …` |
| `Blocked: 这个对比按规格做不了，因为 <X>` | `Blocked: this comparison cannot be done as specified, because <X>` |
| 剩余的 minor 意见与处理建议；建议：开新 issue / 并进 #M / 不处理；理由 | Remaining minor findings and proposals; proposal: new issue / add to #M / no action; reason |
