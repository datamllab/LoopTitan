# CVSM 来源与适配边界

本文保留初次协议导入的来源记录。当前公共模型与原生训练实现见[训练与数据](training.md)，实际验证状态见[验证记录](validation.md)。

复制自 [doem97/cvsm](https://github.com/doem97/cvsm/tree/b9cb290051f481b24c1abee332218efecfaaa02e)，完整 commit `b9cb290051f481b24c1abee332218efecfaaa02e`。保留根目录 Apache-2.0 [LICENSE](../LICENSE)；改动文件由 adoption commit 及后续 Git diff 标识。

本仓库复制 `AGENTS.md`、四个 `.agents/skills/`、Claude/Codex 入口、GitHub 表单/guard/merge/CI、`scripts/` 和 `tests/`。没有复制 CVSM 的宣传网站或 Pages workflow。

初次导入按 CVSM adoption guide 的一次性 bootstrap 规则直接建立 main，再执行 `scripts/setup.sh`；之后使用 Issue、分支和 PR。初次变化为 Project 字段、目录/保护范围、CPU 依赖锁和 CI；核心状态机、issue 标题字段、review verdict、merge 规则与结果评论的规格核对接口保持原样。

TorchTitan 使用 v0.3.0 源码基线 `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`，见 [来源锁](../configs/upstream.json)。这是兼容接口基线，本身不是 GPU、多节点或恢复验收；这些验收见[验证记录](validation.md)。

使用 MLflow HTTP 服务；客户端应显式设置 `MLFLOW_TRACKING_URI`，不直接打开或迁移服务的 SQLite 文件。

后续适配在 `scripts/run.toml` 加入 `require_tracking_uri=true`，由 run/results 共用同一检查；CVSM 的参考 SQLite fallback 仅对未启用该选项的配置保留，便于离线临时数据库测试。默认 experiment 改为 `moeloop-lab`，seed 选项改为原生 `debug.seed`。其余上游 run/results 记录与规格解析算法保留。

新增 `src/moeloop_lab/protocol.py`、只读 doctor、CPU smoke 和相应测试。接口与限制分别见[实验协议](protocol.md)、[训练接入](torchtitan.md)。
