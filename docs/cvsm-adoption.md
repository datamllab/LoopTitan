# CVSM source and adaptation boundary

This document keeps the source record of the first protocol import. For the current public model and native training implementation, see [Training and data](training.md). For the actual validation status, see the [validation record](validation.md).

The protocol is copied from [doem97/cvsm](https://github.com/doem97/cvsm/tree/b9cb290051f481b24c1abee332218efecfaaa02e), full commit `b9cb290051f481b24c1abee332218efecfaaa02e`. CVSM and this repository come from the same author. The imported part is released with this repository under the root MIT [LICENSE](../LICENSE). The adoption commit and later Git diffs identify the changed files.

This repository copies `AGENTS.md`, the four `.agents/skills/`, the Claude/Codex entry points, the GitHub forms/guard/merge/CI, `scripts/` and `tests/`. It does not copy the CVSM project website or its Pages workflow.

The first import created main directly, by the one-time bootstrap rule of the CVSM adoption guide, and then ran `scripts/setup.sh`. After that, all work uses issues, branches and PRs. The first changes were the Project fields, the directory and protection scope, the CPU dependency lock and CI. The core state machine, the issue title fields, the review verdict, the merge rules and the spec-check interface of the results comment stay as they were.

TorchTitan uses the v0.3.0 source baseline `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`; see the [source lock](../configs/upstream.json). This is a baseline for a compatible interface. It is not by itself a GPU, multi-node or resume acceptance. For those acceptances, see the [validation record](validation.md).

The project uses an MLflow HTTP service. Clients should set `MLFLOW_TRACKING_URI` explicitly. They must not open or migrate the SQLite file of the service directly.

A later adaptation adds `require_tracking_uri=true` to `scripts/run.toml`, and run and results share the same check. The reference SQLite fallback of CVSM is kept only for configurations that do not enable this option. This keeps tests with an offline temporary database easy. The default experiment changed to `moeloop-lab`, and the seed option changed to the native `debug.seed`. The other upstream run/results record and spec parsing algorithms are kept.

This repository adds `src/moeloop_lab/protocol.py`, a read-only doctor, a CPU smoke and the matching tests. For the interface and its limits, see the [experiment protocol](protocol.md) and the [training integration](torchtitan.md).
