# TorchTitan experiment record integration (optional)

This document describes the record interface of `moeloop_lab.protocol`. For the use of the public `moeloop` model with the native Trainer, see [Training and data](training.md). For the actual machine qualification, see the [validation record](validation.md).

The interface is based on the [fixed source of the TorchTitan v0.3.0 Trainer](https://github.com/pytorch/torchtitan/blob/086bf6c166ec85c1298eb5596fa9bf95f6a2d840/torchtitan/trainer.py). It does not drift with main. `Trainer.Config.to_dict()` gives the configuration. The batch mesh of `parallel_dims` gives the real data-parallel degree. `gradient_accumulation_steps` gives the accumulation count that the Trainer derived.

## Insert the record into the training entry point

In the existing GPU environment, install this library separately in editable mode. Do not install the control-side MLflow dependency into the training environment. The base package has only standard-library dependencies. Keep the Git checkout and the protected `configs/upstream.json`.

```sh
/training/env/bin/python -m pip install --no-deps -e /path/to/moeloop-lab
```

The model entry point builds the native Trainer.Config, the data and the model itself. Add the calls below. Do not copy the Trainer or change the training mathematics:

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

Call this on every rank. Only global `RANK=0` reads the sources and the manifest and writes files. Clean up the distributed process group as the original training entry point does. Use torchrun to propagate an error on any rank to the other workers. The adapter adds no barrier, and it does not take over the model or the process lifecycle. A record failure must make the entry point exit with a non-zero code.

`native.step_counter` is only a native state field. It cannot certify the number of completed optimizer steps. This version increments the step before it runs train_step, and it exits directly when it catches data exhaustion. Thus the count can still equal the target when the last step did not complete. `FINISHED` likewise only means that the command exited with 0. The actual training entry point must separately verify the successful optimizer steps, the consumed tokens and the necessary checkpoints. The configuration adapter of this version does not give this completion certification.

CVSM wraps **the main command of one complete allocation** from a clean run worktree. The control environment uses `uv run --frozen python scripts/run.py --issue N -- <cmd>`. `<cmd>` must explicitly select the python or torchrun of the existing training environment. Do not use the control environment by mistake, because it has no Torch. `RUN_DIR`, `MLFLOW_RUN_ID` and `MLFLOW_TRACKING_URI` are passed to the child process.

The first version verifies only the local command record interface. Do not wrap a launcher on each remote node separately and treat several runs as the same allocation. Multi-node data distribution, network, scheduler, GPU budget and resume need their own integration.

## Batch and token definitions

```text
effective batch = local_batch_size × batch_mesh_degree × gradient_accumulation_steps
input positions per step = effective batch × seq_len
target input positions = input positions per step × original configured steps
```

For example, with world size 32, batch mesh 16, local batch 1, accumulation 8 and seq_len 4096, one step is 524288 input positions. It is not computed from 32 data replicas. The native automatic mode `global_batch_size=-1` can only correspond to one accumulation step. A declaration that does not match the actual accumulation is refused and not recorded. This prevents recipes that look the same from hiding different token budgets.

This counts the configured input positions. It does not count non-padding tokens, separate samples, unique tokens or FLOPs. Physical parameters, active compute, throughput and valid loss tokens each need their own real measurement. Do not infer them from these products.

## MLflow recorder

```sh
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run --frozen python scripts/doctor.py
```

doctor only reads `/health` and, through the official SDK, makes a bounded query of at most one experiment. It does not create records, migrate the database or restart the service. Authentication comes from the standard MLflow environment variables. Credentials do not go into the URI or Git. The two CVSM commands use the same explicit endpoint. The default project configuration refuses to create a local SQLite store silently.

Use `MLFLOW_EXPERIMENT_NAME` to give each separate study a stable namespace. A remote worker cannot use the `127.0.0.1` of the control machine directly. This version does not change the listen address or the authentication of the server, and it does not promise remote reachability. When the model writes only small JSON artifacts, it does not need access to MLflow. The control-side launcher, which keeps running, does the upload.
