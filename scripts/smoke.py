#!/usr/bin/env python3
"""CPU engineering fixture for the CVSM record interface; never trains a model."""

import argparse

from moeloop_lab.protocol import build_record, write_metrics, write_record


SPEC = """### Question
CPU 工程 fixture 的记录与规格核对接口是否接通？不是模型效果比较。
### Arms
once: native.model_spec.model.loop_repeat=1
twice: native.model_spec.model.loop_repeat=2
### Controlled
record_kind=engineering-fixture  # 明确区分工程检查与训练证据
native.debug.seed in 1,2  # 每组都覆盖相同 fixture seed
derived.global_batch_size=8  # 固定 fixture 的有效 batch
derived.target_input_tokens=64  # 固定 fixture 的输入位置预算
### Adaptable
### Criteria
所有 fixture 退出成功、cfg.* 与 artifact 读回一致，规格核对无偏差。
### Budget
仅 CPU 工程 fixture；不使用 GPU。
### Context
scripts/smoke.py；本规格仅供工程验收，不是已批准的研究 study。
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug.seed", dest="seed", type=int, default=1)
    parser.add_argument("--loop-repeat", type=int, choices=(1, 2), default=1)
    parser.add_argument("--print-spec", action="store_true")
    args = parser.parse_args(argv)
    if args.print_spec:
        print(SPEC)
        return 0
    native = {
        "training": {"local_batch_size": 1, "global_batch_size": 8, "seq_len": 4, "steps": 2},
        "debug": {"seed": args.seed},
        "model_spec": {"name": "engineering-fixture", "model": {"loop_repeat": args.loop_repeat}},
    }
    write_record(build_record(native, world_size=1, data_parallel_degree=1,
                              gradient_accumulation_steps=8, assets={}, sources={},
                              record_kind="engineering-fixture"))
    write_metrics({"fixture.completed": 1})
    print("CPU record fixture completed; no model was trained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
