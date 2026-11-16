#!/usr/bin/env python3
"""CPU engineering fixture for the CVSM record interface; never trains a model."""

import argparse

from moeloop_lab.protocol import build_record, write_metrics, write_record


SPEC = """### Question
Are the record and Spec check interfaces of the CPU engineering fixture connected? This is not a comparison of model quality.
### Arms
once: native.model_spec.model.loop_repeat=1
twice: native.model_spec.model.loop_repeat=2
### Controlled
record_kind=engineering-fixture  # keeps engineering checks apart from training evidence
native.debug.seed in 1,2  # every arm covers the same fixture seeds
derived.global_batch_size=8  # fixed effective batch of the fixture
derived.target_input_tokens=64  # fixed input-position budget of the fixture
### Adaptable
### Criteria
All fixtures exit successfully, cfg.* and the artifacts read back the same, and the Spec check shows no deviations.
### Budget
CPU engineering fixture only; no GPU.
### Context
scripts/smoke.py. This spec is for engineering acceptance only. It is not an approved study.
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
