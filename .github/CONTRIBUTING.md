# Contributing a result

Anyone can enter a result in the Looped MoE Bench, by hand or through an agent, under the same rules as the maintainers. Every complete record is kept. Failed runs and negative results are records too, and they are welcome.

## Enter a result

1. Set up the training environment and prepare the data as [docs/training.md](../docs/training.md#set-up-the-environment) describes. Your data manifest must have the fingerprint that the track fixes in [`eval/bench/tracks.json`](../eval/bench/tracks.json).
2. Write your design as a recipe spec (JSON), commit all code changes, and push that commit to a public repository. A run records its source commit, and a tracked file that differs from it at launch is a protocol mismatch.
3. Run the frozen track on your spec. `tracks.json` gives each track's learning rates and fixed settings, and [Looped MoE Bench](../docs/comparison.md#looped-moe-bench) gives its rules.

   ```sh
   # Tracks P2 (the default) and L: one run at the track's learning rate for your optimizer.
   python scripts/bench.py run --track P2 --spec my-design.json --lr LR --gpus 8 --output runs/my-design
   # Tracks S and M: the track's learning-rate grid and selection rule.
   python scripts/bench.py sweep --track M --spec my-design.json --gpus 8 --output runs/my-design
   ```

   The command writes the folder `runs/my-design/submission/` and prints its report. `python scripts/bench.py validate --dir runs/my-design/submission` prints it again.
4. Copy that folder unchanged to `eval/bench/submissions/<name>/`. Pick a new `<name>` of lowercase letters, digits and hyphens, for example `p2-my-design-<your GitHub account>`.
5. Open a pull request from your fork that adds only that folder. It needs no issue and no title format. Its description states the track, the design, the link to the commit the run used, and the person who answers for the entry.

## What the check binds

The `bench` workflow runs `scripts/bench.py validate --dir` on each folder the pull request adds or changes, with the validator and the tracks of the base branch, and writes the report to the job summary. It recomputes the sha256 of every file and compares the captured configurations and evaluation outputs with the declaration and the frozen track. It does not rerun training or evaluation, so it cannot show that the run produced these files. [Benchmark submission folder](../docs/comparison.md#benchmark-submission-folder) lists what it binds and what it leaves unchecked.

The check fails only when the record is incomplete: a `missing-evidence` reason, a folder the validator cannot read, a symbolic link, a file added directly in `eval/bench/submissions/` instead of a folder, or a folder name that is not lowercase letters, digits and hyphens. A `budget-mismatch` or `protocol-mismatch` verdict passes the check: the record is complete, and the leaderboard ranks `eligible` entries only.

## Merge rule

A maintainer merges every complete record whose pull request names the person who answers for it, whatever its result. A pull request that also changes anything else, a merged entry included, needs an issue first, as [AGENTS.md](../AGENTS.md) requires.

## Who is accountable

The person named in the pull request answers for the entry: its code, its data and its claims. When an agent prepares the entry, the person who sent the agent is named and answers for it.

## Agents

An agent without write access to this repository follows this page. The issue, branch and review protocol of AGENTS.md is the maintainers' and does not apply to an entry.

## Change the rules

Propose a change to a track, the check or this page in an issue: the rule, the change and the reason. The maintainers decide in that issue. Any change to a frozen track is a protocol change.
