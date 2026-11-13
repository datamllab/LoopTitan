# LoopTitan

LoopTitan is a PyTorch library for training and comparing looped (weight-shared) Mixture-of-Experts language models. Published looped recipes are configurations of one explicit model, costed and budget-matched by one cost model, trained on TorchTitan with AdamW or distributed Muon, and exported as standalone Hugging Face models.

## Install

Install from source. Python ≥3.11; verified with Python 3.12, Torch 2.14.0, Transformers 5.20.2, tokenizers 0.23.3 and safetensors 0.8.0. From the source root, for the CPU model and data examples:

```sh
uv venv --python 3.12
. .venv/bin/activate
uv pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cpu
uv pip install '.[data,hf]'
```

GPU training needs a hardware-compatible Torch 2.14.0 CUDA wheel and the `train` extra; see [Training and data](docs/training.md).

## Quickstart

Build the SMELT recipe on CPU, run a forward pass and read the cost model. The six forward cells are one prefix, two core cells visited twice and one suffix.

```python
import torch
from moeloop import LoopedLM
from moeloop.recipes import smelt_spec
from moeloop.costs import estimate_cost

spec = smelt_spec(physical_layers=4, vocab_size=128, dim=32,
                  hidden_dim=64, num_heads=4, num_kv_heads=2,
                  num_experts=4, max_seq_len=64)
model = LoopedLM(spec.model).eval()
with torch.no_grad():
    out = model(torch.randint(0, 128, (2, 16)))
print(tuple(out.logits.shape))  # (2, 16, 128)
print(estimate_cost(spec, seq_len=16)["calls"]["forward_cells"])  # 6
```

## Repository layout

| | Path | Contents |
|---|---|---|
| **Research** | `src/moeloop/` | The model, recipes, cost model, comparison contract, training adapters, export and bench validator. |
| | `recipes/` | Frozen configurations of the reported runs, reloadable with `RecipeSpec.from_dict`. |
| | `scripts/compare_recipes.py` | Costs and budget-matches two recipes on a CPU. |
| | `scripts/recipe_atlas.py` | Sweeps each recipe's shapes and marks those matching a reference on all three budgets. |
| | `eval/bench/` | The frozen bench tracks. |
| | `docs/` | Architecture, recipes, method map, training, comparison and the validation record, written in Chinese; `docs/evidence/` holds the evidence files. |
| **Project process** | `AGENTS.md`, `.agents/` | The protocol the agents developing this repository follow. |
| | `.github/` | Issue forms, CI and merge tooling. |
| | `scripts/run.py`, `scripts/results.py`, `src/moeloop_lab/` | Run tooling: recorded launches and results comments. |

## Research record

The [issues](https://github.com/datamllab/LoopTitan/issues) are a copy of the development record, under the original issue numbers: studies with their criteria, run tasks with their results tables, code tasks and pull-request records. Private paths and details of the operating environment were removed; numbers, tables and verdicts are unchanged.

## Tests

```sh
uv sync --frozen
uv run --frozen python -m pytest -q
```

## License

[Apache-2.0](LICENSE). Built on [CVSM](https://github.com/doem97/cvsm). Cite the repository and the exact source commit ([CITATION.cff](CITATION.cff)); cite the method papers separately, see the [method map](docs/method-map.md).
