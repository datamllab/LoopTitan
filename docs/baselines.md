# Independent baseline check

The executable reference is the **official Transformers `Qwen3MoeForCausalLM`**. Its attention, RoPE, QK norm, router, experts, decoder, CE and KV cache all run the upstream code directly. We only map the parameters and arrange the logical call order of the official decoder. See the [official model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_moe) and the [source at a fixed version](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py).

This is a numerical check that finds large problems in the model implementation. We do not use it to claim a reproduction of the training results of a paper, pretraining quality, or a completed FLOP or throughput comparison. The same configuration name also does not mean that the configuration is parameter-matched or compute-matched to a paper.

Run these commands in an environment that has the project's HF extra installed:

```bash
python scripts/compare_baselines.py --output outputs/baselines/qwen3-cpu.json
python -m pytest -q tests/reference
```

The script does not download models or data. By default it uses the CPU, FP32 and one compute thread, and it runs 4 AdamW steps for each of 6 tiny configurations. The computation itself takes about 1 second, plus the start-up time of Python and the dependencies. The same script also runs on a GPU:

```bash
python scripts/compare_baselines.py --device cuda --output outputs/baselines/qwen3-cuda.json
```

The GPU command is still the FP32 independent reference check. The numerics of native BF16 grouped/Flex, FSDP communication and continuation are recorded separately, in the kernel comparison and in the [training qualification](validation.md).

The check covers the real mechanism combinations below. All configurations have GQA, QK norm, split-half RoPE, pre-norm residuals, SwiGLU and normalized top-k softmax routing:

| Configuration | Logical execution in the official reference | Physical parameter sharing |
|---|---|---|
| `ordinary_moe` | prefix, A, B, suffix, once each | Each layer is independent |
| `shared_stack` | A, B, A, B, A, B | All parameters of the same A/B are shared |
| `shared_layer` | A, A, A, B, B, B | All parameters of the same A/B are shared |
| `partial_attention` | A, B looped for three visits | attention is separate per visit; router/norm/expert are shared |
| `partial_router` | A, B looped for three visits | router is separate per visit; attention/norm/expert are shared |
| `expert_tying` | Each cell for three visits in a row | attention/router/norm are separate per visit; expert is shared |

For each configuration, the check compares the full logits, the shifted next-token CE, **the gradients of all physical parameters**, and the parameters and the CE decrease after 4 AdamW updates. It also splits one sequence into `3+2+2` columns for cached inference. It compares this output with the output of the official `DynamicCache` and with the full-prefix output. Shared attention parameters still map to separate logical cache slots. A negative control changes `ABABAB` to `AAABBB` and confirms that the output really changes. This keeps the test sensitive to the call order.

The parameter mapping covers every physical owner. Our `w1_EFD`/`w3_EFD` map to the first and second halves of the official packed `gate_up_proj`, and `w2_EDF` maps to `down_proj`. `out_proj` maps to `o_proj`. AdamW updates each shared `nn.Parameter` in the reference once (deduplicated). Each logical attention uses a different `layer_idx`. Its Linear/Norm parameter objects are reused as the actual sharing specifies. The reference forward does not call the operators or the `forward` of `moeloop`.

The CPU measurement environment is Python 3.12.3, torch 2.14.0+cpu and Transformers 5.20.2. The table below gives the maximum absolute errors during the 4 updates. The cache column is the maximum error of cached inference after the updates against the official output. The script prints the complete JSON, with the dependency versions, the SHA256 of the official source file, the configuration, the optimizer, the tolerances and the loss at each step.

| Configuration | logits | CE | All gradients | Updated parameters | Cached logits | CE first step → last step |
|---|---:|---:|---:|---:|---:|---:|
| ordinary_moe | 5.96e-6 | 4.77e-7 | 2.17e-6 | 4.82e-5 | 5.84e-6 | 3.8242 → 2.9064 |
| shared_stack | 8.05e-7 | 4.77e-7 | 9.17e-7 | 1.17e-6 | 6.86e-7 | 3.9669 → 3.1369 |
| shared_layer | 1.52e-6 | 2.39e-7 | 9.39e-7 | 1.81e-6 | 1.39e-6 | 3.9152 → 2.9858 |
| partial_attention | 7.16e-7 | 0 | 3.90e-7 | 2.46e-6 | 9.99e-7 | 3.9032 → 2.9439 |
| partial_router | 3.58e-7 | 0 | 2.54e-7 | 4.21e-7 | 4.77e-7 | 3.7068 → 3.0580 |
| expert_tying | 1.20e-6 | 4.77e-7 | 5.63e-7 | 1.90e-5 | 1.64e-6 | 3.9282 → 2.7669 |

We use SDPA, and the reference uses the official eager attention. The floating-point reduction order is different. The AdamW `eps=1e-8` amplifies differences in gradients near zero. Thus the absolute tolerance for updated parameters is `1e-4`. Logits use the tighter `atol=3e-5, rtol=3e-4`, CE uses `3e-6/3e-5` and gradients use `3e-6/5e-4`. All errors are actually computed and bound by assertions. The JSON keeps the actual errors, not only "pass". The loss decrease on these short sequences proves only that the two implementations have the same short-range update behaviour.

The same script also ran **2** updates for each of the six configurations on an H200 with Torch 2.14.0+cu130 and Transformers 5.19.0. The maximum CUDA errors are `2.44e-6` for logits, `2.38e-7` for CE, `5.15e-7` for gradients, `5.05e-5` for updated parameters and `6.06e-6` for cached logits. They pass with the same tolerances. The [CPU JSON](evidence/qwen3-cpu.json) and the [CUDA JSON](evidence/qwen3-cuda.json) keep the complete parameters, dependencies, source file hashes, CE per step and errors of the two runs. The CUDA baseline is still the plain core against the official model. The qualification of native grouped experts/FlexAttention is separate. Do not mix the two into one check.

On an allocated CUDA GPU, the [public native kernel script](../scripts/compare_native.py) compares grouped experts/FlexAttention with the plain expert loop/SDPA on the same BF16 parameters:

```bash
python scripts/compare_native.py --output outputs/baselines/native-kernel.json
```

The script fixes the seed, the input and two configurations. It covers the shared configuration, and tail backpropagation with input injection, prefix and suffix. It includes document/padding masks, CE and the gradients of the balance/z auxiliary losses. The actual H200 check has 80 physical parameter tensors in total. The forward and the auxiliary scalars are identical. The top-k expert set per valid token and per logical call is also exactly identical. The top-k order can differ. Gradients meet both a 3% vector error bound and a maximum coordinate difference bound of `2 * eps_bf16 * reference_peak + 1e-6`. The [exact routing re-test](evidence/native-kernel-routing-qualified.json) covers 4 and 8 calls. The old schema-1 compares only aggregate routing counts. Alone, it cannot support an exact per-token conclusion. This acceptance rests on finite inputs and BF16 precision. It does not imply a general error guarantee. The [validation record](validation.md) has the complete record: the failure of the first element-wise gate, the policy review, the declaration of the bounds above after the first failure, and the re-run.

The matching scope has some explicit limits:

- Qwen3-MoE requires QK norm. Our other configurations can still use `qk_norm=False`.
- `norm_topk_prob=True` is set explicitly. Dropout, sliding window, attention bias and vocabulary weight tying are off.
- All auxiliary objectives are off. Different implementations can aggregate the load balance across layers with different semantics. Thus it is not mixed into the main CE check as if the semantics were equivalent. The per-call auxiliary objectives of this project have their own direct semantic tests.
- This comparison uses full backpropagation, fixed depth, no input injection and no always-on shared expert. State injection, tail backpropagation, padding/document boundaries, data resume and similar items are covered by their own semantic and integration checks.

We also read the [fixed source of xLLM-Loop](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py). Its `build_loop_execution_schedule` explicitly uses prefix + repeated loop span + suffix, which agrees with our stack semantics. Its single-state recurrent update and its no-grad prefix / live-gradient suffix give an additional source reference for training dynamics. **We have not run the complete xLLM model or its training, and this reading is not listed as a numerical pass.** The independent baseline that actually runs is the reproducible official Qwen3-MoE check above. A formal comparison with a specific loop method needs a frozen upstream version, data, budget and the specific mechanisms that must match.
