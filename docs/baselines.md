# 独立 baseline 核对

使用 **Transformers 官方 `Qwen3MoeForCausalLM`** 作为可执行参考。它的 attention、RoPE、QK norm、router、experts、decoder、CE 与 KV cache 均直接执行上游代码；我们只映射参数并安排官方 decoder 的逻辑调用顺序。[官方模型说明](https://huggingface.co/docs/transformers/model_doc/qwen3_moe)、[固定版本源码](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)。

这是发现模型实现大问题的数值检查。我们不据此宣称复现某篇论文的训练结果、获得预训练质量或完成 FLOP/吞吐对比。相同配置名也不意味着与论文 parameter/compute matched。

在已安装项目 HF extra 的环境中执行：

```bash
python scripts/compare_baselines.py --output outputs/baselines/qwen3-cpu.json
python -m pytest -q tests/reference
```

脚本不下载模型或数据。默认 CPU、FP32、单计算线程，6 个 tiny 配置各执行 4 步 AdamW；计算本身约 1 秒，另有 Python/依赖启动时间。在 GPU 上可用同一脚本：

```bash
python scripts/compare_baselines.py --device cuda --output outputs/baselines/qwen3-cuda.json
```

GPU 命令仍是 FP32 独立 reference 核对；原生 BF16 grouped/Flex 的数值、FSDP 通信与 continuation 分别记录在算子对照和[训练资格](validation.md)中。

检查覆盖以下真实机制组合。所有配置都有 GQA、QK norm、split-half RoPE、pre-norm 残差、SwiGLU 与归一化的 top-k softmax 路由：

| 配置 | 官方参考的逻辑执行 | 物理参数共享 |
|---|---|---|
| `ordinary_moe` | prefix、A、B、suffix，各一次 | 各层独立 |
| `shared_stack` | A、B、A、B、A、B | 同一个 A/B 的全部参数共享 |
| `shared_layer` | A、A、A、B、B、B | 同一个 A/B 的全部参数共享 |
| `partial_attention` | A、B 循环三轮 | attention 每轮独立；router/norm/expert 共享 |
| `partial_router` | A、B 循环三轮 | router 每轮独立；attention/norm/expert 共享 |
| `expert_tying` | 每个 cell 连续三轮 | attention/router/norm 每轮独立；expert 共享 |

每个配置都核对完整 logits、shifted next-token CE、**全部物理参数的梯度**、4 次 AdamW 更新后的参数与 CE 下降。还将同一序列拆成 `3+2+2` 列执行缓存推理，对照官方 `DynamicCache` 的输出及完整前缀输出；共享 attention 参数仍对应独立的逻辑缓存槽。负对照会将 `ABABAB` 改成 `AAABBB` 并确认输出确实改变，防止测试对调用顺序失去敏感性。

参数映射覆盖每一个物理 owner。我们的 `w1_EFD`/`w3_EFD` 映射到官方 packed `gate_up_proj` 的前/后半，`w2_EDF` 映射到 `down_proj`；`out_proj` 对应 `o_proj`。参考中的共享 `nn.Parameter` 由 AdamW 去重更新。每个逻辑 attention 使用不同 `layer_idx`，其 Linear/Norm 参数对象按实际共享关系复用。参考前向不调用 `moeloop` 的算子或 `forward`。

CPU 实测环境为 Python 3.12.3、torch 2.14.0+cpu、Transformers 5.20.2。下面是 4 步更新期间的最大绝对误差；缓存列取更新后缓存推理对官方输出的最大误差。脚本会输出完整 JSON，包括依赖版本、官方源文件 SHA256、配置、优化器、容差和每步 loss。

| 配置 | logits | CE | 全部梯度 | 更新参数 | 缓存 logits | CE 首步 → 末步 |
|---|---:|---:|---:|---:|---:|---:|
| ordinary_moe | 5.96e-6 | 4.77e-7 | 2.17e-6 | 4.82e-5 | 5.84e-6 | 3.8242 → 2.9064 |
| shared_stack | 8.05e-7 | 4.77e-7 | 9.17e-7 | 1.17e-6 | 6.86e-7 | 3.9669 → 3.1369 |
| shared_layer | 1.52e-6 | 2.39e-7 | 9.39e-7 | 1.81e-6 | 1.39e-6 | 3.9152 → 2.9858 |
| partial_attention | 7.16e-7 | 0 | 3.90e-7 | 2.46e-6 | 9.99e-7 | 3.9032 → 2.9439 |
| partial_router | 3.58e-7 | 0 | 2.54e-7 | 4.21e-7 | 4.77e-7 | 3.7068 → 3.0580 |
| expert_tying | 1.20e-6 | 4.77e-7 | 5.63e-7 | 1.90e-5 | 1.64e-6 | 3.9282 → 2.7669 |

我们使用 SDPA，参考使用官方 eager attention；浮点归约顺序不同。AdamW 的 `eps=1e-8` 会放大接近零的梯度差异，因此更新参数的绝对容差为 `1e-4`；logits 使用更紧的 `atol=3e-5, rtol=3e-4`，CE 为 `3e-6/3e-5`，梯度为 `3e-6/5e-4`。所有误差都实际计算并受断言约束，JSON 保存实际误差而非仅写“通过”。这些短序列的 loss 下降仅证明两套实现有一致的短程更新行为。

同一脚本也在 H200、Torch 2.14.0+cu130、Transformers 5.19.0 上实际执行了六配置各 **2 次**更新。CUDA 最大 logits 误差为 `2.44e-6`、CE `2.38e-7`、梯度 `5.15e-7`、更新参数 `5.05e-5`、cached logits `6.06e-6`，沿用原容差通过。两次运行的完整参数、依赖、源文件 hash、每步 CE 与误差分别保存在 [CPU JSON](evidence/qwen3-cpu.json) 和 [CUDA JSON](evidence/qwen3-cuda.json)。CUDA baseline 仍是 plain core 对官方模型；native grouped experts/FlexAttention 的资格单列，不能混为同一个检查。

在已分配的 CUDA GPU 上，用[公开原生算子脚本](../scripts/compare_native.py)核对同一份 BF16 参数的 grouped experts/FlexAttention 与 plain expert loop/SDPA：

```bash
python scripts/compare_native.py --output outputs/baselines/native-kernel.json
```

它固定 seed、输入与两种配置，覆盖 shared 和带 input injection/prefix/suffix 的尾段反传，包含 document/padding mask、CE 与 balance/z 辅助梯度。实际 H200 检查共 80 个物理参数张量，forward/辅助标量完全一致；逐有效 token、逐逻辑调用的 top-k expert 集合也精确一致，允许 top-k 顺序不同。梯度同时满足 3% 向量误差和 `2 * eps_bf16 * reference_peak + 1e-6` 的最大坐标差约束。[精确路由重测](evidence/native-kernel-routing-qualified.json)覆盖 4/8 次调用；旧 schema-1 仅比较汇总路由计数，不能单独支撑逐 token 精确结论。该验收基于有限输入与 BF16 精度，不能推定普遍误差保证。首轮逐元素 gate 的失败、策略审阅、首轮失败后声明上述约束及重跑的完整记录见[验证记录](validation.md)。

匹配范围有几个明确限制：

- Qwen3-MoE 必须启用 QK norm；我们的其他配置仍可使用 `qk_norm=False`。
- 显式设置 `norm_topk_prob=True`，关闭 dropout、sliding window、attention bias 和词表权重 tying。
- 辅助目标均关闭。不同实现跨层聚合 load balance 的语义可能不同，不能把它混入主 CE 核对而假装等价；本项目的逐调用辅助目标另有直接语义测试。
- 此项比较使用完整反传、固定深度、无 input injection、无 always-on shared expert。状态注入、尾段反传、padding/document 边界、数据恢复等分别由相应语义与集成检查覆盖。

我们也阅读了 [xLLM-Loop 的固定源码](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)。其中 `build_loop_execution_schedule` 明确采用 prefix + loop span 重复 + suffix，与我们的 stack 语义相符；其单状态 recurrent update 和 no-grad prefix/live gradient suffix 为训练动态提供了额外源码参照。**目前没有执行 xLLM 的完整模型或训练，也没有将这项阅读列为数值通过项。** 实际执行的独立 baseline 是上面的可重复官方 Qwen3-MoE 检查；正式比较特定 loop 方法需冻结对应上游版本、数据、预算和需要匹配的具体机制。
