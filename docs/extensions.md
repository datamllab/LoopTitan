# 写一个不同的状态更新

改 depth、共享 attention 或 router 等已有机制时，直接组合 `ModelConfig` 与 `TrainingRecipe`。改变块内算子顺序时，写一段明确的新 forward。[FFN-chain 示例](../recipes/ffn_chain.py)复用现有 Cell 的物理参数，不给 core 增加论文开关或通用执行器。

该例固定一种语义：attention 一次、FFN norm 一次，之后多次共享 router/expert 更新，并保留 inner 与 outer residual：

```text
a = x + scale * Attention(AttentionNorm(x))
z = FFNNorm(a)
repeat K times:
    z = z + scale * Experts(z, Router(z))
y = a + z
```

即使 K=1，它也不是普通 Cell：outer residual 加的是整个 chain 的结果，其中包含归一化后的初态。这些残差与 norm 的位置属于方法定义，应直接写在 forward 中。其结构参考 [CoE 的固定源码](https://github.com/ZihanWang314/coe/blob/d2af562aab3348f72d5d17132df439a629b9666d/config/models/coe_deepseekv2/modeling_coe.py#L1271-L1309)；此例只选共享 router/bank 和 inner+outer residual，不声称复现论文的路由、独立 gates、数据或训练细节。

从 Git checkout 运行下面的最小例子。`LoopedLM`（`RecursiveLM` 是它的兼容别名）在这里提供已初始化的参数 owners，新的更新函数显式调用这些参数：

```python
import torch
from moeloop import LoopedLM, ModelConfig
from recipes.ffn_chain import forward_ffn_chain

model = LoopedLM(ModelConfig(vocab_size=128, dim=64, hidden_dim=128,
    num_heads=4, num_kv_heads=2, num_experts=4, num_cells=1, num_visits=1))
ids = torch.randint(0, 128, (2, 16))
positions = torch.arange(16).expand_as(ids)
hidden, balance_sum, z_sum = forward_ffn_chain(
    model.core[0], model.embedding(ids), ffn_steps=3, positions=positions)
logits = model.lm_head(model.final_norm(hidden))
```

`balance_sum`、`z_sum` 累加三个 FFN calls，若使用逐调用平均，应显式除以 3。`attention_mask` 接受已准备好的 causal/document mask；`router_mask` 独立标记有效输入。该例采用完整反传。

这只是 reference forward 的扩展示例。直接调用 `model(ids)` 仍执行原有整块 recurrence，导出现有 `LoopedLM` 也只保存原有模型语义。要把新方法作为可训练/可交付模型，需要给自己的模型定义序列化身份，核对新的逻辑 attention/cache 次数，再明确接入原生 runtime 与 HF；当前示例没有取得这些资格。不要通过 monkey patch 训练后沿用旧 config 导出。

检查覆盖实际 attention/norm/router/expert 调用次数、全部参数梯度，以及零更新时的残差反例：`python -m pytest -q tests/model/test_extension.py`。

一致性测试套件 [`moeloop.testing.check_recipe(spec)`](../src/moeloop/testing.py) 检查的是 `LoopedLM(spec.model)` 本身：参数 owner、共享梯度、成本模型的调用数、cache 与 HF 往返。它不会运行这里自写的 forward；自写的更新要有自己的上述检查，细节见[开发指南](development.md#一致性测试与诊断)。

## 替换 attention 与逐轮 LoRA

`LoopedLM(config, attention_factory=...)` 可以换掉每个 Cell 的 attention 模块；原生训练就用它在同一批参数上装入 FlexAttention。Cell 以 `attention(x, positions=..., attention_mask=..., cache=..., slot=...)` 调用它，只有开启 attention LoRA（`lora_attention_rank > 0`）时才另外传 `lora=`。

- 不开 attention LoRA 时，第三方 `attention_factory` 返回的模块无需改动。
- 要支持 attention LoRA，`forward` 需接受 `lora`：当轮的 `nn.ModuleDict`，键为 `q_proj`、`k_proj`、`v_proj`、`out_proj`。把 `lora.q_proj(x)`、`lora.k_proj(x)`、`lora.v_proj(x)` 分别加到对应投影的输出上；`out_proj` 的增量作用于合并后的头输出 `h`，即把 `lora.out_proj(h)` 加到 `out_proj(h)` 上。内置的 [`Attention`](../src/moeloop/model.py) 就是这样写的。`forward` 不接受 `lora` 时，开启 attention LoRA 会直接报 `TypeError`。
- FFN 的 LoRA（`lora_ffn_rank > 0`）由 Cell 自己加在 dense 或 always-on FFN 的 `w1`、`w3`、`w2` 上，对 `expert_factory` 和 `attention_factory` 都没有要求；路由专家不加 LoRA。

`ModelConfig` 的 `lora_attention_rank`、`lora_ffn_rank` 默认为 0，只作用于 core：prefix/suffix 解析后两者都为 0。它们与 EMA 记忆单元（`state_update="ema"`）互斥，`lora_ffn_rank` 要求 core 有 dense 或 always-on FFN；和其他逐轮 owner 一样，要求固定深度与完整反向传播。每份 LoRA 默认零初始化（`A` 正态、`B = 0`），初始模型与共享权重模型逐位一致。要把一个全尺寸模型转换成 relaxed 模型，用 `LoRA.svd_init` 取论文公式 4 的截断 SVD 初始化，使 `W' + B A` 成为该层该轮权重 `W_l` 的最优秩 r 近似：

```python
import torch
from moeloop.model import LoRA

shared, full = torch.randn(64, 32), torch.randn(64, 32)  # W' 与 W_l，形状 [out, in]
lora = LoRA(32, 64, rank=8)
lora.svd_init(full - shared)
approx = shared + lora.B @ lora.A  # W_l 的最优秩 8 近似
```

这一调用约定的检查是 [`tests/model/test_relaxed.py`](../tests/model/test_relaxed.py) 中的 `NoLoRAAttention`：不开 attention LoRA 时结果与内置 attention 逐位相同，开启时报 `TypeError`。recipe 层面的用法见[配方](recipes.md)中的 `relaxed_spec`。
