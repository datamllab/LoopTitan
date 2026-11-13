"""Portable Transformers delivery; the recurrent mathematics stays in model.py."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .cache import DecodeCache
from .model import ModelConfig, RecursiveLM


class MoeloopConfig(PretrainedConfig):
    model_type = "moeloop"

    def __init__(self, model_config: dict | None = None, **kwargs):
        core = ModelConfig.from_dict(model_config or {})
        use_cache = kwargs.pop("use_cache", core.state_update == "residual" and core.initial_state != "random")
        if kwargs.pop("tie_word_embeddings", False):
            raise ValueError("MoE-Loop v1 uses independent embedding and output weights")
        if kwargs.pop("labels_objective", "terminal_ce") != "terminal_ce":
            raise ValueError("HF labels evaluate terminal CE; expected-loss training uses the native training objective")
        for key in ("vocab_size", "hidden_size", "num_attention_heads", "num_key_value_heads", "num_hidden_layers", "max_position_embeddings", "is_encoder_decoder"):
            kwargs.pop(key, None)
        super().__init__(tie_word_embeddings=False, is_encoder_decoder=False, **kwargs)
        self.model_config = core.to_dict()
        self.vocab_size = core.vocab_size
        self.hidden_size = core.dim
        self.num_attention_heads = core.num_heads
        self.num_key_value_heads = core.num_kv_heads
        self.num_hidden_layers = core.prefix_cells + core.num_cells * core.num_visits + core.suffix_cells
        self.max_position_embeddings = core.max_seq_len
        self.use_cache = use_cache
        self.labels_objective = "terminal_ce"

    def to_model_config(self) -> ModelConfig:
        return ModelConfig.from_dict(self.model_config)


class MoeloopCache(Cache):
    """HF operations over one Torch-only logical KV store, without copying KV."""

    def __init__(self, num_visits: int):
        super().__init__(layers=[])
        self.store = DecodeCache(num_visits)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.store.length

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self.store.length + query_length, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def reorder_cache(self, beam_idx: Tensor) -> None:
        self.store.reorder(beam_idx)

    def batch_select_indices(self, indices: Tensor) -> None:
        self.store.reorder(indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.store.valid_mask is not None:
            indices = torch.arange(self.store.valid_mask.size(0), device=self.store.valid_mask.device)
            self.store.reorder(indices.repeat_interleave(repeats))

    def reset(self) -> None:
        self.store.clear()

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_initialized(self) -> bool:
        return self.store.length > 0

    @property
    def is_croppable(self) -> bool:
        return False


class MoeloopForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MoeloopConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["Cell", "MemoryCell"]

    def __init__(self, config: MoeloopConfig):
        super().__init__(config)
        self.model = RecursiveLM(config.to_model_config())
        self.post_init()

    def _init_weights(self, module):
        # RecursiveLM initializes its canonical owners once; HF must not reset them.
        pass

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, value):
        self.model.lm_head = value

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
        old_size = self.model.embedding.weight.shape[0]
        embeddings = super().resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of=pad_to_multiple_of, mean_resizing=mean_resizing)
        size = embeddings.weight.shape[0]
        if size > old_size and not mean_resizing:
            # _init_weights skips already-initialized core owners. New rows
            # still need the core's declared initialization rather than nn defaults.
            with torch.no_grad():
                for owner in (embeddings, self.model.lm_head):
                    torch.nn.init.normal_(owner.weight[old_size:], std=self.model.config.init_std)
        self.model.config = replace(self.model.config, vocab_size=size)
        self.config.model_config = self.model.config.to_dict()
        return embeddings

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length):
        if generation_config.cache_implementation is not None:
            raise ValueError("MoE-Loop supports its exact logical cache; omit cache_implementation")
        if model_kwargs.get("past_key_values") is not None and not isinstance(model_kwargs["past_key_values"], MoeloopCache):
            raise TypeError("past_key_values must be a MoeloopCache")
        if generation_config.use_cache and model_kwargs.get("past_key_values") is None:
            depth = model_kwargs.get("num_visits")
            depth = self.model.config.num_visits if depth is None else depth
            model_kwargs["past_key_values"] = MoeloopCache(depth)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, position_ids=None, **kwargs):
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past:
            input_ids = input_ids[:, past:] if input_ids.size(1) > past else input_ids[:, -1:]
        if position_ids is None and attention_mask is not None:
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        if position_ids is not None:
            position_ids = position_ids[:, -input_ids.size(1):]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True),
            "num_visits": kwargs.get("num_visits"),
            "logits_to_keep": kwargs.get("logits_to_keep", 1),
        }

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                labels=None, use_cache=None, num_visits=None, logits_to_keep=0, return_dict=True,
                output_attentions=False, output_hidden_states=False, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids are required; inputs_embeds is not a v1 interface")
        if output_attentions:
            raise ValueError("attention weight materialization is not supported")
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("inputs_embeds is not supported")
        if labels is not None and past_key_values is not None:
            raise ValueError("labels and an existing inference cache cannot be combined")
        depth = self.model.config.num_visits if num_visits is None else num_visits
        cache_enabled = self.config.use_cache if use_cache is None else use_cache
        cache_enabled = cache_enabled and not self.training and labels is None and not torch.is_grad_enabled()
        if past_key_values is not None and not isinstance(past_key_values, MoeloopCache):
            raise TypeError("past_key_values must be a MoeloopCache")
        if past_key_values is not None and not cache_enabled:
            raise ValueError("past_key_values requires use_cache=True in inference mode")
        cache = past_key_values if cache_enabled else None
        if cache_enabled and cache is None:
            cache = MoeloopCache(depth)
        if cache is not None and cache.store.num_visits != depth:
            raise ValueError("generation depth cannot change within a cache session")
        valid = None if attention_mask is None else attention_mask[:, -input_ids.size(1):].bool()
        out = self.model(input_ids, positions=position_ids, attention_mask=None if attention_mask is None else attention_mask.bool(),
                         router_mask=valid, num_visits=depth, cache=None if cache is None else cache.store,
                         return_hidden=True, collect_readouts=False)
        hidden = out.hidden_states
        if labels is None and isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:]
        elif labels is None and isinstance(logits_to_keep, Tensor):
            hidden = hidden[:, logits_to_keep]
        logits = self.model.lm_head(hidden)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("HF labels must have the same shape as input_ids")
            targets = labels[:, 1:].contiguous()
            count = (targets != -100).sum()
            if count.item() == 0:
                raise ValueError("HF labels have no valid next-token targets")
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum", ignore_index=-100) / count
        result = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache,
                                       hidden_states=(out.hidden_states,) if output_hidden_states else None)
        return result if return_dict else result.to_tuple()


# save_pretrained copies this file and its same-directory relative dependencies.
MoeloopConfig.register_for_auto_class("AutoConfig")
MoeloopForCausalLM.register_for_auto_class("AutoModelForCausalLM")
