#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""OpenELM config"""
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import Normal
from mindspore.ops.function.nn_func import _scaled_dot_product_attention

from ....modules.functional import finfo
from ...modeling_utils import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ....utils import logging
from .configuration_openelm import OpenELMConfig, make_divisible

logger = logging.get_logger(__name__)

# this import has to be relative, otherwise, when setting trust_remote_code=True
# huggingface transformers won't be able to load the module correctly


class OpenELMRMSNorm(nn.Cell):
    def __init__(self, num_features: int, eps: float = 1e-6):
        """
        Initialize the OpenELMRMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ops.ones(num_features))
        self.num_features = num_features

    def _norm(self, x: Tensor) -> Tensor:
        """
        Apply the OpenELMRMSNorm normalization to the input tensor.

        Args:
            x (mindspore.Tensor): The input tensor.

        Returns:
            mindspore.Tensor: The normalized tensor.

        """
        return x * ops.rsqrt(x.pow(2).mean(-1, keep_dims=True) + self.eps)

    def construct(self, x: Tensor) -> Tensor:
        """
        Forward pass through the OpenELMRMSNorm layer.

        Args:
            x (mindspore.Tensor): The input tensor.

        Returns:
            mindspore.Tensor: The output tensor after applying OpenELMRMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class OpenELMPreTrainedModel(PreTrainedModel):
    config_class = OpenELMConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["OpenELMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell: nn.Cell) -> None:
        """Initialize the weights."""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.initialize(Normal(self.config.initializer_range))
            if cell.bias is not None:
                cell.bias.initialize('zeros')
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))

        elif isinstance(cell, OpenELMRMSNorm):
            cell.weight.initialize('ones')


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, axis=-1)
    return ops.cat((-x2, x1), axis=-1)


def _apply_rotary_pos_emb(x: Tensor, pos_sin: Tensor, pos_cos: Tensor) -> Tensor:
    return (x * pos_cos) + (_rotate_half(x) * pos_sin)


class OpenELMRotaryEmbedding(nn.Cell):
    """
    The rotary position embeddings (aka RoPE) from `RoFormer <https://arxiv.org/abs/2104.09864>`_.

    RoPE encodes the position information of tokens using a rotation matrix, and is able to capture
    explicit relative positional dependencies.

    Args:
        model_dim: The dimensionality of the model's hidden state.
        max_seq_length: Maximum sequence length.
        freq_constant: A constant used for computing frequencies.
    """

    def __init__(
        self, model_dim: int, max_seq_length: int, freq_constant: int = 10000
    ) -> None:
        inv_freq = 1.0 / (
            freq_constant
            ** (ops.arange(0, model_dim, 2, dtype=mindspore.float32) / model_dim)
        )
        super().__init__()

        self.model_dim = model_dim
        self.freq_constant = freq_constant
        self.max_seq_length = max_seq_length

        self.inv_freq = inv_freq
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = max_seq_length
        self._compute_sin_cos_embeddings(max_seq_length)

    def extra_repr(self) -> str:
        return f"\tmodel_dim={self.model_dim}, max_seq_length={self.max_seq_length}, freq_constant={self.freq_constant}"

    def _compute_sin_cos_embeddings(
        self,
        key_len: int,
        key_dtype = mindspore.float32,
    ) -> None:
        """
        Compute sine and cos embeddings.

        Args:
            key_len: Number of tokens in the key embeddings in the transformer model.
            key_dtype: Data type of the key embeddings.

        Returns:
            None

        ...note:
            We recalculate the sine and cosine embeddings if any of the following conditions are met:
                1. The number of tokens in key embeddings are greater than the cached sequence length.
                2. Sine and cosine caches are empty.
        """
        if (
            key_len > self._cached_seq_length
            or self._cached_cos is None
            or (self._cached_cos is not None)
            or (self._cached_cos is not None and self._cached_cos.dtype != key_dtype)
            or self._cached_sin is None
            or (self._cached_sin is not None)
            or (self._cached_sin is not None and self._cached_sin.dtype != key_dtype)
        ):
            self._cached_seq_length = max(key_len, self._cached_seq_length)

            # The shape of 'pos_index' is [number of key tokens]
            pos_index = ops.arange(
                self._cached_seq_length,
                dtype=mindspore.float32,
            )
            # The shape of 'pos_index_theta' is [number of key tokens, model dimension]
            pos_index_theta = ops.einsum("i,j->ij", pos_index, self.inv_freq)
            # The shape of 'emb' is [number of key tokens, model dimension]
            emb = ops.cat((pos_index_theta, pos_index_theta), axis=-1)

            # the shape of cos and sin embeddings is [number of key tokens, model_dim]
            cos_emb = emb.cos().to(dtype=key_dtype)
            sin_emb = emb.sin().to(dtype=key_dtype)

            # the shape of cached cos and sin embeddings is [1, 1, number of key tokens, model_dim]
            self._cached_cos = cos_emb[None, None, :, :]
            self._cached_sin = sin_emb[None, None, :, :]

    def construct(
        self,
        query: mindspore.Tensor,
        key: mindspore.Tensor,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        The forward function of RoPE embeddings.

        Args:
            query: Query embeddings in the transformer model. The shape of query embeddings is
                [Batch, number of query heads, number of query tokens, model dimension].
            key: Key embeddings in the transformer model. The shape of key embeddings is
                [Batch, number of key heads, number of key tokens, model dimension].

        Returns:
            A tuple containing the query and key embeddings with positional information. The shape of the returned query
            and key embeddings is the same as the input query and key embeddings respectively.

        ...note:
            The RoPE embedding computation is done in full-precision. After the computation, input query and key tensors
            are casted to original input datatype.
        """
        dim = key.shape[-1]
        key_len = key.shape[2]
        query_len = query.shape[2]

        assert dim == self.model_dim
        assert key.dtype == query.dtype

        # In the context of self-attention, the lengths of keys and queries are equal.
        # However, in generation tasks, such as predicting the next token in a sequence, the lengths of keys and queries
        # can differ. For instance, when employing key-value (KV) caching for sequence prediction, the keys
        # represent embeddings of previous tokens and the current token, while the query corresponds
        # to the embedding of the current token only.
        assert (
            key_len >= query_len
        ), "Number of keys has to be greater than or equal to number of queries."

        query_float = query.float()
        key_float = key.float()

        self._compute_sin_cos_embeddings(
            key_len, key_dtype=key_float.dtype
        )
        query_float = _apply_rotary_pos_emb(
            x=query_float,
            pos_sin=self._cached_sin[..., key_len - query_len : key_len, :],
            pos_cos=self._cached_cos[..., key_len - query_len : key_len, :],
        )
        key_float = _apply_rotary_pos_emb(
            x=key_float,
            pos_sin=self._cached_sin[..., :key_len, :],
            pos_cos=self._cached_cos[..., :key_len, :],
        )

        return query_float.type_as(query), key_float.type_as(key)


class OpenELMMultiHeadCausalAttention(nn.Cell):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Dense(
            in_channels=config.model_dim,
            out_channels=(q_heads + k_heads + v_heads) * head_dim,
            has_bias=False,
        )

        self.pos_embedding = OpenELMRotaryEmbedding(
            model_dim=config.head_dim,
            max_seq_length=config.rope_max_length,
            freq_constant=config.rope_freq_constant,
        )

        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
            self.k_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Dense(
            in_channels=q_heads * head_dim,
            out_channels=config.model_dim,
            has_bias=False,
        )

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"query_heads={self.num_q_heads}, key_heads={self.num_k_heads}, value_heads={self.num_v_heads}"
        )

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[mindspore.Tensor] = None,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """
        Forward pass of multi-head self-attention.

        Args:
            hidden_states: Input tensor of the shape [batch size, sequence length, model dimension].
            past_key_value: Tensor storing the cached keys and values.
            output_attentions: output attention weights.
            use_cache: Specifies whether to use kv-cache for generation.
            cache_position: used for updating the kv-cache.

        Returns:
            The output of the same shape as the input, optionally with a tensor containing cached keys and values.
        """

        # scaled_dot_product_attention does not return attention weights, set output_attentions to False
        output_attentions = False
        batch_size, seq_length, d_model = hidden_states.shape

        # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
        qkv = self.qkv_proj(hidden_states)
        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
        qkv = qkv.reshape(
            batch_size,
            seq_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
        qkv = qkv.swapaxes(1, 2)
        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
        queries, keys, values = qkv.split(
            [self.num_q_heads, self.num_k_heads, self.num_v_heads], axis=1
        )

        if self.q_norm is not None:
            queries = self.q_norm(queries)

        if self.k_norm is not None:
            keys = self.k_norm(keys)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            cache_kwargs = {"cache_position": cache_position}
            keys, values = past_key_value.update(
                keys, values, self.layer_idx, cache_kwargs
            )

        # Add positional embedding
        queries, keys = self.pos_embedding(queries, keys)

        if self.num_groups != 1:
            # GQA
            # [B, k_h, S, h] --> [B, q_h, S, h]
            keys = keys.repeat_interleave(self.num_groups, dim=1)
            # [B, v_h, S, h] --> [B, q_h, S, h]
            values = values.repeat_interleave(self.num_groups, dim=1)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : keys.shape[-2]]

        attn_output, _ = _scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=causal_mask,
            dropout_p=0,
            is_causal=False,
            is_training=self.training,
            dtype=queries.dtype
        )

        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class OpenELMFeedForwardNetwork(nn.Cell):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = nn.Dense(
                in_channels=config.model_dim,
                out_channels=2 * intermediate_dim,
                has_bias=False,
            )
            self.proj_2 = nn.Dense(
                in_channels=intermediate_dim,
                out_channels=config.model_dim,
                has_bias=False,
            )
            self.ffn_with_glu = True
        else:
            # Standard FFN, as described in https://arxiv.org/abs/1706.03762
            self.proj_1 = nn.Dense(
                in_channels=config.model_dim,
                out_channels=intermediate_dim,
                has_bias=False,
            )
            self.proj_2 = nn.Dense(
                in_channels=intermediate_dim,
                out_channels=config.model_dim,
                has_bias=False,
            )
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def extra_repr(self) -> str:
        return super().extra_repr() + f"(ffn_with_glu) : {self.ffn_with_glu}"

    def construct(self, x: Tensor) -> Tensor:
        """Forward function of FFN layer.

        Args:
            x: Input tensor of the shape [batch size, sequence length, model dimension].

        Returns:
            A tensor of the same shape as the input.
        """
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = y_12.chunk(2, axis=-1)
            y = self.act(y_1) * y_2
            return self.proj_2(y)
        else:
            return self.proj_2(self.act(self.proj_1(x)))


class OpenELMDecoderLayer(nn.Cell):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = OpenELMMultiHeadCausalAttention(config=config, layer_idx=layer_idx)
        self.ffn = OpenELMFeedForwardNetwork(config=config, layer_idx=layer_idx)
        self.ffn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )
        self.attn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[mindspore.Tensor] = None,
        **kwargs,
    ) -> Tuple[
        mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]
    ]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(mindspore.Tensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OpenELMModel(OpenELMPreTrainedModel):
    config_class = OpenELMConfig

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.config = config

        self.token_embeddings = nn.Embedding(
            embedding_size=config.model_dim,
            vocab_size=config.vocab_size,
        )

        self.layers = nn.CellList([
            OpenELMDecoderLayer(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_transformer_layers)]
        )
        self.norm = OpenELMRMSNorm(num_features=config.model_dim)
        if config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = nn.Dense(
                in_channels=config.model_dim,
                out_channels=config.vocab_size,
                has_bias=False,
            )
        self.num_transformer_layers = config.num_transformer_layers
        self.gradient_checkpointing = False

        # Register a causal mask to separate causal and padding mask creation. Merging happens in the attention class.
        # NOTE: This is not friendly with TorchScript, ONNX, ExportedProgram serialization for very large `max_context_length`.
        causal_mask = ops.full(
            (config.max_context_length, config.max_context_length),
            fill_value=True,
            dtype=mindspore.bool_,
        )
        self.causal_mask = ops.triu(causal_mask, diagonal=1)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings: mindspore.Tensor):
        self.token_embeddings = new_embeddings

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = ops.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = ops.full(
                (2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]),
                fill_value=1,
            )
            self.causal_mask = ops.triu(causal_mask, diagonal=1)

        # We use the current dtype to avoid any overflows
        min_dtype = finfo(dtype, 'min')
        causal_mask = (
            self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype)
            * min_dtype
        )

        causal_mask = causal_mask.to(dtype=dtype)
        if attention_mask is not None and attention_mask.ndim == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) & attention_mask[
                :, None, None, :
            ].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        return causal_mask


class OpenELMForCausalLM(OpenELMPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.transformer = OpenELMModel(config)
        self.vocab_size = config.vocab_size
        if config.share_input_output_layers:
            self.lm_head = None
        else:
            self.lm_head = nn.Dense(config.model_dim, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.token_embeddings

    def set_input_embeddings(self, value):
        self.transformer.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.lm_head is None:
            # shared
            logits = ops.dense(
                hidden_states, weight=self.transformer.token_embeddings.weight
            )
        else:
            logits = self.lm_head(hidden_states)
        logits = logits[:, : self.config.vocab_size]
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = ops.cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if self.generation_config.cache_implementation == "static":
            # generation with static cache
            cache_position = kwargs.get("cache_position", None)
            if cache_position is None:
                past_length = 0
            else:
                past_length = cache_position[-1] + 1
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = ops.arange(
            past_length,
            past_length + position_ids.shape[-1],
        )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # We could use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past
                ),
            )
        return reordered_past

__all__ = [
    'OpenELMForCausalLM',
    'OpenELMModel',
    'OpenELMPreTrainedModel'
]
