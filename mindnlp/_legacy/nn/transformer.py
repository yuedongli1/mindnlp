# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Transformer modules."""
import copy
from typing import Optional
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, XavierUniform
from mindnlp._legacy.initializer import XavierNormal
from mindnlp._legacy.functional import multi_head_attention_forward, relu, gelu
from .dropout import Dropout


class Linear(nn.Dense):
    """inner Linear."""

class MultiheadAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in the paper `Attention is all you need
    <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector with source length, and the
    key and value vector with target length, the attention will be performed as the following

    .. math::
            MultiHeadAttention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

    where :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`. The default is with a bias.

    if query, key and value tensor is same, then it will be self attention.

    Args:
        embed_dim (int): Total dimension of MultiheadAttention.
        num_heads (int): Number of attention heads. Note that `embed_dim` will be split
            across `num_heads` (i.e. each head will have dimension `embed_dim // num_heads`).
        dropout (float): Dropout probability of `attn_output_weights`. Default: ``0.0``.
        has_bias (bool): Whether adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv (bool): Whether adds bias to the key and value sequences at axis=0. Default: ``False``.
        add_zero_attn (bool): Whether adds a new batch of zeros to the key and value sequences at axis=1.
            Default: ``False``.
        kdim (int): Total number of features for keys. Default: ``None`` (`kdim=embed_dim`).
        vdim (int): Total number of features for values. Default: ``None`` (`vdim=embed_dim`).
        batch_first (bool): If ``True``, then the input and output shape are (batch, seq, feature),
            else (seq, batch, feature). Default: ``False``.

    Inputs:
        - **query** (Tensor): The query embeddings. If `query` is unbatched, the shape is :math:`(L, E_q)`,
          otherwise the shape is :math:`(L, N, E_q)` when `batch_first=False` or :math:`(N, L, E_q)` when
          `batch_first=True`, where :math:`L`is the target sequence length, :math:`N` is the batch size,
          and :math:`E_q` is the query embedding dimension `embed_dim`. Queries are compared against
          key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        - **key** (Tensor): The key embeddings. If `key` is unbatched, the shape is :math:`(S, E_k)`, otherwise
          the shape is :math:`(S, N, E_k)` when `batch_first=False` or :math:`(N, S, E_k)` when
          `batch_first=True`, where :math:`S` is the source sequence length, :math:`N` is the batch size,
          and :math:`E_k` is the key embedding dimension `kdim`. See "Attention Is All You Need" for more details.
        - **value** (Tensor): The value embeddings. If `value` is unbatched, the shape is :math:`(S, E_v)`,
          otherwise the shape is :math:`(S, N, E_v)` when `batch_first=False` or :math:`(N, S, E_v)` when
          `batch_first=True`, where :math:`S` is the source sequence length, :math:`N` is the batch size,
          and :math:`E_v` is the value embedding dimension `vdim`. See "Attention Is All You Need" for more details.
        - **key_padding_mask** (Tensor, optional): If specified, a mask of shape :math:`(N, S)` indicating which
          elements within `key` to ignore for the purpose of attention (i.e. treat as "padding").
          For unbatched `query`, shape should be :math:`(S)`. Binary and byte masks are supported.
          For a binary mask, a ``True`` value indicates that the corresponding `key` value will be ignored for
          the purpose of attention. For a float mask, it will be directly added to the corresponding `key` value.
        - **need_weights** (bool): Whether returns `attn_output_weights` in addition to `attn_outputs`.
          Default: ``True``.
        - **attn_mask** (Tensor, optional): If specified, a 2D or 3D mask preventing attention to certain positions.
          Must be of shape :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the
          batch size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length.
          A 2D mask will be broadcasted across the batch while a 3D mask allows for a different mask for each entry
          in the batch. Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates
          that the corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that
          the corresponding position is not allowed to attend. For a float mask, the mask values will be added to
          the attention weight.
        - **average_attn_weights** (bool): If true, indicates that the returned `attn_weights` should be averaged
          across heads. Otherwise, `attn_weights` are provided separately per head. Note that this flag only
          has an effect when `need_weights=True`. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        Tuple, a tuple contains(`attn_output`, `attn_output_weights`)

        - **attn_output** - Attention outputs. If input is unbatched, the output shape is:math:`(L, E)`, otherwise
          the output shape is :math:`(L, N, E)` when `batch_first=False` or :math:`(N, L, E)` when
          `batch_first=True`, where :math:`L` is the target sequence length, :math:`N` is the batch size,
          and :math:`E` is the embedding dimension `embed_dim`.
        - **attn_output_weights** - Only returned when `need_weights=True`. If `average_attn_weights=True`,
          returns attention weights averaged across heads with shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)` when input is batched, where :math:`N` is the batch size, :math:`L` is
          the target sequence length, and :math:`S` is the source sequence length.
          If `average_attn_weights=False`, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or
          :math:`(N, \text{num\_heads}, L, S)` when input is batched.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, embed_dim)), 'q_proj_weight')
            self.k_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.kdim)), 'k_proj_weight')
            self.v_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.vdim)), 'v_proj_weight')
            self.in_proj_weight = None
        else:
            self.in_proj_weight = Parameter(initializer(XavierUniform(), (3 * embed_dim, embed_dim)), 'in_proj_weight')
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if bias:
            self.in_proj_bias = Parameter(initializer('zeros', (3 * embed_dim)), 'in_proj_bias')
        else:
            self.in_proj_bias = None
        self.out_proj = Linear(embed_dim, embed_dim, has_bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(initializer(XavierNormal(), (1, 1, embed_dim)), 'bias_k')
            self.bias_v = Parameter(initializer(XavierNormal(), (1, 1, embed_dim)), 'bias_v')
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.k_is_v = False
        self.q_is_k = False

    def __call__(self, *args, **kwargs):
        query = kwargs.get('query', args[0])
        key = kwargs.get('key', args[1])
        value = kwargs.get('value', args[2])
        self.k_is_v = key is value
        self.q_is_k = query is key
        return super().__call__(*args, **kwargs)

    def construct(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                  need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True):
        is_batched = query.ndim == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != mindspore.bool_ and not ops.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")

        if self.batch_first and is_batched:
            # k_is_v and q_is_k preprocess in __call__ since Graph mode do not support `is`
            if self.k_is_v:
                if self.q_is_k:
                    query = key = value = query.swapaxes(1, 0)
                else:
                    query = query.swapaxes(1, 0)
                    key = key.swapaxes(1, 0)
                    value = key
            else:
                query = query.swapaxes(1, 0)
                key = key.swapaxes(1, 0)
                value = value.swapaxes(1, 0)

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k)

        if self.batch_first and is_batched:
            attn_output = attn_output.swapaxes(1, 0)
        if need_weights:
            return attn_output, attn_output_weights
        return (attn_output,)


class TransformerEncoderLayer(nn.Cell):
    r"""
    Transformer Encoder Layer. This is an implementation of the single layer of the transformer
    encoder layer, including multihead attention and feedward layer.

    Args:
        d_model (int): The number of features in the input tensor.
        nhead (int): The number of heads in the MultiheadAttention modules.
        dim_feedforward (int): The dimension of the feedforward layer. Default: ``2048``.
        dropout (float): The dropout value. Default: ``0.1``.
        activation (Union[str, callable, Cell]): The activation function of the intermediate layer,
            can be a string (`"relu"` or `"gelu"`), Cell instance (`nn.ReLU()` or `nn.GELU()`) or
            a callable (`ops.relu` or `ops.gelu`). Default: ``"relu"``.
        layer_norm_eps (float): The epsilon value in LayerNorm modules. Default: ``1e-5``.
        batch_first (bool): If `batch_first = True`, then the shape of input and output tensors is
            (batch, seq, feature), otherwise the shape is (seq, batch, feature). Default: ``False``.
        norm_first (bool): If `norm_first = True`, layer norm is done prior to attention and feedforward
            operations, respectively. Default: ``False``.

    Inputs:
        - **src** (Tensor): the sequence to the encoder layer.
        - **src_mask** (Tensor, optional): the mask for the src sequence. Default: ``None``.
        - **src_key_padding_mask** (Tensor, optional): the mask for the src keys per batch.
          Default: ``None``.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = Tensor(np.random.rand(10, 32, 512), mindspore.float32)
        >>> out = encoder_layer(src)
        >>> # Alternatively, when batch_first=True:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = Tensor(np.random.rand(32, 10, 512), mindspore.float32)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = 'relu',
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(p=dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.norm2 = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.dropout1 = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is relu or isinstance(activation, nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is gelu or isinstance(activation, nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def construct(self, src, src_mask=None, src_key_padding_mask=None):
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != mindspore.bool_ and not ops.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Cell):
    r"""
    Transformer Decoder Layer. This is an implementation of the single layer of the transformer
    decoder layer, including self-attention, cross attention and feedward layer.

    Args:
        d_model (int): The number of expected features in the input tensor.
        nhead (int): The number of heads in the MultiheadAttention modules.
        dim_feedforward (int): The dimension of the feedforward layer. Default: ``2048``.
        dropout (float): The dropout value. Default: ``0.1``.
        activation (Union[str, callable, Cell]): The activation function of the intermediate layer,
            can be a string (`"relu"` or `"gelu"`), Cell instance (`nn.ReLU()` or `nn.GELU()`) or
            a callable (`ops.relu` or `ops.gelu`). Default: ``"relu"``
        layer_norm_eps (float): The epsilon value in LayerNorm modules. Default: ``1e-5``.
        batch_first (bool): If `batch_first = True`, then the shape of input and output tensors is
            (batch, seq, feature), otherwise the shape is (seq, batch, feature). Default: ``False``.
        norm_first (bool): If `norm_first = True`, layer norm is done prior to attention and feedforward
            operations, respectively. Default: ``False``.

    Inputs:
        - **tgt** (Tensor): The sequence to the decoder layer.
        - **memory** (Tensor): The sequence from the last layer of the encoder.
        - **tgt_mask** (Tensor, optional): The mask of the tgt sequence. Default: ``None``.
        - **memory_mask** (Tensor, optional): The mask of the memory sequence. Default: ``None``.
        - **tgt_key_padding_mask** (Tensor, optional): The mask of the tgt keys per batch.
          Default: ``None``.
        - **memory_key_padding_mask** (Tensor, optional): The mask of the memory keys per batch.
          Default: ``None``.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = Tensor(np.random.rand(10, 32, 512), mindspore.float32)
        >>> tgt = Tensor(np.random.rand(20, 32, 512), mindspore.float32)
        >>> out = decoder_layer(tgt, memory)
        >>> # Alternatively, when `batch_first` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = Tensor(np.random.rand(32, 10, 512), mindspore.float32)
        >>> tgt = Tensor(np.random.rand(32, 20, 512), mindspore.float32)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = 'relu',
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(p=dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.norm2 = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.norm3 = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.dropout1 = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)
        self.dropout3 = Dropout(p=dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def construct(self, tgt, memory, tgt_mask = None, memory_mask = None,
                  tgt_key_padding_mask = None, memory_key_padding_mask = None):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerEncoder(nn.Cell):
    r"""
    Transformer Encoder module with multi-layer stacked of `TransformerEncoderLayer`, including multihead self
    attention and feedforward layer. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer (Cell): An instance of the TransformerEncoderLayer() class.
        num_layers (int): The number of encoder-layers in the encoder.
        norm (Cell, optional): The layer normalization module.

    Inputs:
        - **src** (Tensor): The sequence to the encoder.
        - **src_mask** (Tensor, optional): The mask of the src sequence. Default: ``None``.
        - **src_key_padding_mask** (Tensor, optional): the mask of the src keys per batch .
          Default: ``None``.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = Tensor(np.random.rand(10, 32, 512), mindspore.float32)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def construct(self, src: Tensor, src_mask = None, src_key_padding_mask = None):
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != mindspore.bool_ and not ops.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        output = src
        src_key_padding_mask_for_layers = src_key_padding_mask
        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask_for_layers)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Cell):
    r"""
    Transformer Decoder module with multi-layer stacked of `TransformerDecoderLayer`, including multihead self
    attention, cross attention and feedforward layer.

    Args:
        decoder_layer (Cell): An instance of the TransformerDecoderLayer() class.
        num_layers (int): The number of decoder-layers in the decoder.
        norm (Cell): The layer normalization module.

    Inputs:
        - **tgt** (Tensor): The sequence to the decoder.
        - **memory** (Tensor): The sequence from the last layer of the encoder.
        - **tgt_mask** (Tensor, optional): the mask of the tgt sequence. Default: ``None``.
        - **memory_mask** (Tensor, optional): the mask of the memory sequence. Default: ``None``.
        - **tgt_key_padding_mask** (Tensor, optional): the mask of the tgt keys per batch.
          Default: ``None``.
        - **memory_key_padding_mask** (Tensor, optional): the mask of the memory keys per batch.
          Default: ``None``.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = Tensor(np.random.rand(10, 32, 512), mindspore.float32)
        >>> tgt = Tensor(np.random.rand(20, 32, 512), mindspore.float32)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def construct(self, tgt, memory, tgt_mask = None,
                  memory_mask = None, tgt_key_padding_mask = None,
                  memory_key_padding_mask = None):

        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Transformer(nn.Cell):
    r"""
    Transformer module including encoder and decoder. The difference with the original implements is the module use
    the residual addition before the layer normalization. And the default hidden act is `gelu`.
    The details can be found in `Attention is all you need <https://arxiv.org/pdf/1706.03762v5.pdf>`_.

    Args:
        d_model (int): The number of expected features in the inputs tensor. Default: ``512``.
        nhead (int): The number of heads in the MultiheadAttention modules. Default: ``8``.
        num_encoder_layers (int): The number of encoder-layers in the encoder. Default: ``6``.
        num_decoder_layers (int): The number of decoder-layers in the decoder. Default: ``6``.
        dim_feedforward (int): The dimension of the feedforward layer. Default: ``2048``.
        dropout (float): The dropout value. Default: ``0.1``.
        activation (Union[str, callable, Cell]): The activation function of the intermediate layer,
            can be a string (`"relu"` or `"gelu"`), Cell instance (`nn.ReLU()` or `nn.GELU()`) or
            a callable (`ops.relu` or `ops.gelu`). Default: ``"relu"``
        custom_encoder (Cell): Custom encoder. Default: ``None``.
        custom_decoder (Cell): Custom decoder. Default: ``None``.
        layer_norm_eps (float): the epsilion value in layer normalization module. Default: ``1e-5``.
        batch_first (bool): If `batch_first = True`, then the shape of input and output tensors is
            (batch, seq, feature), otherwise the shape is (seq, batch, feature). Default: ``False``.
        norm_first (bool): If `norm_first = True`, layer norm is done prior to attention and feedforward
            operations, respectively. Default: ``False``.

    Inputs:
        - **src** (Tensor): The source sequence to the encoder.
        - **tgt** (Tensor): The target sequence to the decoder.
        - **src_mask** (Tensor, optional): The mask of the src sequence. Default: ``None``.
        - **tgt_mask** (Tensor, optional): The mask of the tgt sequence. Default: ``None``.
        - **memory_mask** (Tensor, optional): The additive mask of the encoder output.
          Default: ``None``.
        - **src_key_padding_mask** (Tensor, optional): The mask of src keys per batch.
          Default: ``None``.
        - **tgt_key_padding_mask** (Tensor, optional): The mask of tgt keys per batch.
          Default: ``None``.
        - **memory_key_padding_mask** (Tensor, optional): The mask of memory keys per batch.
          Default: ``None``.

    Outputs:
        Tensor.

    Examples:
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = Tensor(np.random.rand(10, 32, 512), mindspore.float32)
        >>> tgt = Tensor(np.random.rand(20, 32, 512), mindspore.float32)
        >>> out = transformer_model(src, tgt)
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = 'relu',
                 custom_encoder = None, custom_decoder = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False):
        super().__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first)
            encoder_norm = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first)
            decoder_norm = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def construct(self, src, tgt, src_mask = None, tgt_mask = None,
                memory_mask = None, src_key_padding_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None):

        is_batched = src.ndim == 3
        if not self.batch_first and src.shape[1] != tgt.shape[1] and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if self.batch_first and src.shape[0] != tgt.shape[0] and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.shape[-1] != self.d_model or tgt.shape[-1] != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for _, p in self.parameters_and_names():
            if p.ndim > 1:
                p.set_data(initializer('xavier_uniform', p.shape, p.dtype))

def _get_activation_fn(activation: str):
    if activation == "relu":
        return relu
    if activation == "gelu":
        return gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])


__all__ = [
    'Transformer', 'TransformerEncoder', 'TransformerDecoder',
    'TransformerEncoderLayer', 'TransformerDecoderLayer',
    'MultiheadAttention'
]
