import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor, nn, ops


class KVCacheMgr(nn.Cell):
    """KVCache Manager."""

    def __init__(
        self,
        n_head,
        head_dim,
        max_batch_size=8,
        max_seq_length=4096,
        use_kvcache_op=True,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.use_kvcache_op = use_kvcache_op
        self.is_first_iteration = True

        self.cache_length_tensor = Tensor([max_batch_size * max_seq_length], dtype=mstype.int32)
        self.cache_pad_tensor = Tensor([3], dtype=mstype.int64)
        self.seq_length_tensor = Tensor([max_seq_length], dtype=mstype.int32)
        self.seq_length_tensor_pad = Tensor([max_seq_length, 3], dtype=mstype.int64)
        self.seqlen_axis_tensor_pad = Tensor([2, 3], dtype=mstype.int64)
        self.pad_before = Tensor([0, 0, 0, 0, 0], mstype.int32)
        self.pad_after = Tensor([0, 0], mstype.int32)
        self.pad_zero = Tensor(0.0)

        if self.use_kvcache_op:
            # pylint: disable=W0212
            self.prompt_kvcache = ops.operations._inner_ops.PromptKVCache()
            # pylint: disable=W0212
            self.decoder_kvcache = ops.operations._inner_ops.DecoderKVCache()
        else:
            self.add = ops.Add()
            self.mul = ops.Mul()
            self.assign = ops.Assign()
        self.concat = ops.Concat(axis=0)
        self.sub = ops.Sub()
        self.div = ops.Div()
        self.pad = ops.PadV3()
        self.slice = ops.StridedSlice()
        self.cast = ops.Cast()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape().add_prim_attr("skip_redistribution", True)

        kv_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.key_past = Parameter(Tensor(np.zeros(kv_shape), ms.float32), name="key_past", requires_grad=False)
        self.value_past = Parameter(Tensor(np.zeros(kv_shape), ms.float32), name="value_past", requires_grad=False)

    def padding(self, key, value, seq_length):
        """padding key, value"""
        pad_length = self.sub(self.seq_length_tensor, seq_length)
        # calculate padding parameter: (0, 0),(0,0),(0,pad_length),(0,0), append values of 'pad_length' in axis
        pad_config = self.concat((self.pad_before, pad_length, self.pad_after))
        key_padding = self.pad(key, pad_config, self.pad_zero)
        value_padding = self.pad(value, pad_config, self.pad_zero)
        return key_padding, value_padding

    def auto_caching(self, key_update, value_update, batch_valid_length, seq_length_tensor_pad, batch_index_pad=None):
        """use kvcache op to cache key, value"""
        # key_update shape: [real_bs, n_head, max_seqlen, head_dim]
        if self.is_first_iteration:
            batch_valid_length = batch_valid_length * 0
            self.prompt_kvcache(
                self.key_past,
                key_update,
                batch_valid_length,
                batch_index_pad,
                self.seqlen_axis_tensor_pad,
                seq_length_tensor_pad,
                seq_length_tensor_pad,
            )
            self.prompt_kvcache(
                self.value_past,
                value_update,
                batch_valid_length,
                batch_index_pad,
                self.seqlen_axis_tensor_pad,
                seq_length_tensor_pad,
                seq_length_tensor_pad,
            )
            return None

        key_cache = self.key_past
        value_cache = self.value_past
        self.decoder_kvcache(
            self.key_past,
            key_update,
            batch_valid_length,
            batch_index_pad,
            self.seqlen_axis_tensor_pad,
            seq_length_tensor_pad,
            seq_length_tensor_pad,
        )
        self.decoder_kvcache(
            self.value_past,
            value_update,
            batch_valid_length,
            batch_index_pad,
            self.seqlen_axis_tensor_pad,
            seq_length_tensor_pad,
            seq_length_tensor_pad,
        )
        key_cache = ops.depend(key_cache, key_update)
        value_cache = ops.depend(value_cache, value_update)
        return key_cache, value_cache

    def manual_caching(self, key_update, value_update, valid_length_vector):
        """use assign to cache key, value"""
        # key_update shape: [real_bs, n_head, 1, head_dim]
        if self.is_first_iteration:
            self.assign(self.key_past, self.mul(key_update, valid_length_vector))
            self.assign(self.value_past, self.mul(value_update, valid_length_vector))
            return None

        key = self.add(self.key_past, self.mul(key_update, valid_length_vector))
        value = self.add(self.value_past, self.mul(value_update, valid_length_vector))
        self.assign(self.key_past, key)
        self.assign(self.value_past, value)
        # key shape: [real_bs, n_head, max_cache_len // real_bs, head_dim]
        return key, value

    def construct(self, key, value, kvcache_inputs=None):
        """The forward compute of KVCacheMgr."""
        # TODO: add inputs check
        batch_valid_length, zactivate_len, batch_index_pad, seq_length_tensor_pad = kvcache_inputs
        batch_size, _, seq_length, _ = ops.shape(key)

        if self.use_kvcache_op:
            new_kv_with_cache = self.auto_caching(key, value, batch_valid_length, seq_length_tensor_pad, batch_index_pad)
        else:
            new_kv_with_cache = self.manual_caching(key, value, batch_valid_length)
        #  for the 2nd+ iteration, cached key_value are loaded and concat with current key_value
        if not self.is_first_iteration:
            return new_kv_with_cache
        # for the first iteration, current key_value are cached in self.key_past/value_past
        return key, value


class KVCachePreprocess(nn.Cell):
    """KVCache Manager."""

    def __init__(
        self,
        max_seq_length=4096,
        use_kvcache_op=False,
    ):
        super().__init__()
        self.use_kvcache_op = use_kvcache_op
        self.max_cache_length = max_seq_length
        self.range = Tensor(np.arange(max_seq_length).reshape((1, 1, -1)), mstype.int32)
        self.seq_length_tensor = Tensor([max_seq_length], dtype=mstype.int32)
        self.seq_length_tensor_pad = Tensor([max_seq_length, 3], dtype=mstype.int64)
        self.is_first_iteration = True

    def construct(self, batch_size, batch_valid_length, zactivate_len=None):
        """precompute kvcache inputs"""
        seq_range = self.range

        if self.use_kvcache_op:
            batch_index = ops.arange(0, batch_size, 1)
            batch_index_pad = ops.concat((batch_index, Tensor([3], dtype=mstype.int64)))
            seq_length_tensor_pad = Tensor([batch_size * self.max_seq_length, 3], dtype=mstype.int64)
            batch_valid_length = ops.cast(ops.reshape(batch_valid_length, (-1,)), mstype.int64)
            kvcache_inputs = (batch_valid_length, zactivate_len, batch_index_pad, seq_length_tensor_pad)
        else:
            if self.is_first_iteration:
                valid_length_vector = ops.less(seq_range, ops.reshape(batch_valid_length, (-1, 1, 1)))
            else:
                valid_length_vector = ops.equal(seq_range, ops.reshape(batch_valid_length, (-1, 1, 1)))
            valid_length_vector = ops.expand_dims(valid_length_vector, 3)
            kvcache_inputs = (valid_length_vector, zactivate_len, None, None)
        return kvcache_inputs
