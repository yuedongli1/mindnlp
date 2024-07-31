# coding=utf-8
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Testing suite for the Mindspore Pop2Piano8 model. """

import copy
import tempfile
import unittest

import numpy as np
from datasets import load_dataset

from mindnlp.transformers import Pop2PianoConfig
from mindnlp.transformers.feature_extraction_utils import BatchFeature
from mindnlp.utils.testing_utils import (
    require_mindspore,
    require_essentia,
    require_librosa,
    require_scipy,
    slow,
)
from mindnlp.utils.testing_utils import is_mindspore_available, is_essentia_available, is_librosa_available, is_scipy_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor

if is_mindspore_available():
    import mindspore
    from mindspore import ops
    
    from mindnlp.transformers.models.pop2piano.modeling_pop2piano import Pop2PianoForConditionalGeneration
    from mindnlp.transformers.models.pop2piano.modeling_pop2piano import POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST


@require_mindspore
class Pop2PianoModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=9,
        # For common tests
        is_training=False,
        use_attention_mask=True,
        use_labels=True,
        hidden_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
        scope=None,
        decoder_layers=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.scope = None
        self.decoder_layers = decoder_layers

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = (
            ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size) if self.use_labels else None
        )
        lm_labels = lm_labels.astype("int32")

        return self.get_config(), input_ids, decoder_input_ids, attention_mask, decoder_attention_mask, lm_labels

    def get_pipeline_config(self):
        return Pop2PianoConfig(
            vocab_size=166,  # Pop2Piano forces 100 extra tokens
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def get_config(self):
        return Pop2PianoConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def check_prepare_lm_labels_via_shift_left(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = Pop2PianoForConditionalGeneration(config=config)
        # model.to(torch_device)
        model.set_train(False)

        # make sure that lm_labels are correctly padded from the right
        lm_labels.masked_fill((lm_labels == self.decoder_start_token_id), self.eos_token_id)

        # add causal pad token mask
        triangular_mask = ops.tril(lm_labels.new_ones(lm_labels.shape)).logical_not()
        lm_labels.masked_fill(triangular_mask, self.pad_token_id)
        decoder_input_ids = model._shift_right(lm_labels)

        for i, (decoder_input_ids_slice, lm_labels_slice) in enumerate(zip(decoder_input_ids, lm_labels)):
            # first item
            self.parent.assertEqual(decoder_input_ids_slice[0].item(), self.decoder_start_token_id)
            if i < decoder_input_ids_slice.shape[-1]:
                if i < decoder_input_ids.shape[-1] - 1:
                    # items before diagonal
                    self.parent.assertListEqual(
                        decoder_input_ids_slice[1 : i + 1].tolist(), lm_labels_slice[:i].tolist()
                    )
                # pad items after diagonal
                if i < decoder_input_ids.shape[-1] - 2:
                    self.parent.assertListEqual(
                        decoder_input_ids_slice[i + 2 :].tolist(), lm_labels_slice[i + 1 : -1].tolist()
                    )
            else:
                # all items after square
                self.parent.assertListEqual(decoder_input_ids_slice[1:].tolist(), lm_labels_slice[:-1].tolist())

    def create_and_check_model(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = Pop2PianoForConditionalGeneration(config=config)
        model.set_train(False)
        result = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        result = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        decoder_past = result.past_key_values
        encoder_output = result.encoder_last_hidden_state

        self.parent.assertEqual(encoder_output.shape, (self.batch_size, self.encoder_seq_length, self.hidden_size))
        # There should be `num_layers` key value embeddings stored in decoder_past
        self.parent.assertEqual(len(decoder_past), config.num_layers)
        # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past tuple
        self.parent.assertEqual(len(decoder_past[0]), 4)

    def create_and_check_with_lm_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = Pop2PianoForConditionalGeneration(config=config).set_train(False)
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].shape, (self.batch_size, self.decoder_seq_length, self.vocab_size))
        self.parent.assertEqual(outputs["loss"].shape, ())

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = Pop2PianoForConditionalGeneration(config=config).get_decoder().set_train(False)
        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = ops.cat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        self.parent.assertTrue(np.allclose(output_from_past_slice.asnumpy(), output_from_no_past_slice.asnumpy(), atol=1e-3))

    def create_and_check_decoder_model_attention_mask_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = Pop2PianoForConditionalGeneration(config=config).get_decoder()
        model.set_train(False)

        # create attention mask
        attn_mask = ops.ones(input_ids.shape, dtype=mindspore.int64)

        half_seq_length = input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past_key_values = model(input_ids, attention_mask=attn_mask, use_cache=True).to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = ops.cat([input_ids, next_tokens], axis=-1)
        attn_mask = ops.cat(
            [attn_mask, ops.ones((attn_mask.shape[0], 1), dtype=mindspore.int64)],
            axis=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values, attention_mask=attn_mask)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        self.parent.assertTrue(np.allclose(output_from_past_slice.asnumpy(), output_from_no_past_slice.asnumpy(), atol=1e-3))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = Pop2PianoForConditionalGeneration(config=config).get_decoder().set_train(False)
        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = ops.cat([input_ids, next_tokens], axis=-1)
        next_attention_mask = ops.cat([attention_mask, next_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(np.allclose(output_from_past_slice.asnumpy(), output_from_no_past_slice.asnumpy(), atol=1e-3))

    def create_and_check_generate_with_past_key_values(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = Pop2PianoForConditionalGeneration(config=config).set_train(False)
        # torch.manual_seed(0)
        output_without_past_cache = model.generate(
            input_ids[:1], num_beams=2, max_length=5, do_sample=True, use_cache=False
        )
        # torch.manual_seed(0)
        output_with_past_cache = model.generate(input_ids[:1], num_beams=2, max_length=5, do_sample=True)
        self.parent.assertTrue(ops.all(output_with_past_cache == output_without_past_cache))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = Pop2PianoForConditionalGeneration(config=config).half().set_train(False)
        output = model(input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask)[
            "encoder_last_hidden_state"
        ]
        self.parent.assertFalse(ops.isnan(output).any().item())

    def create_and_check_encoder_decoder_shared_weights(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        for model_class in [Pop2PianoForConditionalGeneration]:
            # torch.manual_seed(0)
            model = model_class(config=config).set_train(False)
            # load state dict copies weights but does not tie them
            mindspore.load_param_into_net(model.encoder, model.decoder.parameters_dict(), \
                                          strict_load=False)

            # torch.manual_seed(0)
            tied_config = copy.deepcopy(config)
            tied_config.tie_encoder_decoder = True
            tied_model = model_class(config=tied_config).set_train(False)

            model_result = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )

            tied_model_result = tied_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )

            # check that models has less parameters
            self.parent.assertLess(
                tied_model.num_parameters(), model.num_parameters()
            )
            random_slice_idx = ids_tensor((1,), model_result[0].shape[-1]).item()

            # check that outputs are equal
            self.parent.assertTrue(
                np.allclose(
                    model_result[0][0, :, random_slice_idx].asnumpy(),
                    tied_model_result[0][0, :, random_slice_idx].asnumpy(),
                    atol=1e-4
                )
            )

            # check that outputs after saving and loading are equal
            with tempfile.TemporaryDirectory() as tmpdirname:
                tied_model.save_pretrained(tmpdirname)
                tied_model = model_class.from_pretrained(tmpdirname)
                # tied_model.to(torch_device)
                tied_model.set_train(False)

                # check that models has less parameters
                self.parent.assertLess(
                    tied_model.num_parameters(), model.num_parameters()
                )
                random_slice_idx = ids_tensor((1,), model_result[0].shape[-1]).item()

                tied_model_result = tied_model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                )

                # check that outputs are equal
                self.parent.assertTrue(
                    np.allclose(
                        model_result[0][0, :, random_slice_idx].asnumpy(),
                        tied_model_result[0][0, :, random_slice_idx].asnumpy(),
                        atol=1e-4,
                    )
                )

    def check_resize_embeddings_pop2piano_v1_1(
        self,
        config,
    ):
        prev_vocab_size = config.vocab_size

        config.tie_word_embeddings = False
        model = Pop2PianoForConditionalGeneration(config=config).set_train(False)
        model.resize_token_embeddings(prev_vocab_size - 10)

        self.parent.assertEqual(model.get_input_embeddings().weight.shape[0], prev_vocab_size - 10)
        self.parent.assertEqual(model.get_output_embeddings().weight.shape[0], prev_vocab_size - 10)
        self.parent.assertEqual(model.config.vocab_size, prev_vocab_size - 10)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": False,
        }
        return config, inputs_dict


@require_mindspore
class Pop2PianoModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Pop2PianoForConditionalGeneration,) if is_mindspore_available() else ()
    all_generative_model_classes = ()
    pipeline_model_mapping = (
        {"automatic-speech-recognition": Pop2PianoForConditionalGeneration} if is_mindspore_available() else {}
    )
    all_parallelizable_model_classes = ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = True
    test_model_parallel = False
    is_encoder_decoder = True

    def setUp(self):
        self.model_tester = Pop2PianoModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Pop2PianoConfig, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_shift_right(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_prepare_lm_labels_via_shift_left(*config_and_inputs)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_v1_1(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        # check that gated gelu feed forward and different word embeddings work
        config = config_and_inputs[0]
        config.tie_word_embeddings = False
        config.feed_forward_proj = "gated-gelu"
        self.model_tester.create_and_check_model(config, *config_and_inputs[1:])

    def test_config_and_model_silu_gated(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        config.feed_forward_proj = "gated-silu"
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_lm_head(*config_and_inputs)

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    def test_decoder_model_past_with_attn_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_attention_mask_past(*config_and_inputs)

    def test_decoder_model_past_with_3d_attn_mask(self):
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = self.model_tester.prepare_config_and_inputs()

        attention_mask = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.encoder_seq_length, self.model_tester.encoder_seq_length],
            vocab_size=2,
        )
        decoder_attention_mask = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.decoder_seq_length, self.model_tester.decoder_seq_length],
            vocab_size=2,
        )

        self.model_tester.create_and_check_decoder_model_attention_mask_past(
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_encoder_decoder_shared_weights(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encoder_decoder_shared_weights(*config_and_inputs)

    # @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    def test_v1_1_resize_embeddings(self):
        config = self.model_tester.prepare_config_and_inputs()[0]
        self.model_tester.check_resize_embeddings_pop2piano_v1_1(config)

    @slow
    def test_model_from_pretrained(self):
        for model_name in POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Pop2PianoForConditionalGeneration.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_pass_with_input_features(self):
        input_features = BatchFeature(
            {
                "input_features": ops.rand((75, 100, 512)).type(mindspore.float32),
                "beatsteps": ops.randint(size=(1, 955), low=0, high=100).type(mindspore.float32),
                "extrapolated_beatstep": ops.randint(size=(1, 900), low=0, high=100).type(mindspore.float32),
            }
        )
        model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
        model_opts = model.generate(input_features=input_features["input_features"], return_dict_in_generate=True)

        self.assertEqual(model_opts.sequences.ndim, 2)

    def test_pass_with_batched_input_features(self):
        input_features = BatchFeature(
            {
                "input_features": ops.rand((220, 70, 512)).type(mindspore.float32),
                "beatsteps": ops.randint(size=(5, 955), low=0, high=100).type(mindspore.float32),
                "extrapolated_beatstep": ops.randint(size=(5, 900), low=0, high=100).type(mindspore.float32),
                "attention_mask": ops.cat(
                    [
                        ops.ones((120, 70), dtype=mindspore.int32),
                        ops.zeros((1, 70), dtype=mindspore.int32),
                        ops.ones((50, 70), dtype=mindspore.int32),
                        ops.zeros((1, 70), dtype=mindspore.int32),
                        ops.ones((47, 70), dtype=mindspore.int32),
                        ops.zeros((1, 70), dtype=mindspore.int32),
                    ],
                    axis=0,
                ),
                "attention_mask_beatsteps": ops.ones((5, 955), dtype=mindspore.int32),
                "attention_mask_extrapolated_beatstep": ops.ones((5, 900), dtype=mindspore.int32),
            }
        )
        model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
        model_opts = model.generate(
            input_features=input_features["input_features"],
            attention_mask=input_features["attention_mask"],
            return_dict_in_generate=True,
        )

        self.assertEqual(model_opts.sequences.ndim, 2)


@require_mindspore
class Pop2PianoModelIntegrationTests(unittest.TestCase):
    @slow
    def test_mel_conditioner_integration(self):
        composer = "composer1"
        model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
        input_embeds = ops.ones((10, 100, 512))

        composer_value = model.generation_config.composer_to_feature_token[composer]
        composer_value = mindspore.tensor(composer_value)
        composer_value = composer_value.repeat(input_embeds.shape[0])
        outputs = model.mel_conditioner(
            input_embeds, composer_value, min(model.generation_config.composer_to_feature_token.values())
        )

        # check shape
        self.assertEqual(outputs.shape, (10, 101, 512))

        # check values
        EXPECTED_OUTPUTS = mindspore.tensor(
            [[1.0475305318832397, 0.29052114486694336, -0.47778210043907166], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        )

        self.assertTrue(np.allclose(outputs[0, :3, :3].asnumpy(), EXPECTED_OUTPUTS.asnumpy(), atol=1e-4))

    @slow
    @require_essentia
    @require_librosa
    @require_scipy
    def test_full_model_integration(self):
        if is_librosa_available() and is_scipy_available() and is_essentia_available() and is_mindspore_available():
            from mindnlp.transformers import Pop2PianoProcessor

            speech_input1 = np.zeros([1_000_000], dtype=np.float32)
            sampling_rate = 44_100

            processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
            input_features = processor.feature_extractor(
                speech_input1, sampling_rate=sampling_rate, return_tensors="ms"
            )

            model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
            outputs = model.generate(
                input_features=input_features["input_features"], return_dict_in_generate=True
            ).sequences

            # check for shapes
            self.assertEqual(outputs.shape[0], 70)

            # check for values
            self.assertEqual(outputs[0, :2].asnumpy().tolist(), [0, 1])

    # This is the test for a real music from K-Pop genre.
    @slow
    @require_essentia
    @require_librosa
    @require_scipy
    def test_real_music(self):
        if is_librosa_available() and is_scipy_available() and is_essentia_available() and is_mindspore_available():
            from mindnlp.transformers import Pop2PianoFeatureExtractor, Pop2PianoTokenizer

            model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
            model.set_train(False)
            feature_extractor = Pop2PianoFeatureExtractor.from_pretrained("sweetcocoa/pop2piano")
            tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")
            ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")

            output_fe = feature_extractor(
                ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"], return_tensors="ms"
            )
            output_model = model.generate(input_features=output_fe["input_features"], composer="composer1")
            output_tokenizer = tokenizer.batch_decode(token_ids=output_model, feature_extractor_output=output_fe)
            pretty_midi_object = output_tokenizer["pretty_midi_objects"][0]

            # Checking if no of notes are same
            self.assertEqual(len(pretty_midi_object.instruments[0].notes), 59)
            predicted_timings = []
            for i in pretty_midi_object.instruments[0].notes:
                predicted_timings.append(i.start)

            # Checking note start timings(first 6)
            EXPECTED_START_TIMINGS = [
                0.4876190423965454,
                0.7314285635948181,
                0.9752380847930908,
                1.4396371841430664,
                1.6718367338180542,
                1.904036283493042,
            ]

            np.allclose(EXPECTED_START_TIMINGS, predicted_timings[:6])

            # Checking note end timings(last 6)
            EXPECTED_END_TIMINGS = [
                12.341403007507324,
                12.567797183990479,
                12.567797183990479,
                12.567797183990479,
                12.794191360473633,
                12.794191360473633,
            ]

            np.allclose(EXPECTED_END_TIMINGS, predicted_timings[-6:])
