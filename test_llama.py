import mindspore as ms
ms.set_context(mode=0)

from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('path')
tokenizer = AutoTokenizer.from_pretrained('path')
tokenizer.padding_side = 'right'

prompt = 'Hey, who are you?'
inputs = tokenizer(prompt, return_tensors='ms', padding='max_length', max_length=1024)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

input_embeds = model.get_input_embeddings()(input_ids)
eos_token_id = tokenizer.convert_tokens_to_ids('\n')

generation_config = dict(
    num_beams=1,
    max_new_tokens=20,
    do_sample=False,
    eos_token_id=eos_token_id
)

generate_ids = model.generate(inputs_embeds=input_embeds,
                              attention_mask=attention_mask,
                              output_hidden_states=False,
                              return_dict=False,
                              use_cache=True,
                              **generation_config)
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_np_tokenization_spaces=False)[0]

print(outputs)
