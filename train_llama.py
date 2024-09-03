from datasets import load_dataset

import mindspore as ms
ms.set_context(mode=0)

from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.engine import TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/llama-7b", add_prefix_space=True)

dataset = load_dataset('codyburker/yelp_review_sampled')


def tokenize_function(x):
    y = tokenizer(x['text'], padding='max_length', truncation=True, max_length=512)
    y['labels'] = y['input_ids']
    return {'sample': y}


tokenized_datasets = dataset.map(tokenize_function, batched=False)

small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))

training_args = TrainingArguments(
    "output",
    fp16=False,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=2,
    logging_steps=100,
    save_strategy='epoch',
    use_parallel=False,
    dataset_drop_last=True,
    remove_unused_columns=True,
    column_name_collate=['attention_mask', 'input_ids', 'labels']
)

trainer = Trainer(
    model=model,
    train_dataset=small_train_dataset,
    args=training_args,
)

trainer.train()
