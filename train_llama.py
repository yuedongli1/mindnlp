from datasets import load_dataset

import mindspore as ms
ms.set_context(mode=0)
from mindspore import ops

from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.engine import TrainingArguments, Trainer
from mindnlp.dataset import BaseMapFuction

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"].bool(),
            labels=inputs["input_ids"],
            return_dict=False
        )[0]

model = AutoModelForCausalLM.from_pretrained("path")
tokenizer = AutoTokenizer.from_pretrained("path", add_prefix_space=True)

dataset = load_dataset('codyburker/yelp_review_sampled')

tokenized_datasets = dataset.map(lambda  x: {'sample': tokenizer(
                        x['text'], padding='max_length', truncation=True, max_length=512)}, batched=False)

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
    column_name_collate=['attention_mask', 'input_ids']
)

trainer = ModifiedTrainer(
    model=model,
    train_dataset=small_train_dataset,
    args=training_args,
)

trainer.train()
