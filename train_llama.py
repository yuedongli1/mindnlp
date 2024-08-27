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
            attention_mask=ops.ones_like(inputs["input_ids"]).bool(),
            labels=inputs["input_ids"],
            return_dict=False
        )[0]

model = AutoModelForCausalLM.from_pretrained("path")
tokenizer = AutoTokenizer.from_pretrained("path", add_prefix_space=True)

dataset = load_dataset('tatsu-lab/alpaca')
print(dataset.get_col_names())

class ModifiedMapFunction(BaseMapFuction):
    def __call__(self, text):
        tokenized = tokenizer(text, max_length=512, padding="max_length", truncation=True)
        return tokenized['input_ids']

training_args = TrainingArguments(
    "output",
    fp16=False,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=2,
    logging_steps=100,
    save_strategy='epoch'
)

trainer = ModifiedTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    map_fn=ModifiedMapFunction('text', 'input_ids'),
)

trainer.train()
