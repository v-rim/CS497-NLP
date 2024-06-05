# Load model directly
import os
from transformers import RobertaConfig, DataCollatorForLanguageModeling, \
                        Trainer, TrainingArguments
from BitNetModel import BitnetForCausalLM
from BitNetTokenizer import BitnetTokenizer
from datasets import load_dataset

def get_datasets(tokenizer):
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split="train[:111607]",
        num_proc = os.cpu_count()
    )
    dataset = dataset.train_test_split(test_size=0.3)

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["content"]], padding=True, truncation=True, max_length = 128)

    tokenized = dataset.map(
        preprocess_function, batched=True, remove_columns=dataset["train"].column_names, num_proc=os.cpu_count()
    )

    return tokenized

    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= block_size:
    #         total_length = (total_length // block_size) * block_size
    #     # Split by chunks of block_size.
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result

    # grouped = tokenized.map(group_texts, batched=True, num_proc=os.cpu_count())

    # return grouped

def finetune():
    tokenizer = BitnetTokenizer.from_pretrained('1bitLLM/bitnet_b1_58-large', use_fast=False)
    model = BitnetForCausalLM.from_pretrained('1bitLLM/bitnet_b1_58-large')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    datasets = get_datasets(tokenizer)
    print(datasets)

    training_args = TrainingArguments(
        output_dir="./BitNet",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_accumulation_steps=1,
        num_train_epochs=10,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model("./BitNet")

if __name__ == '__main__':
    finetune()