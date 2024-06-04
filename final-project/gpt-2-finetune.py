from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from transformers import DataCollatorForLanguageModeling


def train_tokenizer():
    # Iterator for Training
    def batch_iterator(dataset, batch_size=8):
        for _ in tqdm(range(0, len(dataset), batch_size)):
            yield [next(iter(dataset))["content"] for _ in range(batch_size)]

    # Base tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    base_vocab = list(bytes_to_unicode().values())

    # Load dataset
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split="train[:10%]",
    )

    # Training and saving
    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(dataset), vocab_size=200000, initial_alphabet=base_vocab
    )
    new_tokenizer.save_pretrained("gpt2/stack-tokenizer")


def get_model():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./gpt2/stack-tokenizer")

    # Configuration
    config_kwargs = {
        "vocab_size": len(tokenizer),
        "scale_attn_by_layer_idx": True,
        "reorder_and_upcast_attn": True,
    }

    # Load model with config and push to hub
    config = AutoConfig.from_pretrained("openai-community/gpt2-large", **config_kwargs)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained("gpt2/stack-model")


def get_datasets(tokenizer):
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split="train[:100]",
    )
    dataset = dataset.train_test_split(test_size=0.3)

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["content"]])

    tokenized = dataset.map(
        preprocess_function, batched=True, remove_columns=dataset["train"].column_names
    )

    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(group_texts, batched=True)

    return grouped


def train_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("./gpt2/stack-tokenizer")
    model = AutoModelForCausalLM.from_pretrained("./gpt2/stack-model")

    # Load train and test sets
    datasets = get_datasets(tokenizer)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="gpt2-stack-finetune",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=True,
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    # train_tokenizer()
    # get_model()
    train_model()
