from datasets import load_dataset, IterableDataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
import torch
from torch.utils.data import DataLoader


class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.epoch = 0
        self.infinite = infinite

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)


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
    tokenizer = AutoTokenizer.from_pretrained("gpt2/stack-tokenizer")

    # Configuration
    config_kwargs = {
        "vocab_size": len(tokenizer),
        "scale_attn_by_layer_idx": True,
        "reorder_and_upcast_attn": True,
    }

    # Load model with config and push to hub
    config = AutoConfig.from_pretrained("openai-community/gpt2-large", **config_kwargs)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained("gpt2/stack-model", push_to_hub=True)


def train_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2/stack-tokenizer")
    model = AutoModelForCausalLM.from_pretrained("gpt2/stack-model")
    model.gradient_checkpointing_enable()

    def create_dataloaders():
        train_data = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train[:7%]",
        )
        train_data = train_data.shuffle(buffer_size=8)
        valid_data = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train[7%:10%]",
        )

        train_dataset = ConstantLengthDataset(tokenizer, train_data, infinite=True)
        valid_dataset = ConstantLengthDataset(tokenizer, valid_data, infinite=False)

        train_dataloader = DataLoader(train_dataset, batch_size=8)
        eval_dataloader = DataLoader(valid_dataset, batch_size=8)
        return train_dataloader, eval_dataloader

    train_dataloader, eval_dataloader = create_dataloaders()


if __name__ == "__main__":
    # train_tokenizer()
    get_model()
