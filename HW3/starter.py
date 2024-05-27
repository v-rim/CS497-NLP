import json
import math
import sys
import time

import evaluate
import numpy as np
import torch
import torch.optim as optim
from datasets import Dataset, load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    get_scheduler,
)


class BERTWithClassificationHead(nn.Module):
    def __init__(self):
        super(BERTWithClassificationHead, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, 1)

    def forward(self, ids):
        sequence_output = self.bert(ids).last_hidden_state
        return self.linear(sequence_output[:, 0, :].view(-1, 768))


def Q1():
    torch.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BERTWithClassificationHead()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    def generate_dataset(file_name):
        answers = ["A", "B", "C", "D"]
        data = []

        with open(file_name) as json_file:
            json_list = list(json_file)
        for i in range(len(json_list)):
            json_str = json_list[i]
            result = json.loads(json_str)

            base = (
                "[CLS] "
                + result["fact1"]
                + " [SEP] "
                + result["question"]["stem"]
                + " "
            )
            ans = answers.index(result["answerKey"])

            for j in range(4):
                text = base + result["question"]["choices"][j]["text"] + " [SEP]"
                if j == ans:
                    label = 1
                else:
                    label = 0
                data.append({"text": text, "label": label})

        dataset = Dataset.from_list(data).map(tokenize_function, batched=True)
        dataset = dataset.remove_columns(["text"])
        dataset.set_format(type="torch")
        dataset = dataset.select(range(8 * 8))
        return dataset

    train = generate_dataset("train_complete.jsonl")
    valid = generate_dataset("dev_complete.jsonl")
    test = generate_dataset("test_complete.jsonl")

    train_dataloader = DataLoader(train, shuffle=False, batch_size=8)
    valid_dataloader = DataLoader(valid, shuffle=False, batch_size=8)
    test_dataloader = DataLoader(test, shuffle=False, batch_size=8)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        print()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in (pbar := tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            data = batch["input_ids"]
            targets = batch["label"]

            optimizer.zero_grad()
            outputs = model(data)

            grouped_probs = outputs.view(-1, 4)
            target_index = torch.argmax(targets.view(-1, 4), dim=-1)

            loss = criterion(grouped_probs, target_index)
            pbar.set_description(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        metric = evaluate.load("accuracy")
        model.eval()
        for batch in valid_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            data = batch["input_ids"]
            targets = batch["label"]

            with torch.no_grad():
                outputs = model(data)

            grouped_probs = outputs.view(-1, 4)
            references = torch.argmax(targets.view(-1, 4), dim=1)
            predictions = torch.argmax(grouped_probs, dim=-1)

            metric.add_batch(predictions=predictions, references=references)
        print(f"Validation Accuracy: {metric.compute()["accuracy"]:.4f}")
        
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            data = batch["input_ids"]
            targets = batch["label"]

            with torch.no_grad():
                outputs = model(data)

            grouped_probs = outputs.view(-1, 4)
            references = torch.argmax(targets.view(-1, 4), dim=1)
            predictions = torch.argmax(grouped_probs, dim=-1)

            metric.add_batch(predictions=predictions, references=references)
        print(f"Test Accuracy: {metric.compute()['accuracy']:.4f}")


def Q2():
    torch.manual_seed(0)
    answers = ["A", "B", "C", "D"]

    train = []
    test = []
    valid = []

    dict_train = []
    dict_valid = []
    dict_test = []

    # Function to add data to a dictionary
    def add_to_list(result, l):
        inputs = (
            "[START] "
            + result["fact1"]
            + " [SEP] "
            + result["question"]["stem"]
            + " [SEP] "
            + "[A] "
            + result["question"]["choices"][0]["text"]
            + " "
            + "[B] "
            + result["question"]["choices"][1]["text"]
            + " "
            + "[C] "
            + result["question"]["choices"][2]["text"]
            + " "
            + "[D] "
            + result["question"]["choices"][3]["text"]
            + " [ANSWER] "
            + result["answerKey"]
        )

        l.append(inputs)

    def write_to_file(file_name, data):
        text_data = open(file_name, "w")
        for d in data:
            text_data.write(d + "\n")
        text_data.close()

    # Train dataset
    file_name = "train_complete.jsonl"
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)

        base = result["fact1"] + " [SEP] " + result["question"]["stem"]
        ans = answers.index(result["answerKey"])

        obs = []
        for j in range(4):
            text = base + result["question"]["choices"][j]["text"] + " [SEP]"
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text, label])
        train.append(obs)
        add_to_list(result, dict_train)
    write_to_file("train.txt", dict_train)

    # Validation dataset
    file_name = "dev_complete.jsonl"
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)

        base = result["fact1"] + " [SEP] " + result["question"]["stem"]
        ans = answers.index(result["answerKey"])

        obs = []
        for j in range(4):
            text = base + result["question"]["choices"][j]["text"] + " [SEP]"
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text, label])
        valid.append(obs)
        add_to_list(result, dict_valid)
    write_to_file("valid.txt", dict_valid)

    # Test dataset
    file_name = "test_complete.jsonl"
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)

        base = result["fact1"] + " [SEP] " + result["question"]["stem"]
        ans = answers.index(result["answerKey"])

        obs = []
        for j in range(4):
            text = base + result["question"]["choices"][j]["text"] + " [SEP]"
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text, label])
        test.append(obs)
        add_to_list(result, dict_test)
    write_to_file("test.txt", dict_test)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_dataset("text", data_files="train.txt")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    train_dataset = train_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    training_args = TrainingArguments(output_dir="results", num_train_epochs=1.0)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset["train"],
    )
    trainer.train()
    trainer.save_model()

    ids = tokenizer.encode(dict_test[0][:-1], return_tensors="pt").to("cuda")
    print(f"Input: {dict_test[0][: -1]}\n")
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_new_tokens=1,
    )
    print(ids)
    print(type(final_outputs))
    print(final_outputs)
    print(
        f"Output -2: {tokenizer.decode(final_outputs[0][-2], skip_special_tokens=True)}"
    )
    print(
        f"Output -1: {tokenizer.decode(final_outputs[0][-1], skip_special_tokens=True)}"
    )


if __name__ == "__main__":
    Q1()
    # Q2()
