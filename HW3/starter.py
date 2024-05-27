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

def array_to_csv(data, header, name):
    f = open(f"{name}.csv", "w")
    f.write(f"{header}\n")
    for row in data:
        for i in range(len(row)):
            v = row[i]
            f.write(f"{v}")

            if i != len(row) - 1:
                f.write(", ")
            else:
                f.write("\n")
    f.close()

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
        return dataset

    train = generate_dataset("train_complete.jsonl")
    valid = generate_dataset("dev_complete.jsonl")
    test = generate_dataset("test_complete.jsonl")

    train_dataloader = DataLoader(train, shuffle=False, batch_size=16)
    valid_dataloader = DataLoader(valid, shuffle=False, batch_size=16)
    test_dataloader = DataLoader(test, shuffle=False, batch_size=16)

    num_epochs = 20
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    criterion = nn.CrossEntropyLoss()
    
    train_loss = []
    valid_loss = []
    test_loss = []
    valid_accuracy = []
    test_accuracy = []
    batch_index = 0
    
    print()
    print("Zero-shot performance")
    
    model.eval()
    metric = evaluate.load("accuracy")
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
    
    model.eval()
    metric = evaluate.load("accuracy")
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
            train_loss.append([batch_index, loss.item()])
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            batch_index += 1

        model.eval()
        metric = evaluate.load("accuracy")
        losses = []
        for batch in valid_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            data = batch["input_ids"]
            targets = batch["label"]

            with torch.no_grad():
                outputs = model(data)
                grouped_probs = outputs.view(-1, 4)
                target_index = torch.argmax(targets.view(-1, 4), dim=-1)
                
            loss = criterion(grouped_probs, target_index)
            losses.append(loss.item())

            grouped_probs = outputs.view(-1, 4)
            references = torch.argmax(targets.view(-1, 4), dim=1)
            predictions = torch.argmax(grouped_probs, dim=-1)

            metric.add_batch(predictions=predictions, references=references)
        valid_loss.append([batch_index, np.mean(losses)])
        acc = metric.compute()["accuracy"]
        print(f"Validation Accuracy: {acc:.4f}")
        valid_accuracy.append([batch_index, acc])
        
        model.eval()
        metric = evaluate.load("accuracy")
        losses = []
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            data = batch["input_ids"]
            targets = batch["label"]

            with torch.no_grad():
                outputs = model(data)
                grouped_probs = outputs.view(-1, 4)
                target_index = torch.argmax(targets.view(-1, 4), dim=-1)
                
            loss = criterion(grouped_probs, target_index)
            losses.append(loss.item())

            grouped_probs = outputs.view(-1, 4)
            references = torch.argmax(targets.view(-1, 4), dim=1)
            predictions = torch.argmax(grouped_probs, dim=-1)

            metric.add_batch(predictions=predictions, references=references)
        test_loss.append([batch_index, np.mean(losses)])
        acc = metric.compute()["accuracy"]
        print(f"Test Accuracy: {acc:.4f}")
        test_accuracy.append([batch_index, acc])
        
    array_to_csv(train_loss, "Batch Index, Loss", "q1_results/train_loss")
    array_to_csv(valid_loss, "Batch Index, Loss", "q1_results/valid_loss")
    array_to_csv(test_loss, "Batch Index, Loss", "q1_results/test_loss")
    array_to_csv(valid_accuracy, "Batch Index, Accuracy", "q1_results/valid_accuracy")
    array_to_csv(test_accuracy, "Batch Index, Accuracy", "q1_results/test_accuracy")


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

    # Zero-shot Accuracy Validation
    correct = 0
    total = 0
    for i in range(len(dict_valid)):
        inputs = tokenizer(dict_valid[i][:-1], return_tensors="pt")
        generation_output = model.generate(
            **inputs,
            return_dict_in_generate=True,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id)
        if dict_valid[i][-1] in tokenizer.decode(generation_output['sequences'][0][-1], ignore_special_tokens=True):
            correct += 1
        total += 1
    print(f"Zero-shot Accuracy Validation: {correct/total}")

    # Zero-shot Accuracy Test
    correct = 0
    total = 0
    for i in range(len(dict_test)):
        inputs = tokenizer(dict_test[i][:-1], return_tensors="pt")
        generation_output = model.generate(
            **inputs,
            return_dict_in_generate=True,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id)
        if dict_test[i][-1] in tokenizer.decode(generation_output['sequences'][0][-1], ignore_special_tokens=True):
            correct += 1
        total += 1
    print(f"Zero-shot Accuracy Test: {correct/total}")

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=20.0)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset["train"],
    )
    trainer.train()
    trainer.save_model()

    # Fine-tuned Accuracy Validation
    correct = 0
    total = 0
    for i in range(len(dict_valid)):
        inputs = tokenizer(dict_valid[i][:-1], return_tensors="pt").to("cuda")
        generation_output = model.generate(
            **inputs,
            return_dict_in_generate=True,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id)
        if dict_valid[i][-1] in tokenizer.decode(generation_output['sequences'][0][-1], ignore_special_tokens=True):
            correct += 1
        total += 1
    print(f"Fine-tuned Accuracy Validation: {correct/total}")
    
    # Fine-tuned Accuracy Test
    correct = 0
    total = 0
    for i in range(len(dict_test)):
        inputs = tokenizer(dict_test[i][:-1], return_tensors="pt").to("cuda")
        generation_output = model.generate(
            **inputs,
            return_dict_in_generate=True,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id)
        if dict_test[i][-1] in tokenizer.decode(generation_output['sequences'][0][-1], ignore_special_tokens=True):
            correct += 1
        total += 1
    print(f"Fine-tuned Accuracy Test: {correct/total}")


if __name__ == "__main__":
    Q1()
    Q2()
