from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification
import torch.optim as optim
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import evaluate
from collections import defaultdict
from torch.utils.data import DataLoader

import torch
import math
import time
import sys
import json
import numpy as np

def Q1():  
    torch.manual_seed(0)
    answers = ['A','B','C','D']

    train = []
    valid = []
    test = []
    
    file_name = 'train_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem'] + " "
        ans = answers.index(result['answerKey'])
        
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            train.append({"label": label, "text": text})
                
    file_name = 'dev_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem'] + " "
        ans = answers.index(result['answerKey'])
        
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            valid.append({"label": label, "text": text})
        
    file_name = 'test_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem'] + " "
        ans = answers.index(result['answerKey'])
        
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            test.append({"label": label, "text": text})

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    linear = torch.rand(768,1)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    train = Dataset.from_list(train).map(tokenize_function, batched=True)
    valid = Dataset.from_list(train).map(tokenize_function, batched=True)
    test = Dataset.from_list(train).map(tokenize_function, batched=True)
    
    train = train.remove_columns(["text"])
    valid = valid.remove_columns(["text"])
    test = test.remove_columns(["text"])
    
    train = train.rename_column("label", "labels")
    valid = valid.rename_column("label", "labels")
    test = test.rename_column("label", "labels")
    
    train.set_format(type="torch")
    valid.set_format(type="torch")
    test.set_format(type="torch")
    
    train = train.select(range(100))
    valid = train.select(range(100))
    test = train.select(range(100))

    train_dataloader = DataLoader(train, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(valid, batch_size=8)
    
    from transformers import get_scheduler

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    from tqdm.auto import tqdm

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        

    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(metric.compute())

def Q2():  
    torch.manual_seed(0)
    answers = ['A','B','C','D']

    train = []
    test = []
    valid = []

    dict_train = []
    dict_valid = []
    dict_test = []

    # Function to add data to a dictionary
    def add_to_list(result, l):
        inputs = "[START] " +  result['fact1'] + " [SEP] "+ \
        result['question']['stem'] + " [SEP] " + \
        "[A] " + result['question']['choices'][0]['text'] + " " + \
        "[B] " + result['question']['choices'][1]['text'] + " " + \
        "[C] " + result['question']['choices'][2]['text'] + " " + \
        "[D] " + result['question']['choices'][3]['text'] + \
        " [ANSWER] " + result['answerKey']
        
        l.append(inputs)

    def write_to_file(file_name, data):
        text_data = open(file_name, "w")
        for d in data:
            text_data.write(d + "\n")
        text_data.close()
    
    # Train dataset
    file_name = 'train_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        train.append(obs)
        add_to_list(result, dict_train)
    write_to_file("train.txt", dict_train)
    
                
    # Validation dataset
    file_name = 'dev_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        valid.append(obs)
        add_to_list(result, dict_valid)
    write_to_file("valid.txt", dict_valid)
    
    # Test dataset
    file_name = 'test_complete.jsonl'
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        test.append(obs)
        add_to_list(result, dict_test)
    write_to_file("test.txt", dict_test)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_dataset('text', data_files='train.txt')
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=1.0)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset["train"],
    )
    trainer.train()
    trainer.save_model()

    ids = tokenizer.encode(dict_test[0][:-1], return_tensors='pt').to('cuda')
    print(f"Input: {dict_test[0][: -1]}\n")
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_new_tokens=1,
    )
    print(ids)
    print(type(final_outputs))
    print(final_outputs)
    print(f"Output -2: {tokenizer.decode(final_outputs[0][-2], skip_special_tokens=True)}")
    print(f"Output -1: {tokenizer.decode(final_outputs[0][-1], skip_special_tokens=True)}")



                 
if __name__ == "__main__":
    Q1()
    # Q2()
