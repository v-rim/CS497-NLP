from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import evaluate

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
    test = []
    valid = []
    
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
        
        print(obs)
        print(' ')
        
        print(result['question']['stem'])
        print(' ',result['question']['choices'][0]['label'],result['question']['choices'][0]['text'])
        print(' ',result['question']['choices'][1]['label'],result['question']['choices'][1]['text'])
        print(' ',result['question']['choices'][2]['label'],result['question']['choices'][2]['text'])
        print(' ',result['question']['choices'][3]['label'],result['question']['choices'][3]['text'])
        print('  Fact: ',result['fact1'])
        print('  Answer: ',result['answerKey'])
        print('  ')
                
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

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    linear = torch.rand(768,2)
    
#    Add code to fine-tune and test your MCQA classifier.

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
    # Q1()
    Q2()
