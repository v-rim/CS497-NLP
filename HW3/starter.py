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
    Q2()
