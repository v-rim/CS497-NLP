from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai-community/gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# python run_clm.py --model_name_or_path openai-community/gpt2 --validation_file wiki2.test.txt --per_device_eval_batch_size 1 --do_eval --output_dir /tmp
# perplexity              =    23.8789

for i in range(11):
    # Load text from a local file
    file_path = f"{i}.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Tokenize the text
    encodings = tokenizer(text, return_tensors="pt").to(device)

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]

        # Prepare target labels for calculating loss
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        target_ids = target_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # Loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity of example {i + 1}: {ppl.item()}")

# Perplexity of example 1: 18.226287841796875
# Perplexity of example 2: 92.43034362792969
# Perplexity of example 3: 12.343201637268066
# Perplexity of example 4: 170.7036895751953
# Perplexity of example 5: 68.51326751708984
# Perplexity of example 6: 39.765769958496094
# Perplexity of example 7: 40.72014617919922
# Perplexity of example 8: 19.503610610961914
# Perplexity of example 9: 127.75165557861328
# Perplexity of example 10: 20.4046630859375
# Perplexity of example 11: 14.58392333984375