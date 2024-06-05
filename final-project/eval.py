import torch
from datasets import VerificationMode, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

path_to_model = "./gpt2-stack-finetune/checkpoint-610"
path_to_tokenizer = "./gpt2/stack-tokenizer"
device = "cuda"
# num_examples = 27902
# target_tokens = int(10000000 * (.3 / .7)) # 4285714
target_tokens = 5000000
humaneval_tries = 10


def get_encodings(tokenizer):
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split=f"train[:{2500}]",  # Usually enought for 5M tokens
        num_proc=64,
        verification_mode=VerificationMode.NO_CHECKS,
        cache_dir="./cache",
    )
    dataset = dataset.shuffle()
    encodings = tokenizer("\n\n".join(dataset["content"]), return_tensors="pt")
    return encodings


def calculate_perplexity():
    model = AutoModelForCausalLM.from_pretrained(path_to_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer)

    encodings = get_encodings(tokenizer)
    # max_length = model.config.n_positions
    max_length = 256 # Lower because I think it's getting caught up between programs
    stride = 256
    seq_len = encodings.input_ids.size(1)
    print(f"Sequence length: {seq_len}")
    if seq_len < target_tokens:
        print("Dataset too small to calculate perplexity")
        return
    seq_len = target_tokens

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl}")


def calculate_humaneval():
    # model = AutoModelForCausalLM.from_pretrained(path_to_model).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = '# print "Hello, World!"\n'
    prompt_array = [prompt for _ in range(humaneval_tries)]
    inputs = tokenizer(prompt_array, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        inputs,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=10,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    resp = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, r in enumerate(resp):
        print(f"Try {i + 1}:")
        print(r)
        print()


if __name__ == "__main__":
    # calculate_perplexity()
    calculate_humaneval()
