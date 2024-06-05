from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, VerificationMode
import torch
from tqdm import tqdm

path_to_model = "./gpt2-stack-finetune/checkpoint-6100"
path_to_tokenizer = "./gpt2/stack-tokenizer"
# num_examples = 27902
num_examples = 100
device = "cuda"


def get_encodings(tokenizer):
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split=f"train[:{int(num_examples * 0.3)}]",
        num_proc=64,
        verification_mode=VerificationMode.NO_CHECKS,
        cache_dir="./cache",
    )
    encodings = tokenizer("\n\n".join(dataset["content"]), return_tensors="pt")
    return encodings


def main():
    model = AutoModelForCausalLM.from_pretrained(path_to_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer)

    encodings = get_encodings(tokenizer)
    max_length = model.config.n_positions
    stride = 512
    # seq_len = encodings.input_ids.size(1)
    seq_len = int(10000000 * (.3 / .7))

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

if __name__ == "__main__":
    main()
