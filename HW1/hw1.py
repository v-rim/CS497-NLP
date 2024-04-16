from nltk.tokenize import TreebankWordTokenizer
from transformers import GPT2TokenizerFast

from utils import NGramModel


def part_1():
    print("Starting Part 1")
    # Creates GPT2 Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

    filenames = ["wiki2.train", "wiki2.valid", "wiki2.test"]
    for name in filenames:

        # Open file
        file_text = None
        with open(f"texts/{name}.txt", "r", encoding="utf-8") as f:
            file_text = f.read()

        # Tokenize using NLTK
        file_text_tokenized = TreebankWordTokenizer().tokenize(file_text)
        with open(f"tokens/{name}.tok", "w", encoding="utf-8") as f:
            for token in file_text_tokenized:
                f.write(f"{token}\n")

        # Tokenize using GPT2
        words = tokenizer.tokenize(file_text, add_special_tokens=False)
        with open(f"tokens/{name}_gpt.tok", "w", encoding="utf-8") as f:
            for word in words:
                f.write(f"{word}\n")


def part_2():
    print("\nStarting Part 2")
    for n in [1, 2, 3, 7]:
        print(f"{n}-gram model")
        ngram = NGramModel(n, "tokens/wiki2.train")
        ngram_gpt = NGramModel(n, "tokens/wiki2.train_gpt")

        ppl = ngram.get_perplexity("tokens/wiki2.test")
        ppl_gpt = ngram_gpt.get_perplexity("tokens/wiki2.test_gpt")
        print(f"  PPL for NLTK is {ppl}")
        print(f"  PPL for GPT is {ppl_gpt}")


def part_3():
    print("\nStarting Part 3")
    for n in [1, 2, 3, 7]:
        print(f"{n}-gram model")
        ngram = NGramModel(n, "tokens/wiki2.train", True)
        ngram_gpt = NGramModel(n, "tokens/wiki2.train_gpt", True)

        ppl = ngram.get_perplexity("tokens/wiki2.test", False)
        ppl_gpt = ngram_gpt.get_perplexity("tokens/wiki2.test_gpt", False)
        print(f"  PPL for NLTK is {ppl}")
        print(f"  PPL for GPT is {ppl_gpt}")


def part_4():
    pass


def part_5a():
    print("\nStarting Part 5")

    # Open file
    file_text = None
    f = open(f"examples.txt", "r", encoding="utf-8")
    lines = f.readlines()

    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

    for i, l in enumerate(lines):
        file_text_tokenized = TreebankWordTokenizer().tokenize(l)

        # Write tokenized text
        with open(f"examples/{i}.tok", "w", encoding="utf-8") as f:
            for token in file_text_tokenized:
                f.write(f"{token}\n")

def part_5b():
    # for i in range(11):
    #     print(f"Example {i}")
    #     for n in [1, 2, 3, 7]:
    #         ngram = NGramModel(n, "tokens/wiki2.train", True)
    #         ppl = ngram.get_perplexity(f"examples/{i}")
    #         print(f"  {n}-gram model PPL for NLTK is {ppl}")
            
    for n in [1, 2, 3, 7]:
        ngram = NGramModel(n, "ex-0", True)
        ppl = ngram.get_perplexity(f"ex-0")
        print(f"  {n}-gram model PPL for NLTK is {ppl}")

def main():
    # part_1()
    # part_2()
    # part_3()
    # part_4()
    part_5a()
    part_5b()


if __name__ == "__main__":
    main()
