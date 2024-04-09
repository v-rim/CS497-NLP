from nltk.tokenize import TreebankWordTokenizer
from utils import NGramModel

def part_1():
    filenames = ["wiki2.train", "wiki2.valid", "wiki2.test"]
    for name in filenames:
        # Open file
        file_text = None
        with open(f"{name}.txt", "r") as f:
            file_text = f.read()

        # Tokenize
        file_text_tokenized = TreebankWordTokenizer().tokenize(file_text)
        
        # Write tokenized text
        with open(f"{name}.tok", "w") as f:
            for token in file_text_tokenized:
                f.write(f"{token}\n")

def part_2():
    pass


def part_3():
    pass


def part_4():
    pass


def part_5():
    pass


def main():
    part_1()
    part_2()
    part_3()
    part_4()
    part_5()


if __name__ == "__main__":
    main()
