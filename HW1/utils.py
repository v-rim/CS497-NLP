from collections import defaultdict

class NGramModel:
    def __init__(self, n=1, filename=None, add_1_smoothing=False):
        self.n = n
        self.filename = filename
        self.add_1_smoothing = add_1_smoothing

        self.tokens = None
        self.num_tokens = 0
        self.prefix_count = defaultdict(int)
        self.ngram_count = defaultdict(int)    

        if filename is not None:
            self.set_corpus(filename)  

    def set_corpus(self, filename):
        with open(f"{filename}.tok", "r") as f:
            self.tokens = [line.rstrip() for line in f]

        # Load file text into object
        self.num_tokens = len(self.tokens)
        if self.num_tokens < self.n:
            print("ERROR: Less tokens in file than ngram context size")
            return None
        
        # Count ngrams using a sliding window, converting to tuple for keys
        context = self.tokens[:self.n]
        for token in self.tokens[self.n:]:
            self.prefix_count[tuple(context[:-1])] += 1
            self.ngram_count[tuple(context)] += 1
            context.pop(0)
            context.append(token)
        # Above loop misses token
        self.prefix_count[tuple(context[:-1])] += 1
        self.ngram_count[tuple(context)] += 1

    def get_prob(self, ngram):
        # Convert to tuples for keys
        prefix = tuple(ngram[:-1])
        ngram = tuple(ngram)

        # Make sure that the right size ngram is being passed in
        if len(ngram) != self.n:
            print("ERROR: Length of ngram not equal to model ngram size")
            return None
        
        # Case for unigrams
        if len(ngram) == 1:
            return self.ngram_count[ngram] / self.num_tokens
        
        # Case for non-unigrams
        num_prefix = self.prefix_count[prefix] if prefix in self.prefix_count else 0
        num_ngram = self.ngram_count[ngram] if ngram in self.ngram_count else 0
        return num_ngram / num_prefix


    def get_perplexity(self, string):
        pass