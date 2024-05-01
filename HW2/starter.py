import argparse
import os
import sys
import shutil
import random
import numpy as np
import time
import copy
import math
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from transformers import GPT2TokenizerFast

def read_corpus(filename,tokenizer):
    seq = []
    with open(filename,'rt', encoding="utf-8") as f:
        for line in f:
            line = line.replace('\n','')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return(seq)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x.int())

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 4096, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        # x2 = self.norm_2(x) Get rid of self attention completely
        # x = x + self.dropout_2(self.attn_2(x2, x2x2, trg_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x    

    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout):
        super().__init__()
        self.decoder = Decoder(vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, vocab)

    def forward(self, trg, mask):
        d_output = self.decoder(trg, mask)
        output = self.out(d_output)
        return output

def get_model(opt, vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)
       
    if opt.loadname is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.loadname))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    return model
    
def split_sequences(data, opt):
    seqs = [data[i:i + opt.seqlen] for i in range(0, len(data), opt.seqlen)]
    if len(seqs[-1]) < opt.seqlen:
        pad_num = opt.seqlen - len(seqs[-1])
        seqs[-1].extend([0] * pad_num)

    seqs = [torch.tensor(seq) for seq in seqs]
    return seqs

def batchify(seqs, opt):
    batches = []
    for i in range(0, len(seqs), opt.batchsize):
        batch = nn.utils.rnn.pad_sequence(seqs[i:i+opt.batchsize], batch_first=True)
        batches.append(batch)

    return batches

def join_vocab(vocab1, vocab2, opt):
    seqs1 = split_sequences(vocab1, opt)
    seqs2 = split_sequences(vocab2, opt)
    
    vocab = seqs1 + seqs2
    return vocab

def train_model(model, opt):
    print("training model...")
    model.train()
    model.to(opt.device)

    batches = batchify(opt.vocab, opt)

    for epoch in range(opt.epochs):
        total_loss = 0.0
        total_tokens = 0.0

        for i, batch in enumerate(batches):
            nopeak_mask = torch.stack([torch.tril(torch.ones(opt.seqlen - 1, opt.seqlen - 1)) for b in batch])
            
            # Move tensors to GPU. Really should just initialize them there though
            nopeak_mask = nopeak_mask.to(opt.device)
            batch = batch.to(opt.device)

            opt.optimizer.zero_grad()
            # print(batch.device, nopeak_mask.device)
            output = model(batch[:, :-1], nopeak_mask)
            targets = batch[:, 1:]
            
            predictions = output.view(-1, opt.vocab_size)
            targets = targets.view(-1)
            
            # print(f"{predictions.size() = }")
            # print(f"{targets.size() = }")

            loss = F.cross_entropy(predictions, targets)
            loss.backward()
            opt.optimizer.step()

            total_loss += loss.item()
            total_tokens += targets.size(0)

            ppl = torch.exp(loss)

            #  6. report intermediate trainining perplexity
            # I don't know how often we want to print this
            avg_loss = total_loss / len(batches)
            print(f'Epoch {epoch+1}, Batch: {i}, Loss: {avg_loss:.4f} Perplexity: {ppl:.4f}')
        
        test_model(model, opt)

    torch.save(model.state_dict(), opt.savename)
        
    
def test_model(model, opt):
    print("testing model...")
    model.eval()
    total_loss = 0

    for i, batch in enumerate(opt.test):
        nopeak_mask = torch.stack([torch.tril(torch.ones(opt.seqlen -1, opt.seqlen -1)) for b in range(opt.batchsize)])
        nopeak_mask.to(opt.device)

        output = model(batch[:, :-1], nopeak_mask)

        predictions = output.view(-1, model.vocab_size)
        targets = batch[:, 1:]
        targets = targets.view(-1)
    
        loss = F.cross_entropy(predictions, targets)

        total_loss += loss.item()
        total_tokens += targets.size(0)
        
    avg_loss = total_loss / len(opt.test)
    ppl = torch.exp(avg_loss)
    
    print(f'Perplexity: {ppl:.4f}')
    
    model.train()

def main():
    
    random.seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.00001)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-savename', type=str)    
    parser.add_argument('-loadname', type=str)    
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str,default='model')
    parser.add_argument('-norm', type=float, default=2.0)
                
    opt = parser.parse_args()
    opt.verbose = False

    # opt.device = 0 if opt.no_cuda is False else -1
    # if opt.device == 0:
    #     assert torch.cuda.is_available()
    opt.device = torch.device("cuda:0" if (torch.cuda.is_available() and not opt.no_cuda) else "cpu")
    
    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % (opt.dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    source_name = sys.argv[0]
    dir_name = dir_name + "//"
    opt.dir_name = dir_name
    # shutil.copy(source_name,dir_name + source_name)
    opt.log_file = dir_name + "log_file.txt"
    
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    opt.train = read_corpus('wiki2.train.txt',tokenizer)
    opt.valid = read_corpus('wiki2.valid.txt',tokenizer)
    opt.test = read_corpus('wiki2.test.txt',tokenizer)
    opt.vocab = join_vocab(opt.train, opt.test, opt)
    opt.test = split_sequences(opt.test, opt)
    opt.test = batchify(opt.test, opt)

    
    obs = len(opt.train)
    opt.vocab_size = 50257
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()
    
    model = get_model(opt,opt.vocab_size)
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])        
    text = 'total params: %d' % (params)
    print(text)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.savename is not None:
        try:
            os.mkdir(opt.savename)
        except:
            nothing = 1
    opt.src_pad = 0
    opt.trg_pad = 0
            
    train_model(model,opt)
    test_model(model,opt,-1)
        
if __name__ == "__main__":
    main()        