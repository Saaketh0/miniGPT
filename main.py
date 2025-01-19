# PyTorch Imports
import torch
from torch import nn
from torch.nn import functional as F
# Hugging Face Imports
import transformers
from datasets import load_dataset

# ----------- This is the main training file of my miniGPT model ------------
# ---- All torch terms used in this file are explained either in the READ.ME or in an excel sheet, need to finish this first ----

torch.manual_seed(11)

# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 # Training iterations
eval_interval = 500
learning_rate = 3e-4 # Controls step size during optimization, lower is more accurate but slower
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

#print(torch.__version__)
#print(transformers.__version__)
with open('input.txt','r',encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)


# This is the tokenizer, but it is extremely simple, change it to a more complex one later
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encodes the text in UTF-8 format
decode = lambda l: ''.join([itos[i] for i in l]) # decodes text in same format


data = torch.tensor(encode(text), dtype=torch.long)

# Splitting up training data and validation data, validation is last 10%
n = int(len(data) * 0.9)
training_data = data[:n]
validation_data = data[n:]




class Head(nn.module): # Basically 3.2.1 from the paper "Attention is All You Need"
    def __init__(self, head_size):
        super()._init__()

        self.key = nn.Linear(n_embd,head_size)
        self.value = nn.Linear(n_embd,head_size)
        self.query = nn.Linear(n_embd,head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout) # Helps accuracy by preventing overfitting on smaller datasets

    def forward(self,x):
        # Defines a matrix size 512x512 with all values below the diagonal set to 1
        B,T,C = x.shape
        rando_init = torch.randn(C,C)
        tril = torch.tril(torch.ones(T,T))
        key,query,value = self.key(x),self.query(x),self.value(x) # Populates the linear with random values

        wei = torch.zeros(C,C)
        wei = key @ query.transpose(-1,-2) * (C**-0.5) # Dot Product of Key and Query Here, divided by the square root of the length of the key to reduce differences between points
        wei = wei.masked_fill(tril==0, float('-inf')) # Masking upper hald of matrix, for decoder architecture
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        result = wei @ value

        return result

class MultiHeadAttention(nn.Module): # Basically 3.2.2 from the paper "Attention is All You Need"
    # When calling this function, call the function with n_embd // num_heads instead of original with a single head
    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(num_heads) for _ in range(num_heads)])
        self.linear = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        # Concatenates the heads and linear layer
        # Multiplies by W0, which is x.shape
        result = torch.cat([h(x) for h in self.heads], dim=-1) # Tensor size B,T,head_size * head_num
        result = self.dropout(self.proj(result)) # Resizeing it while also applying dropout
        return result
    
class FeedForward(nn.Module): # Basically 3.3 from the paper "Attention is All You Need"
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( # When called, goes through all of this in order (in sequence haha)
            nn.Linear(n_embd, 4*n_embd), # Paper had an outer layer of 512 and inner layer of 2048, so 4 times
            nn.ReLU(), # Makes any negative values 0 (Technical term is that it adds non-linearity)
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
        #self.dropout = nn.Dropout(dropout) Moved it inside
    
    def forward(self,x):
        return self.net(x)

class miniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = nn.Embedding(vocab_size, n_embd)
        self.MultiHeadAttention = MultiHeadAttention(n_head,n_embd//n_head) # 64 embeds times 6 equals the original 384
        self.FeedForward = FeedForward()
    
    def forward(self, x):
        x = self.Embedding(x)
        return x
    
    def parameters(self, recurse = True):
        return super().parameters(recurse)+ self.Embedding.parameters()

nn = miniGPT()
optimizer = torch.optim.AdamW(nn.parameters(), lr=learning_rate)