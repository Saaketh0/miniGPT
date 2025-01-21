# PyTorch Imports
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
# Hugging Face Imports
import transformers
from datasets import load_dataset
# Misc Imports
import time

print("Program Start: " + time.strftime("%H:%M:%S"), end="\n")
print(torch.xpu.is_available())

# ----------- This is the main training file of my miniGPT model ------------
# ---- All torch terms used in this file are explained either in the READ.ME or in an excel sheet, need to finish this first ----

torch.manual_seed(11)

# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
# Originally 64, but reducing down for testing purposes
block_size = 256 # what is the maximum context length for predictions?
max_iters = 80 # Training iterations
eval_interval = 500
learning_rate = 3e-4 # Controls step size during optimization, lower is more accurate but slower
device = 'xpu' if torch.xpu.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2 # Dropout used was 0.1 in the paper, could be changed later to 0.2


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

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = training_data if (split == 'train') else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module): # Basically 3.2.1 from the paper "Attention is All You Need"
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd,head_size, bias=False) # Bias is set to false to prevent overfitting
        self.value = nn.Linear(n_embd,head_size, bias=False)
        self.query = nn.Linear(n_embd,head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout) # Helps accuracy by preventing overfitting on smaller datasets

    def forward(self,x):
        # Defines a matrix size 512x512 with all values below the diagonal set to 1
        B,T,C = x.shape
        rando_init = torch.randn(C,C)
        tril = torch.tril(torch.ones(T,T)).to(device)
        key,query,value = self.key(x),self.query(x),self.value(x) # Populates the linear with random values

        wei = torch.zeros(C,C)
        wei = key @ query.transpose(-1,-2) * (C**-0.5) # Dot Product of Key and Query Here, divided by the square root of the length of the key to reduce differences between points
        wei = wei.masked_fill(tril==0, float('-inf')).to(device) # Masking upper hald of matrix, for decoder architecture
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        result = wei @ value

        return result

class MultiHeadAttention(nn.Module): # Basically 3.2.2 from the paper "Attention is All You Need"
    # When calling this function, call the function with n_embd // num_heads instead of original with a single head
    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        # Concatenates the heads and linear layer
        # Multiplies by W0, which is x.shape
        result = torch.cat([h(x) for h in self.heads], dim=-1) # Tensor size B,T,head_size * num_heads
        result = self.dropout(self.proj(result)) # Resizing it while also applying dropout
        return result
    
class FeedForward(nn.Module): # Basically 3.3 from the paper "Attention is All You Need"
    def __init__(self,n_embd):
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
    

# -------------- This is the transformer block, which runs the classes called above. ------------
class DecoderBlock(nn.Module): # The 3 classes above ran the whole thing, while this basically just puts it all together--
    def __init__(self,n_embd,n_head):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(n_head,n_embd//n_head)
        self.FeedForward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self,x):
        # Adding them add up the residual connections
        x = x + self.MultiHeadAttention(self.ln1(x)) # This is the main function of the class, it is called when you call the class
        x = x + self.FeedForward(self.ln2(x)) # This is the Feed Forward Layer, making it more informational
        return x
    
class miniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(vocab_size, n_embd)
        self.position = nn.Embedding(block_size, n_embd)
        self.DecoderBlock = nn.Sequential(*[DecoderBlock(n_embd,n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token(idx) # (B,T,C)
        pos_emb = self.position(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.DecoderBlock(x) # (B,T,C)
        x = self.ln(x) # (B,T,C)
        logits = self.lm(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = miniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters): # Training loop, iterates through the data

    #if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    print("Iteration " + str(iter) + ": " + time.strftime("%H:%M:%S"), end="\n")

    optimizer.zero_grad() # Clearing gradients
    xb,yb = get_batch("train") # Tokenized text, and the next token after that
    logits, loss = model(xb,yb) # Performs the forward pass on the data
    if torch.isnan(loss) or torch.isinf(loss):
        print("Loss is bugging, skipping this iteration")
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevents exploding gradients
        optimizer.step() #  Perform a single optimization step    

context = torch.zeros((1,1), dtype=torch.long).to(device)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist())) # Generates text based on the model

