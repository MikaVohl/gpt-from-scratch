import torch # Pytorch import
import torch.nn as nn # Pytorch's neural network library. Needed for layer, embedding table, and Module objects.
from torch.nn import functional as F # Pytorch's functional library for stateless methods for neural network operations. F.softmax, F.cross_entropy

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions? Will look at only the past {block_size} characters when generating a new one
max_iters = 5000 # Iterations for training
eval_interval = 500 # interval for which the loss will be printed to the console
learning_rate = 1e-3 # gradient descent learning rate (through Adam optimizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu' # allows for using GPU if available
eval_iters = 200 # iterations for evaluation function
n_embd = 32 # dimension of the embedding space. Each character/token will have its own learned position in this embedding space. (a vector in 32 dimensions)
# ------------

# torch.manual_seed(1337) # Manually set seed for the sake of consistent comparison between computers and executions.

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt # Retrieve the dataset (text file)
with open('input.txt', 'r', encoding='utf-8') as f: # Open text file
    text = f.read() # Read text file into the 'text' variable

# here are all the unique characters that occur in this text
chars = sorted(list(set(text))) # vocabulary of known/valid characters to the LLM
vocab_size = len(chars) # size of vocabulary
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) # 1D tensor (array) of characters occuring in the data eg. [0, 23, 21, 5, 1]. These are encoded
n = int(0.9*len(data)) # first 90% will be training data, the rest validation data
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 1D tensor of batch_size randomly picked integers from 0 to len(data) - block_size. Same integer can be picked more than once.
    x = torch.stack([data[i:i+block_size] for i in ix]) # Pytorch Stack object where each element of the stack is a randomly picked block of characters
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Pytorch Stack object that corresponds to the ones in x, but with each element replaced with the next one
    x, y = x.to(device), y.to(device) # Necessary for CUDA compatibility
    return x, y

@torch.no_grad() # disable gradient tracking (saves memory/compute during evaluation). Usually gradient tracker caches operations for backpropagation
def estimate_loss():
    """ Averages the loss over a bunch of different batches for both training and evaluation data """
    out = {} # initialize output dictionary. Will look like: { 'train': ..., 'val': ... }
    model.eval() # Switches certain layers (only those that behave differently during training vs. inference) into “evaluation” behavior.
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # initialize zero array
        for k in range(eval_iters): # iterates from 0 to eval_iters-1
            X, Y = get_batch(split) # gets a random batch of the data
            logits, loss = model(X, Y) # performs a forward pass on the data, and evaluates the loss. Also returns the logits (character predictions, but as probability scores for each character (unnormalized, raw 'prediction scores', not probabilities yet]))
            losses[k] = loss.item() # records the loss
        out[split] = losses.mean() # averages the loss for the type of split (training or evaluation)
    model.train() # returns the model to training mode
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__() # calling constructor of nn.Module
        self.key = nn.Linear(n_embd, head_size, bias=False) # initializes a weight matrix for key
        self.query = nn.Linear(n_embd, head_size, bias=False) # initializes a weight matrix for query
        self.value = nn.Linear(n_embd, head_size, bias=False) # initializes a weight matrix for value
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # stores the lower triangular matrix but doesn't update it with gradient descent (not a parameter, rather a 'buffer')

    def forward(self, x):
        B,T,C = x.shape # shape of input data. Batch, Time, and Channel dimension
        k = self.key(x)   # (B,T,C). Apply the key matrix to the input data
        q = self.query(x) # (B,T,C). Apply the query matrix to the input data
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T). Now wei contains the normalized weights to assign to each token's value vector
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out # the contextualized embeddings for each token (batch B, sequence length T, emb‑dim C)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size) # language-modeling head. Turns hidden embeddings into logit scores

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C). Lookup token embeddings for each index
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C). Lookup learned positional embeddings for each of the T positions
        x = tok_emb + pos_emb # (B,T,C). In practice, learned token and positional embeddings tend to end up somewhat orthogonal to eachother in high dimensional space. This means the sum of the two vectors contains meaningful information and is roughly one-to-one.
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size). Apply the learned linear layer weights to convert embeddings into logit scores for each possible token

        if targets is None: # if in inference, don't compute loss. (No provided correct answers)
            loss = None
        else: # if in training, compute the loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # flattens the logits so it can be passed into cross entropy. Now (B*T,C)
            targets = targets.view(B*T) # flattens the targets so it can be passed into cross entropy. Now (B*T,)
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

model = BigramLanguageModel() # initialize model
m = model.to(device) # device specific model instance

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # 1x1 tensor of the 0 token
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # generate 500 tokens and display them