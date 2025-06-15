# GPT From Scratch
*My personal code and annotation of [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) on recreating OpenAI's GPT-3 on a miniature scale.*

*This code is focused on the pretraining aspect of GPT-3 and pays less attention to the more resource-intensive of fine tuning supervised fine tuning (SFT) and reinforcement learning with human feedback (RLHF)*

---
## Example Output
*Achieved on a Kaggle Notebook instance using a P100 GPU with the following hyperparameters and approximately 10 minutes training time.*
### Hyperparameters:
```
batch_size = 256, block_size = 32, max_iters = 10000
eval_interval = 1000, learning_rate = 3e-4, eval_iters = 20,
n_embd = 128, n_head = 8, n_layer = 8, dropout = 0.1
```
### Loss:
```
step 0: train loss 4.6352, val loss 4.6392
step 1000: train loss 1.7505, val loss 1.8864
step 2000: train loss 1.5438, val loss 1.7098
step 3000: train loss 1.4454, val loss 1.6519
step 4000: train loss 1.3937, val loss 1.6068
step 5000: train loss 1.3626, val loss 1.5881
step 6000: train loss 1.3304, val loss 1.5751
step 7000: train loss 1.3013, val loss 1.5716
step 8000: train loss 1.2843, val loss 1.5696
step 9000: train loss 1.2679, val loss 1.5492  
```
### Output:
```
MENENIUS:
Brat God's subject, put now, rate their minds
You beseeming and fouler; but your own men, instruments, well-benefit
To strive ought she is, and vialet, desires to very grass.

ROMEO:
And she that words me arrived and love;
If every incense, nor heights, it fall.

LADY CAMILLO:
No, Jesit she!

ANDIUS:
O respect Now misfortune.

QUEEN ELIZABETH:
O, the sweet Duke of Landalm hold a flesh required minibers of accuse
For brothers; Need sir, for my banishment
As I can my brains in else wide 
```
*This output is shakespeare-like and uses roughly proper english, but does not make for a good story. Extending the context length, MLP layers, attention heads, and training epochs would likely improve the results*

---

## Features
This Large Language Model implementation uses many vital aspects of frontier models or simplified versions to conserve resources.
- Character-level tokenization
- Multi-headed self-attention
- Feed-forward layers
- Layer norm
- Adjustable hyperparameters

---

## Requirements
To balance depth of learning and simplicity, PyTorch was the only library used. As an industry-standard tool, PyTorch allowed for simplified processes like backpropagation, parallel computing, and device optimization.

## Data Set
This implementation was trained on all of Shakespeare's works concatenated in one text file. This file is available at https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## Directory Structure

```
├─ bigram.py   # bigram model without attention mechanism
└─ gpt.py      # full gpt model with multiheaded attention
```
