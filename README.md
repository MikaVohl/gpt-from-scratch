# GPT From Scratch
*My personal code and annotation of [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) on recreating OpenAI's GPT-3 on a miniature scale.*

*This code is focused on the pretraining aspect of GPT-3 and pays less attention to the more resource-intensive of fine tuning supervised fine tuning (SFT) and reinforcement learning with human feedback (RLHF)*

---
## Example Output
*Achieved on Intel-Based MacBook Pro with the following hyperparameters and approximately 5 minutes training time.*
### Hyperparameters:
```
batch_size = 16, block_size = 16, max_iters = 10000
eval_interval = 500, learning_rate = 7e-4, eval_iters = 200
n_embd = 64, n_head = 4, n_layer = 6, dropout = 0.05
```
### Loss:
```
step 0: train loss 4.5225, val loss 4.5241  
step 500: train loss 2.3356, val loss 2.3550  
step 1000: train loss 2.1512, val loss 2.1794  
step 1500: train loss 2.0501, val loss 2.1052  
...  
step 8000: train loss 1.6969, val loss 1.8560  
step 8500: train loss 1.6901, val loss 1.8485  
step 9000: train loss 1.6760, val loss 1.8468  
step 9500: train loss 1.6764, val loss 1.8324  
```
### Output:
```
Of eyes with will friends fwifever?

PULIXRLAND:
A praint him ready;
Wherefammend instay and there come, Gawdow yet would speak;
Speell, it we mate thy ndwuer,
Theld my dire he gones may bring the see;
And tywith onle.
If I do beseen wred? it live he sI cannot is, and songry
no in this bith
Mhe eybred regove rests respares our sleaving heave and Her discors.

WOXINABEL:
I change must brotther before;
Only gonely! Of marry mant life thy villaniaguinus peals
'Deel to he king gest off them.
```
*This output is shakespeare-like but it's english is incomprehensible due to a lack of compute resources which constrained me to a smaller model architecture*

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
