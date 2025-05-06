# GPT From Scratch
*My personal code and annotation of [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) on recreating OpenAI's GPT-3 on a miniature scale.*

*This code is focused on the pretraining aspect of GPT-3 and pays less attention to the more resource-intensive of fine tuning supervised fine tuning (SFT) and reinforcement learning with human feedback (RLHF)*

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
