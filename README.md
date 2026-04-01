# Character-Level Language Model: Sherlock Holmes

A PyTorch implementation of character-level LSTM text generation, trained on the complete Sherlock Holmes corpus by Arthur Conan Doyle. This project compares optimizer performance and learning rate schedules for character-level recurrent language models.

The architecture and training conventions are based on [torch-rnn](https://github.com/jcjohnson/torch-rnn) by Justin Johnson, which itself is a cleaner re-implementation of Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn). Our implementation ports the core ideas to PyTorch, retaining torch-rnn's learned character embeddings, contiguous batch streaming, and checkpoint-based model selection.

## Overview

The project consists of two experiments:

1. **Optimizer comparison** — RMSprop vs. SGD with momentum vs. Adam, trained for 30 epochs under identical hyperparameters. Adam and RMSprop achieve comparable validation losses (~1.36), while SGD with momentum fails to converge meaningfully (~2.98).

2. **Hyperparameter and LR schedule search** — Six configurations varying batch size, sequence length, dropout, and learning rate schedule (step decay, aggressive decay, cosine annealing). The best configuration combines a higher learning rate, larger batch size, longer sequence length, reduced dropout, and cosine annealing, achieving a validation loss of **1.2368** at 30 epochs.

## Model Architecture

| Parameter | Value |
|-----------|-------|
| RNN type | LSTM |
| Hidden size | 256 |
| Layers | 2 |
| Embedding dim | 64 (learned, following [torch-rnn](https://github.com/jcjohnson/torch-rnn)) |
| Dropout | 0.5 (baseline) / 0.3 (best config) |
| Gradient clipping | 5.0 |

The model uses learned character embeddings rather than one-hot encoding, following the `nn.LookupTable` approach from [torch-rnn](https://github.com/jcjohnson/torch-rnn). Hidden states are carried across batches within an epoch (equivalent to torch-rnn's `remember_states` mode) and detached to truncate backpropagation through time.

## Repository Structure

```
.
├── data/
│   └── cnus.txt                        # Sherlock Holmes corpus
├── cv/                                 # Saved checkpoints
│   └── config_D_30ep.pt
├── charrnn.ipynb                       # Experiment 1: optimizer comparison
├── hyperparameter_experiment.ipynb     # Experiment 2: LR schedule search
├── hyperparameter_comparison.png       # Learning curves (6 configs)
├── optimizer_comparison.png            # Optimizer comparison plots
└── README.md
```

## Quick Start

### Requirements

```
Python 3.8+
PyTorch >= 1.12
NumPy
Matplotlib
```

### Training

The experiments are self-contained in the Jupyter notebooks. To reproduce from scratch:

```bash
# Run optimizer comparison (RMSprop, SGD, Adam — 30 epochs each)
jupyter notebook charrnn.ipynb

# Run hyperparameter/LR schedule search (6 configs — 20 epochs each)
jupyter notebook hyperparameter_experiment.ipynb
```

### Text Generation

After training, generate text from a checkpoint:

```python
# Load best model (Config D, 30 epochs)
checkpoint = torch.load('cv/config_D_30ep.pt')
model = CharRNN(vocab_size, embedding_size=64, rnn_size=256, num_layers=2, dropout=0.3)
model.load_state_dict(checkpoint['model_state'])

# Generate
text = generate(model, prime_text="Sherlock Holmes", length=1000, temperature=0.5)
print(text)
```

## Experiments

### Experiment 1: Optimizer Comparison

Three optimizers trained for 30 epochs with char-rnn default settings (lr=0.002, step decay 0.97× after epoch 10):

| Optimizer | Final Train Loss | Best Val Loss |
|-----------|-----------------|---------------|
| RMSprop | ~1.46 | ~1.36 |
| SGD+Momentum | ~2.98 | ~2.98 |
| Adam | ~1.46 | ~1.36 |

Adam was selected for further experiments.

### Experiment 2: Hyperparameter and LR Schedule Search

Six configurations tested with Adam for 20 epochs:

| Config | LR | Batch | Seq | Dropout | Schedule | Best Val |
|--------|-----|-------|-----|---------|----------|----------|
| A – Baseline | 2e-3 | 50 | 50 | 0.5 | Step decay | 1.3930 |
| B – Higher LR+Batch | 3e-3 | 100 | 50 | 0.5 | Step decay | 1.3856 |
| C – Long Seq+Cosine | 2e-3 | 50 | 100 | 0.5 | Cosine | 1.4049 |
| **D – Combined** | **3e-3** | **100** | **100** | **0.3** | **Cosine** | **1.2568** |
| E – torch-rnn decay | 2e-3 | 50 | 50 | 0.5 | 0.5× every 5 ep | 1.3655 |
| F – Faster decay | 2e-3 | 50 | 50 | 0.5 | 0.5× every 3 ep | 1.3809 |

Configuration D extended to 30 epochs achieves **best validation loss = 1.2368**.

![Hyperparameter Comparison](hyperparameter_comparison.png)

### Sample Generated Text

Config D, 30 epochs, temperature 0.5, primed with "Sherlock Holmes walked into the room":

> Sherlock Holmes walked into the room. "It is a too as the door of the man when you will be an examination of the house," said he. "We was a senses which he is not to lose the constant of the house of the danger. I have to the train and a scleating of the moment of the good brow man. We can see that we have been done."

The model captures dialogue formatting, character references, and the narrative rhythm of the Sherlock Holmes stories, though sentence-level coherence is limited.

## Acknowledgments

This project builds on the following work:

- **[torch-rnn](https://github.com/jcjohnson/torch-rnn)** by Justin Johnson — our architecture follows torch-rnn's design choices: learned character embeddings (replacing char-rnn's one-hot encoding), the Adam optimizer, and the contiguous batch streaming strategy. The aggressive step decay schedule (halving every N epochs) tested in our Experiment 2 is the default LR schedule in torch-rnn.

- **[char-rnn](https://github.com/karpathy/char-rnn)** by Andrej Karpathy — the original Torch/Lua character-level RNN that established the training conventions (step decay, gradient clipping, checkpoint naming) used throughout this project.

- **Alex Graves**, "Generating Sequences With Recurrent Neural Networks" (arXiv:1308.0850, 2013) — foundational work on LSTM sequence generation that motivated the character-level approach.

## License

MIT
