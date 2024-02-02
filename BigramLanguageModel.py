import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__ (self, vocab_size):
        super().__init__()
        # Token embedding table : numerical vector that represents a token in a n-dimensional space
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx,  targets=None):
        # idx and targets are tensors of size (B, T)
        logits = self.token_embedding_table(idx)
        # logits (B, T, C) : Batch(4) x Time(8) x Channel(vocab_size)
        
        if targets is None:
            loss = None
        else :
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # stretching the tensor to be bidimensional
            targets = targets.view(B*T)
            # Quality of the logits
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # Take only the last time step
            logits = logits[:, -1, :] # (B, T, C) -> (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat([idx, idx_next], dim=-1)
            
        return idx
