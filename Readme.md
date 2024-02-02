# Concepts

## Embedding
``` nn.Embedding(num_embeddings: int, embedding_dim : int) ```  
* num_embeddings : size of the dictionary of embeddings  
* embedding_dim : the size of each embedding vector  

## Lookup table  
Like a hashmap, but instead of using a hash function to hash the key and then access the value, we use directly the key to access the value.  

## Embedding lookup table in machine learning
Transform one integer token into a vector that has a defined size and different weights. This weights will be trained using the loss gradient.

* integer token : Take a letter and use a tokenizer to transform this token into a number. This number becomes then the signature of this token.
``` encode (input : str) ```  

## LOSS
* Cross Entropy : 

## LOGITS
In the context of machine learning it is the raw output of the model. It represents the predictions of it.

## Batch, Temporal, Channels

# References :  
Andrej KARPATHY, Let's build GPT: from scratch, in code, spelled out. https://www.youtube.com/watch?v=kCc8FmEb1nY  

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. Attention Is All You Need. https://arxiv.org/abs/1706.03762  