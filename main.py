import torch
from BigramLanguageModel import BigramLanguageModel 

with open('Resources/shakespeare_data.txt', 'r', encoding='utf-8') as file:
    text = file.read()


## Create a unique set of characters in the text
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
# print(''.join(vocab))
# print(f'Vocabulary size: {vocab_size}')
    
## Create a mapping from character to index and vice versa
## Tokenizer
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

# print(encode('hello there'))
# print(decode(encode('hello there')))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

# Split data into train and validation set
train_size = int(0.9 * len(data))
train_data, val_data = data[:train_size], data[train_size:]

## Chunking the data
# block_size = 8
# train_data[:block_size+1]

# x = train_data[:block_size]
# y = train_data[1:block_size+1]

# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f'Context: {(context)} -> Target: {(target)}')

torch.manual_seed(1337)

batch_size = 4 # How many idependent sequences to train on in parallel
block_size = 8 # What is the maximum context length for predictions

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, data.size(0) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
# print('inputs: ')
# print(xb)
# print('targets: ')
# print(yb)

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f'Context: {context} -> Target: {target}')

## Bigram language model

model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)

# print(logits.shape)
# print(loss)
# idx = model.generate(idx=torch.zeros(1,1, dtype=torch.long),max_new_tokens=100)
# print(decode(idx[0].tolist()))

## create optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 32
for step in range(10000):
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(f'Step: {step}, Loss: {loss.item()}')

idx = model.generate(idx=torch.zeros(1,1, dtype=torch.long),max_new_tokens=100)
print(decode(idx[0].tolist()))