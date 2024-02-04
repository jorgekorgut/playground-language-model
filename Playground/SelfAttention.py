import torch

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# print(x.shape)

# x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C)) #bow = bag of words
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] #(t, C)
        xbow[b,t] = torch.mean(xprev, 0)


## matrix version

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B,T, T) @ (B,T,C) -> (B,T,C)

print(xbow)
print(xbow2)

