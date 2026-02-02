import numpy as np

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    d = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d)   # (n,n)
    weights = softmax(scores)         # (n,n)
    return weights @ V, weights       # (n,d), (n,n)

# 4 tokens, embedding dim=3 (toy numbers)
X = np.array([
    [1.0, 0.0, 1.0],  # token 1 embedding
    [0.0, 2.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 2.0],
])

# in real transformers: Q,K,V = X @ Wq, X @ Wk, X @ Wv
Q = K = V = X

Y, W = attention(Q, K, V)
print("Attention weights:\n", np.round(W, 3))
print("Output embeddings:\n", np.round(Y, 3))
