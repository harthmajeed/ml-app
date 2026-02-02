import random
from collections import defaultdict, Counter

TEXT = """
you are learning ml ops step by step.
ml ops is about training, tracking, deploying, and monitoring models.
models drift when data changes or the world changes.
"""

def tokenize(s: str):
    # super-simple tokenization: lower + split on spaces
    return [t for t in s.lower().replace("\n", " ").split(" ") if t]

tokens = tokenize(TEXT)

# bigram counts: P(next | current)
next_counts = defaultdict(Counter)
for a, b in zip(tokens, tokens[1:]):
    next_counts[a][b] += 1

def sample_next(curr: str, temperature: float = 1.0):
    counts = next_counts.get(curr)
    if not counts:
        return random.choice(tokens)

    # turn counts into probabilities (with temperature)
    items = list(counts.items())  # (token, count)
    words, freqs = zip(*items)

    # temperature sampling (simple version): raise probs to (1/temperature)
    total = sum(freqs)
    probs = [f / total for f in freqs]
    probs = [p ** (1.0 / max(temperature, 1e-6)) for p in probs]
    z = sum(probs)
    probs = [p / z for p in probs]

    r = random.random()
    c = 0.0
    for w, p in zip(words, probs):
        c += p
        if r <= c:
            return w
    return words[-1]

def generate(start="ml", n=30, temperature=0.9):
    out = [start]
    curr = start
    for _ in range(n):
        nxt = sample_next(curr, temperature)
        out.append(nxt)
        curr = nxt
    return " ".join(out)

print(generate(start="ml", n=25, temperature=0.8))
print(generate(start="models", n=25, temperature=1.2))
