# zfrag

**zfrag** is a lightweight Python wrapper around the native `frag` library for fast text fingerprinting and similarity estimation. It provides a simple `Model` interface to train on text data, generate compact 64-bit signatures, and quickly compare text snippets via Hamming distance or similarity scoring.

```python
from zfrag import Model

model = Model()
model.train("Sample training text to build the model vocabulary.")

sig_a = model.sign("Hello world")
sig_b = model.sign("Hello there world")

sim = Model.similarity(sig_a, sig_b)
dist = Model.distance(sig_a, sig_b)

print(f"Similarity: {sim:.2f}, Distance: {dist}")
```
