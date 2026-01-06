# GENREG Alphabet Recognition

A 32-neuron neural network trained through pure evolutionary pressure to recognize letters of the alphabet. No backpropagation. No gradients. Just selection.

## The Numbers

| Metric | Value |
|--------|-------|
| Architecture | 10,000 → 32 → 26 |
| Parameters | ~321,000 |
| Accuracy | 78.4% |
| Perfect Letters | B, F, H, K, N, P, S, T, Z |
| Generations | 134,000+ |

## How It Learns

There is no loss function. There is no gradient descent.

A population of genomes competes for survival. Each generation:
1. Every genome attempts to classify letters
2. Genomes are scored by accuracy
3. Low performers are culled
4. Survivors reproduce with mutation
5. Repeat

Over 134,000 generations, the population discovered how to compress 10,000 pixels into 32 hidden neurons while maintaining discriminative power.

## The Augmentation Gauntlet

Every letter can appear:
- Rotated ±25 degrees
- Shifted ±20% off center
- Sized at 12pt or 64pt
- White on black or black on white

No convolutional layers. No pooling. No architectural priors for spatial invariance. The network learned to handle rotation and position variation purely from selection pressure on raw pixels.

## Gradient Descent Cannot Go Here

When a neuron's output approaches ±1, the gradient of tanh approaches zero. Gradient descent gets a weaker learning signal the closer you get to saturation. The math discourages networks from using the full activation range.

Evolution does not care. If a mutation pushes a neuron to 0.999 and that genome survives better, it gets selected.

This model averages 0.994 activation magnitude across all 32 hidden neurons. Evolution explored a region of weight space that backpropagation structurally avoids.

## Files

```
best_genome.pkl                  # Trained weights
alphabet_inference_headless.py   # Inference script
```

## Run It

```bash
python alphabet_inference_headless.py best_genome.pkl

# More variations
python alphabet_inference_headless.py best_genome.pkl --variations 8

# More cycles
python alphabet_inference_headless.py best_genome.pkl --cycles 10
```

## Requirements

```bash
pip install numpy pillow torch
```

PyTorch is optional. The script falls back to pure NumPy if unavailable.

## The Forward Pass

```python
hidden = tanh(W1 @ pixels + b1)
output = softmax(W2 @ hidden + b2)
letter = argmax(output)
```

Two matrix multiplies. One nonlinearity. That is the entire model.

## Inference Speed

Tested on RTX 4080 and CPU (PyTorch):

| Device | Avg Inference | Throughput | Total (520 tests) |
|--------|---------------|------------|-------------------|
| RTX 4080 | 0.31ms | 291/sec | 1.78s |
| CPU | 0.20ms | 245/sec | 2.12s |

The model is small enough that CPU and GPU perform nearly the same. CUDA kernel launch overhead negates most GPU advantage at this scale.

This runs on anything. A Raspberry Pi could probably hit 50-100 inferences per second.

## Output

```
============================================================
TEST RESULTS SUMMARY
============================================================

Device: GPU: NVIDIA GeForce RTX 4080
  CUDA: 12.1
  GPU Memory: 16.0 GB

Overall Accuracy: 75.0%
Correct: 390 / 520

────────────────────────────────────────
INFERENCE SPEED
────────────────────────────────────────
  Total time:        1.784 s
  Throughput:        291.4 inferences/sec
  Avg inference:     0.4491 ms
  Min inference:     0.2901 ms
  Median inference:  0.3113 ms

────────────────────────────────────────
ACCURACY BREAKDOWN
────────────────────────────────────────

Perfect Letters (9/26):
  B F H K N P S T Z

Problem Letters:
  J: 25% (confused with: Q(5), T(5), W(5))
  D: 50% (confused with: I(5), W(5))
  E: 50% (confused with: Q(5), Z(5))
  G: 50% (confused with: T(5), L(5))
  I: 50% (confused with: T(5), H(5))
  R: 50% (confused with: K(5), I(5))
  X: 50% (confused with: Q(5), O(5))
  A: 75% (confused with: K(5))
  C: 75% (confused with: W(5))
```

## License

MIT

## Citation

```
@misc{genreg2026,
  author = {Payton Miller},
  title = {GENREG: Evolved Visual Representations Without Gradients},
  year = {2026}
}
```
