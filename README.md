# Digital Communication System Experiments

A collection of Python-based digital communication experiments covering PCM, spread-spectrum techniques, and convolutional coding.

## Experiments

| Experiment | File | Description |
|---|---|---|
| 1 | `1.py` | Pulse Code Modulation (PCM): sampling, quantization, PCM encoding, and signal reconstruction |
| 6 | `6.py` | Direct Sequence Spread Spectrum (DSSS) using BPSK and a 7-chip PN sequence |
| 7 | `7.py` | Frequency Hopping Spread Spectrum (FHSS) using BPSK and seven hopping frequencies |
| 8 | `8.py` | Convolutional coding + BPSK + AWGN + Viterbi decoding with BER performance analysis |

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install the required packages with:

```bash
pip install numpy matplotlib
```

## How to Run

Run each experiment separately:

```bash
python 1.py
python 6.py
python 7.py
python 8.py
```

## Experiment Details

### Experiment 1 — PCM

The program generates an analog sine-wave signal and performs:

- Sampling
- Quantization
- PCM binary encoding
- Signal reconstruction

The script displays the original analog signal, sampled signal, quantized signal, PCM bit stream, and reconstructed signal.

### Experiment 6 — DSSS

The program demonstrates Direct Sequence Spread Spectrum using:

- Input bits
- BPSK modulation
- A 7-chip PN spreading sequence
- DSSS signal generation

The script also saves the generated plot as:

```text
dsss_final.png
```

### Experiment 7 — FHSS

The program demonstrates Frequency Hopping Spread Spectrum using:

- Input bits
- BPSK modulation
- Seven available hopping frequencies
- One frequency hop per bit

The hopping pattern is generated using a fixed random seed for reproducible results.

The script saves the generated plot as:

```text
fhss_final.png
```

### Experiment 8 — Convolutional Coding

The program implements a complete communication chain:

```text
Message Bits
     ↓
Convolutional Encoder
     ↓
BPSK Modulation
     ↓
AWGN Channel
     ↓
Hard Decision
     ↓
Viterbi Decoder
     ↓
Decoded Bits
```

The implementation uses a rate `1/2` convolutional code with constraint length `K = 3` and generator polynomials `(7,5)` in octal.

The experiment also compares coded and uncoded BPSK performance using BER versus `Eb/N0`.

Three visualizations are produced:

1. Signal flow from message bits to decoded bits
2. Trellis diagram with the Viterbi-selected path
3. BER versus `Eb/N0` curve

The convolutional coding implementation does not use a communication-system library; the encoder, AWGN noise generation, hard decision, Viterbi decoder, and BER calculation are implemented directly in Python.

## Reference

The repository includes lecture material on convolutional coding from MIT 6.02, covering convolutional encoders, parity equations, state-machine representation, trellis structures, maximum-likelihood decoding, and Viterbi decoding.
