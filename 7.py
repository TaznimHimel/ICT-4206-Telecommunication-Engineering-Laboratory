import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Parameters
# ------------------------------------------------
bits = np.array([1, 0, 1, 1, 0, 1, 0])
Tb = 1                              # Bit duration (s)
fs = 1000                           # Sampling frequency (Hz)
fc = 10                             # BPSK carrier frequency (Hz)

freqs = np.array([20, 40, 60, 80, 100, 120, 140])   # 7 spreading frequencies

t_bit = np.arange(0, Tb, 1 / fs)
t = np.arange(0, len(bits) * Tb, 1 / fs)

# ------------------------------------------------
# 1. Original bit sequence
# ------------------------------------------------
data = np.repeat(bits, len(t_bit))

# ------------------------------------------------
# 2. BPSK modulation (1 -> +1, 0 -> -1)
# ------------------------------------------------
bpsk = np.concatenate([
    (2 * bit - 1) * np.cos(2 * np.pi * fc * t_bit)
    for bit in bits
])

# ------------------------------------------------
# 3. Frequency-hopping pattern (one hop per bit)
# ------------------------------------------------
np.random.seed(7)
hop = np.random.choice(freqs, len(bits))

# Spread signal: which frequency is active at each instant (staircase)
spread = np.repeat(hop, len(t_bit))

# ------------------------------------------------
# 4. FHSS signal: BPSK data riding on the hopped carrier
# ------------------------------------------------
fhss = np.concatenate([
    (2 * bit - 1) * np.cos(2 * np.pi * f * t_bit)
    for bit, f in zip(bits, hop)
])

# ------------------------------------------------
# Plot
# ------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

ax[0].step(t, data, where="post")
ax[0].set_title("Original Bit Sequence")
ax[0].set_ylabel("Bit")
ax[0].set_ylim(-0.2, 1.2)
ax[0].grid(alpha=0.3)

ax[1].plot(t, bpsk)
ax[1].set_title("BPSK-Modulated Signal")
ax[1].set_ylabel("Amplitude")
ax[1].grid(alpha=0.3)

ax[2].step(t, spread, where="post")
ax[2].set_title("Spread Signal — Hopping Pattern (7 Frequencies)")
ax[2].set_ylabel("Frequency (Hz)")
ax[2].grid(alpha=0.3)

ax[3].plot(t, fhss)
ax[3].set_title("Frequency-Hopped Spread Spectrum (FHSS) Signal")
ax[3].set_xlabel("Time (s)")
ax[3].set_ylabel("Amplitude")
ax[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fhss_final.png", dpi=150)
plt.show()

print("Original bits :", bits)
print("Hop sequence  :", hop)
