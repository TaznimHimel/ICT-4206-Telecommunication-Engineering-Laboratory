import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Parameters
# ------------------------------------------------
bits = np.array([1, 0, 1, 1, 0])
pn = np.array([1, 1, -1, 1, -1, 1, -1])   # 7-chip spreading code

fs = 1000     # Sampling frequency (Hz)
Tb = 1        # Bit duration (s)
fc = 5        # Carrier frequency (Hz)

data = 2 * bits - 1                       # 1 -> +1, 0 -> -1
t_bit = np.arange(0, Tb, 1 / fs)          # samples per bit (exact: fs * Tb)

# ------------------------------------------------
# 1. Original bit sequence
# ------------------------------------------------
bit_signal = np.repeat(bits, len(t_bit))

# ------------------------------------------------
# 2. BPSK modulation
# ------------------------------------------------
bpsk = np.concatenate([
    b * np.cos(2 * np.pi * fc * t_bit)
    for b in data
])

# ------------------------------------------------
# 3. Spreading: each bit -> 7 chips (PN sequence)
# ------------------------------------------------
spread_chips = np.concatenate([b * pn for b in data])   # length = len(bits) * len(pn)

# Split each bit's samples into 7 near-equal groups (sizes sum exactly to
# len(t_bit), even when fs isn't a multiple of len(pn) -> no dropped/extra
# samples, no drift).
chip_index_groups = np.array_split(np.arange(len(t_bit)), len(pn))
samples_per_chip_list = [len(g) for g in chip_index_groups]   # e.g. [143,143,143,143,143,143,142]

spread = np.concatenate([
    np.repeat(pn, samples_per_chip_list) * b
    for b in data
])

# ------------------------------------------------
# 4. DSSS signal: spread chips riding on the carrier
# ------------------------------------------------
dsss = spread * np.concatenate([np.cos(2 * np.pi * fc * t_bit) for _ in data])

# ------------------------------------------------
# Time axes (all exact, all aligned)
# ------------------------------------------------
t_bits = np.arange(len(bit_signal)) / fs
t_bpsk = np.arange(len(bpsk)) / fs
t_spread = np.arange(len(spread)) / fs
t_dsss = np.arange(len(dsss)) / fs

# ------------------------------------------------
# Plot
# ------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

ax[0].step(t_bits, bit_signal, where="post")
ax[0].set_title("Original Bit Sequence")
ax[0].set_ylabel("Bit")
ax[0].set_ylim(-1.2, 1.2)
ax[0].grid(alpha=0.3)

ax[1].plot(t_bpsk, bpsk)
ax[1].set_title("BPSK-Modulated Signal")
ax[1].set_ylabel("Amplitude")
ax[1].grid(alpha=0.3)

ax[2].step(t_spread, spread, where="post")
ax[2].set_title("Spread Signal (7-Chip PN Sequence)")
ax[2].set_ylabel("Chip value")
ax[2].grid(alpha=0.3)

ax[3].plot(t_dsss, dsss)
ax[3].set_title("Direct Sequence Spread Spectrum (DSSS) Signal")
ax[3].set_xlabel("Time (s)")
ax[3].set_ylabel("Amplitude")
ax[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("dsss_final.png", dpi=150)
plt.show()

print("Original bits :", bits)
print("PN code       :", pn)
print("len(dsss)     :", len(dsss), " -> duration:", len(dsss) / fs, "s (matches", len(bits) * Tb, "s)")
