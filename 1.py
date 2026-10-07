import numpy as np
import matplotlib.pyplot as plt

# =====================================
# Parameters
# =====================================
fm = 5          # Message frequency (Hz)
fs = 100        # Sampling frequency (Hz)
duration = 1    # Duration (s)
bits = 4        # Bits per sample

# =====================================
# Original Analog Signal
# =====================================
t = np.linspace(0, duration, 1000)
analog = np.sin(2 * np.pi * fm * t)

# =====================================
# Sampling
# =====================================
ts = np.arange(0, duration, 1/fs)
samples = np.sin(2 * np.pi * fm * ts)

# =====================================
# Quantization
# =====================================
levels = 2 ** bits

# Quantization indices (0 to levels-1)
q_index = np.round((samples + 1) * (levels - 1) / 2).astype(int)

# Quantized amplitudes
quantized = (q_index / (levels - 1)) * 2 - 1

# =====================================
# PCM Encoding
# =====================================
pcm_codes = [format(i, f'0{bits}b') for i in q_index]

print("\nPCM Encoded Binary Stream:\n")
print(" ".join(pcm_codes))

# Complete bit stream
bit_stream = [int(bit) for code in pcm_codes for bit in code]

# =====================================
# Demodulation (Reconstruction)
# =====================================
reconstructed = np.interp(t, ts, quantized)

# =====================================
# Plotting
# =====================================
plt.figure(figsize=(12, 12))

# 1. Analog Signal
plt.subplot(5,1,1)
plt.plot(t, analog, 'b', linewidth=2)
plt.title("Original Analog Signal")
plt.ylabel("Amplitude")
plt.grid(True)

# 2. Sampled Signal
plt.subplot(5,1,2)
plt.plot(t, analog, color='gray', alpha=0.5)
plt.stem(ts, samples, linefmt='r-', markerfmt='ro', basefmt=' ')
plt.title("Sampled Signal")
plt.ylabel("Amplitude")
plt.grid(True)

# 3. Quantized Signal
plt.subplot(5,1,3)
plt.step(ts, quantized, where='mid', color='g', linewidth=2)
plt.scatter(ts, quantized, color='black', s=20)
plt.title("Quantized Signal")
plt.ylabel("Amplitude")
plt.grid(True)

# 4. PCM Bit Stream
plt.subplot(5,1,4)
plt.step(np.arange(len(bit_stream)+1),
         bit_stream+[bit_stream[-1]],
         where='post',
         linewidth=2)

plt.ylim(-0.2,1.2)
plt.xlim(0,len(bit_stream))
plt.yticks([0,1])
plt.title("PCM Encoded Bit Stream")
plt.ylabel("Bit")
plt.grid(True)

# 5. Demodulated Signal
plt.subplot(5,1,5)
plt.plot(t, analog, '--', color='gray', label='Original')
plt.plot(t, reconstructed, 'm', linewidth=2, label='Reconstructed')
plt.scatter(ts, quantized, color='black', s=15)
plt.title("PCM Demodulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()