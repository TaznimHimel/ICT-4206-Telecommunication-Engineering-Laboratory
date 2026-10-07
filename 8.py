# Convolutional Coding + BPSK + AWGN + Viterbi Decoding + BER
# Rate 1/2, constraint length K=3, generators (7,5) octal
# No communication library used. matplotlib is used only for drawing.

import random, math
import matplotlib.pyplot as plt

# ---------- 1. Encoder (state machine) ----------
# state = (s1, s2) = last two input bits, stored as number 0..3
def step(state, b):
    s1 = state // 2
    s2 = state % 2
    o1 = (b + s1 + s2) % 2        # generator 111  (7)
    o2 = (b + s2) % 2             # generator 101  (5)
    next_state = b * 2 + s1
    return next_state, o1, o2

def encode(bits):
    state = 0
    out = []
    for b in bits:
        state, o1, o2 = step(state, b)
        out.append(o1)
        out.append(o2)
    return out

# ---------- 2. BPSK + AWGN channel ----------
def bpsk(bits):                    # 0 -> -1,  1 -> +1
    return [2 * b - 1 for b in bits]

def noise(sigma):                  # Gaussian noise (Box-Muller)
    u1 = 1.0 - random.random()
    u2 = random.random()
    return sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

def awgn(signal, sigma):
    return [x + noise(sigma) for x in signal]

def hard_decision(rx):             # > 0 means bit 1
    return [1 if x > 0 else 0 for x in rx]

# ---------- 3. Viterbi decoder (hard decision, Hamming distance) ----------
def viterbi(rx_bits):
    n = len(rx_bits) // 2
    BIG = 10**9
    cost = [0, BIG, BIG, BIG]      # start in state 0
    history = []                   # history[t][state] = (previous state, input bit)

    for t in range(n):
        r1 = rx_bits[2 * t]
        r2 = rx_bits[2 * t + 1]
        new_cost = [BIG] * 4
        back = [None] * 4
        for s in range(4):
            if cost[s] >= BIG:
                continue
            for b in (0, 1):
                ns, o1, o2 = step(s, b)
                d = (o1 != r1) + (o2 != r2)        # Hamming distance
                c = cost[s] + d
                if c < new_cost[ns]:               # keep the survivor
                    new_cost[ns] = c
                    back[ns] = (s, b)
        cost = new_cost
        history.append(back)

    # traceback: message ends in state 0 (we added 2 tail zeros)
    state = 0
    decoded = []
    path = [0]
    for t in range(n - 1, -1, -1):
        prev, b = history[t][state]
        decoded.append(b)
        path.append(prev)
        state = prev
    decoded.reverse()
    path.reverse()
    return decoded, path

# ---------- 4. One full transmission ----------
def transmit(msg, ebn0_db, coded=True):
    ebn0 = 10 ** (ebn0_db / 10)
    if coded:
        tx_bits = encode(msg + [0, 0])           # add 2 tail zeros
        sigma = math.sqrt(1 / (2 * 0.5 * ebn0))  # rate 1/2
    else:
        tx_bits = msg
        sigma = math.sqrt(1 / (2 * ebn0))
    tx = bpsk(tx_bits)
    rx = awgn(tx, sigma)
    rx_bits = hard_decision(rx)
    if coded:
        decoded, path = viterbi(rx_bits)
        decoded = decoded[:len(msg)]             # remove tail
    else:
        decoded, path = rx_bits, None
    return tx_bits, tx, rx, rx_bits, decoded, path

def count_errors(a, b):
    e = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            e += 1
    return e

# ================= DEMO (small example, to visualize) =================
random.seed(1)
msg = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
tx_bits, tx, rx, rx_bits, decoded, path = transmit(msg, ebn0_db=4)
print("Message  :", msg)
print("Encoded  :", tx_bits)
print("Received :", rx_bits)
print("Decoded  :", decoded)
print("Channel bit errors:", count_errors(tx_bits, rx_bits),
      "| Errors after decoding:", count_errors(msg, decoded))

# ---- Figure 1: signal at every block ----
fig, ax = plt.subplots(6, 1, figsize=(11, 11))
n2 = len(tx_bits)

ax[0].step(range(len(msg) + 1), msg + [msg[-1]], where="post", color="tab:blue")
ax[0].set_title("1. Message bits")

ax[1].step(range(n2 + 1), tx_bits + [tx_bits[-1]], where="post", color="tab:green")
ax[1].set_title("2. Convolutional encoded bits (rate 1/2, 2 output bits per input bit)")

ax[2].step(range(n2 + 1), tx + [tx[-1]], where="post", color="tab:orange")
ax[2].set_title("3. BPSK signal (0 -> -1, 1 -> +1)")

ax[3].stem(range(n2), rx)
ax[3].axhline(0, color="gray", linewidth=0.8)
ax[3].set_title("4. Received signal after AWGN channel (Eb/N0 = 4 dB)")

ax[4].step(range(n2 + 1), rx_bits + [rx_bits[-1]], where="post", color="tab:red")
for i in range(n2):                                    # mark channel errors
    if rx_bits[i] != tx_bits[i]:
        ax[4].plot(i + 0.5, rx_bits[i], "kx", markersize=10)
ax[4].set_title("5. Hard decision bits (x = channel error)")

ax[5].step(range(len(msg) + 1), msg + [msg[-1]], where="post",
           color="tab:blue", label="Sent")
ax[5].step(range(len(decoded) + 1), decoded + [decoded[-1]], where="post",
           color="tab:red", linestyle="--", label="Decoded")
ax[5].legend()
ax[5].set_title("6. Viterbi decoded bits vs sent bits")

for a in ax:
    a.grid(True, alpha=0.3)
plt.tight_layout()

# ---- Figure 2: Trellis diagram with the decoded path ----
fig2, ax2 = plt.subplots(figsize=(12, 5))
n = len(path) - 1
for t in range(n):
    for s in range(4):
        for b in (0, 1):
            ns, o1, o2 = step(s, b)
            style = "--" if b == 0 else "-"            # dashed = input 0, solid = input 1
            ax2.plot([t, t + 1], [s, ns], style, color="lightgray", zorder=1)
for t in range(n):                                     # decoded (survivor) path
    ax2.plot([t, t + 1], [path[t], path[t + 1]], color="red", linewidth=3, zorder=2)
for t in range(n + 1):
    for s in range(4):
        ax2.plot(t, s, "o", color="black", zorder=3)
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels(["00", "01", "10", "11"])
ax2.set_xlabel("Time step")
ax2.set_ylabel("State (s1 s2)")
ax2.set_title("Trellis diagram: red = path chosen by Viterbi (dashed = input 0, solid = input 1)")
ax2.grid(True, alpha=0.2)

# ================= BER vs SNR =================
random.seed(1)
snr_list = [0, 1, 2, 3, 4, 5, 6, 7]
N = 50000                                              # bits per SNR point
ber_coded = []
ber_uncoded = []
for snr in snr_list:
    m = [1 if random.random() > 0.5 else 0 for _ in range(N)]
    dec_c = transmit(m, snr, coded=True)[4]
    dec_u = transmit(m, snr, coded=False)[4]
    ber_coded.append(count_errors(m, dec_c) / N)
    ber_uncoded.append(count_errors(m, dec_u) / N)
    print("Eb/N0 =", snr, "dB | uncoded BER =", ber_uncoded[-1],
          "| coded BER =", ber_coded[-1])

# ---- Figure 3: BER curve ----
fig3, ax3 = plt.subplots(figsize=(8, 5.5))
ax3.semilogy(snr_list, ber_uncoded, "o-", color="tab:blue", label="Uncoded BPSK")
ax3.semilogy(snr_list, [max(b, 1e-6) for b in ber_coded], "s-",
             color="tab:red", label="Convolutional (1/2, K=3) + Viterbi")
ax3.set_xlabel("Eb/N0 (dB)")
ax3.set_ylabel("Bit Error Rate (BER)")
ax3.set_title("BER vs Eb/N0")
ax3.grid(True, which="both", alpha=0.3)
ax3.legend()

plt.show()