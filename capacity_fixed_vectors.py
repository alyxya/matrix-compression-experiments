import torch
import matplotlib.pyplot as plt
from permutation_rotation import train, random_unit_vectors, random_ortho, newton_schulz, evaluate

dims = [5, 10, 20, 50, 100, 200, 500, 1000]
ns = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
steps = 10000
lr = 0.1

results = {}

for dim in dims:
    results[dim] = {}
    for n in ns:
        if n < dim:
            continue
        correct = train(dim, n, steps, lr=lr, learn_vectors=False)
        pct = 100 * correct / n
        results[dim][n] = pct
        print(f"dim={dim:>3}, n={n:>4}: {correct}/{n} ({pct:.1f}%)", flush=True)

# Plot 1: accuracy vs n for each dim
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for dim in dims:
    ns_dim = sorted(results[dim].keys())
    accs = [results[dim][n] for n in ns_dim]
    ax1.plot(ns_dim, accs, 'o-', label=f"dim={dim}", linewidth=2)

ax1.set_xlabel("Number of vectors (n)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Fixed vectors, learned orthogonal matrix")
ax1.set_xscale("log")
ax1.xaxis.set_major_formatter(plt.ScalarFormatter())
ax1.xaxis.set_minor_formatter(plt.NullFormatter())
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: accuracy vs n/dim ratio
for dim in dims:
    ns_dim = sorted(results[dim].keys())
    ratios = [n / dim for n in ns_dim]
    accs = [results[dim][n] for n in ns_dim]
    ax2.plot(ratios, accs, 'o-', label=f"dim={dim}", linewidth=2)

ax2.set_xlabel("n / dim")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Fixed vectors, learned orthogonal matrix (normalized by dim)")
ax2.set_xscale("log")
ax2.xaxis.set_major_formatter(plt.ScalarFormatter())
ax2.xaxis.set_minor_formatter(plt.NullFormatter())
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("images/capacity_fixed_vectors.png", dpi=150)
plt.show()
