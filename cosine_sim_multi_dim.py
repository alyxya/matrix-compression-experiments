import torch
import matplotlib.pyplot as plt

n_vectors = 5000
dims = [10, 50, 100, 500, 1000, 5000]

plt.figure(figsize=(12, 7))

# plot widest first (back) to narrowest last (front)
for dim in dims:
    vecs = torch.randn(n_vectors, dim)
    vecs = vecs / vecs.norm(dim=1, keepdim=True)
    sims = vecs @ vecs.T
    idx = torch.triu_indices(n_vectors, n_vectors, offset=1)
    pairwise_sims = sims[idx[0], idx[1]]
    plt.hist(pairwise_sims.numpy(), bins=200, density=True, label=f"d={dim}", edgecolor="none")
    print(f"d={dim:>5}: mean={pairwise_sims.mean():.4f}, std={pairwise_sims.std():.4f}")

plt.xlim(-0.4, 0.4)
plt.xlabel("Cosine Similarity")
plt.ylabel("Probability Density (area under each curve = 1)")
plt.title("Cosine Similarity Distribution by Dimension (random unit vectors)")
plt.legend()
plt.tight_layout()
plt.savefig("images/cosine_sim_multi_dim.png", dpi=150)
plt.show()
