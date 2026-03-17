import torch
import matplotlib.pyplot as plt

n_vectors = 10000
dim = 100

vecs = torch.randn(n_vectors, dim)
vecs = vecs / vecs.norm(dim=1, keepdim=True)

# cosine similarity = dot product for unit vectors
sims = vecs @ vecs.T

# extract upper triangle (exclude self-similarities)
idx = torch.triu_indices(n_vectors, n_vectors, offset=1)
pairwise_sims = sims[idx[0], idx[1]]

plt.figure(figsize=(10, 6))
plt.hist(pairwise_sims.numpy(), bins=200, alpha=0.7)
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.title(f"Cosine Similarity Distribution ({n_vectors} random unit vectors, d={dim})")
plt.tight_layout()
plt.savefig("images/cosine_sim_dist.png", dpi=150)
plt.show()
print(f"Mean: {pairwise_sims.mean():.4f}, Std: {pairwise_sims.std():.4f}")
