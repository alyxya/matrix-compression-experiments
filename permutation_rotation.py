import torch

def newton_schulz(M, num_iters=15):
    """Project M onto nearest orthogonal matrix via Newton-Schulz iterations."""
    d = M.shape[0]
    I = torch.eye(d)
    # scale so singular values are in convergence range (0, sqrt(3))
    X = M * (d ** 0.5) / M.norm()
    for _ in range(num_iters):
        X = 0.5 * X @ (3 * I - X.T @ X)
    return X

def random_ortho(dim):
    return newton_schulz(torch.randn(dim, dim), num_iters=50)

def random_unit_vectors(n, dim):
    vecs = torch.randn(n, dim)
    return vecs / vecs.norm(dim=1, keepdim=True)

def evaluate(M, vecs, perm):
    outputs = (M @ vecs.T).T  # (n, dim)
    sims = outputs @ vecs.T   # (n, n)
    predictions = sims.argmax(dim=1)
    correct = (predictions == perm).sum().item()
    return correct

def train(dim, n, steps, lr=0.01, learn_vectors=False):
    vecs = random_unit_vectors(n, dim)
    M = random_ortho(dim)
    perm = torch.randperm(n)
    inv_perm = torch.argsort(perm)

    for step in range(steps):
        outputs = M @ vecs.T          # (dim, n)
        targets = vecs[perm].T        # (dim, n)
        errors = targets - outputs    # (dim, n)

        # matrix update: accumulate small rotations across all pairs
        # for each pair: R_i = I + lr * (target - output) @ output^T
        # batched average: R = I + lr/n * errors @ outputs^T
        R = torch.eye(dim) + lr * (errors @ outputs.T) / n
        M = R @ M
        M = newton_schulz(M)

        if learn_vectors:
            # as input: rotate v_i toward M^T @ v_{perm[i]}
            ideal_input = (M.T @ targets).T              # (n, dim)
            ideal_input = ideal_input / ideal_input.norm(dim=1, keepdim=True)

            # as output: rotate v_i toward M @ v_{inv_perm[i]}
            ideal_output = (M @ vecs[inv_perm].T).T       # (n, dim)
            ideal_output = ideal_output / ideal_output.norm(dim=1, keepdim=True)

            # average both signals
            vecs = vecs + lr * (ideal_input + ideal_output - 2 * vecs)
            vecs = vecs / vecs.norm(dim=1, keepdim=True)

        print_every = max(500, steps // 10)
        if step % print_every == 0 or step == steps - 1:
            correct = evaluate(M, vecs, perm)
            print(f"  Step {step:>5}: {correct}/{n} ({100*correct/n:.1f}%)")

    return evaluate(M, vecs, perm)


if __name__ == "__main__":
    dim = 10
    n = 100
    steps = 5000

    print(f"=== Fixed vectors (dim={dim}, n={n}) ===")
    train(dim, n, steps, learn_vectors=False)

    print(f"\n=== Learned vectors (dim={dim}, n={n}) ===")
    train(dim, n, steps, learn_vectors=True)
