import modal

app = modal.App("capacity-fixed-vectors")

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "matplotlib")

@app.function(image=image, gpu="T4", timeout=3600)
def run_sweep():
    import torch
    import matplotlib.pyplot as plt

    # inline the functions so they run remotely
    def newton_schulz(M, num_iters=15):
        d = M.shape[0]
        I = torch.eye(d, device=M.device)
        X = M * (d ** 0.5) / M.norm()
        for _ in range(num_iters):
            X = 0.5 * X @ (3 * I - X.T @ X)
        return X

    def random_ortho(dim, device='cpu'):
        return newton_schulz(torch.randn(dim, dim, device=device), num_iters=50)

    def random_unit_vectors(n, dim, device='cpu'):
        vecs = torch.randn(n, dim, device=device)
        return vecs / vecs.norm(dim=1, keepdim=True)

    def evaluate(M, vecs, perm):
        outputs = (M @ vecs.T).T
        sims = outputs @ vecs.T
        predictions = sims.argmax(dim=1)
        return (predictions == perm).sum().item()

    def train(dim, n, lr=0.01, learn_vectors=False, max_steps=100000, patience=1000, device='cuda'):
        vecs = random_unit_vectors(n, dim, device=device)
        M = random_ortho(dim, device=device)
        I = torch.eye(dim, device=device)
        perm = torch.randperm(n, device=device)
        inv_perm = torch.argsort(perm)

        best_correct = 0
        steps_since_improve = 0

        for step in range(max_steps):
            outputs = M @ vecs.T
            targets = vecs[perm].T
            errors = targets - outputs

            R = I + lr * (errors @ outputs.T) / n
            M = R @ M
            M = newton_schulz(M)

            if learn_vectors:
                ideal_input = (M.T @ targets).T
                ideal_input = ideal_input / ideal_input.norm(dim=1, keepdim=True)
                ideal_output = (M @ vecs[inv_perm].T).T
                ideal_output = ideal_output / ideal_output.norm(dim=1, keepdim=True)
                vecs = vecs + lr * (ideal_input + ideal_output - 2 * vecs)
                vecs = vecs / vecs.norm(dim=1, keepdim=True)

            eval_every = max(100, patience // 10)
            if step % eval_every == 0:
                correct = evaluate(M, vecs, perm)
                if correct > best_correct:
                    best_correct = correct
                    steps_since_improve = 0
                else:
                    steps_since_improve += eval_every
                if step % (eval_every * 10) == 0:
                    print(f"  Step {step:>6}: {correct}/{n} ({100*correct/n:.1f}%)", flush=True)
                if correct == n or steps_since_improve >= patience:
                    print(f"  Step {step:>6}: {correct}/{n} ({100*correct/n:.1f}%) [saturated]", flush=True)
                    break

        return best_correct

    dims = [5, 10, 20, 50, 100, 200, 500, 1000]
    ns = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    lr = 0.1

    results = {}

    for dim in dims:
        results[dim] = {}
        for n in ns:
            if n < dim:
                continue
            correct = train(dim, n, lr=lr, learn_vectors=False)
            pct = 100 * correct / n
            results[dim][n] = pct
            print(f"dim={dim:>4}, n={n:>5}: {correct}/{n} ({pct:.1f}%)", flush=True)

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for dim in dims:
        if not results[dim]:
            continue
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

    for dim in dims:
        if not results[dim]:
            continue
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
    plt.savefig("/tmp/capacity_fixed_vectors.png", dpi=150)

    return results

@app.local_entrypoint()
def main():
    results = run_sweep.remote()
    print("\n=== Final Results ===")
    for dim in sorted(results.keys()):
        for n in sorted(results[dim].keys()):
            print(f"  dim={dim:>4}, n={n:>5}: {results[dim][n]:.1f}%")
