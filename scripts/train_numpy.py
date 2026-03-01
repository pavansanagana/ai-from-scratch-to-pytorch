import time
import numpy as np
from tqdm import tqdm

from src.numpy_nn.mlp import MLP, cross_entropy
from src.utils.mnist import get_mnist

def to_numpy(batch):
    x, y = batch
    x = x.view(x.shape[0], -1).numpy().astype(np.float32)
    y = y.numpy().astype(np.int64)
    return x, y

def accuracy(model, loader, limit_batches=None):
    correct = 0
    total = 0
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x, y = to_numpy(batch)
        pred = model.predict(x)
        correct += (pred == y).sum()
        total += y.shape[0]
    return correct / total

def main():
    batch_size = 256
    epochs = 3
    lr = 1e-3

    train_loader, test_loader = get_mnist(batch_size=batch_size)
    model = MLP(in_dim=784, hidden=256, out_dim=10)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            x, y = to_numpy(batch)
            p, cache = model.forward(x)
            loss = cross_entropy(p, y)
            grads = model.backward(cache, y)
            model.adam_step(grads, lr=lr)
            losses.append(loss)

        train_acc = accuracy(model, train_loader, limit_batches=200)
        test_acc = accuracy(model, test_loader)
        dt = time.time() - t0
        print(f"Epoch {ep}: loss={np.mean(losses):.4f} train_acc≈{train_acc:.4f} test_acc={test_acc:.4f} time={dt:.1f}s")

    model.save("models/numpy_mlp_mnist.npz")
    print("Saved: models/numpy_mlp_mnist.npz")

if __name__ == "__main__":
    main()