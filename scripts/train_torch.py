import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.utils.mnist import get_mnist


class TorchMLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    batch_size = 256
    epochs = 3
    lr = 1e-3

    train_loader, test_loader = get_mnist(batch_size=batch_size)

    model = TorchMLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            x = x.view(x.size(0), -1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_acc = accuracy(model, train_loader)
        test_acc = accuracy(model, test_loader)
        dt = time.time() - t0

        print(
            f"Epoch {ep}: loss={sum(losses)/len(losses):.4f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} time={dt:.2f}s"
        )

    torch.save(model.state_dict(), "models/torch_mlp_mnist.pt")
    print("Saved: models/torch_mlp_mnist.pt")


if __name__ == "__main__":
    main()
