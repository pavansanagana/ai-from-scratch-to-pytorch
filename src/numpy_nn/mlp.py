import numpy as np

def softmax(logits):
    # stabilize softmax to prevent overflow
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(probs, y):
    # probs shape: (N, C), y shape: (N,)
    N = y.shape[0]
    return -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

class MLP:
    """
    2-layer neural network:
    X -> Linear -> ReLU -> Linear -> Softmax
    Includes backprop + Adam optimizer.
    """
    def __init__(self, in_dim=784, hidden=256, out_dim=10, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.02, size=(in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros((1, hidden), dtype=np.float32)
        self.W2 = rng.normal(0, 0.02, size=(hidden, out_dim)).astype(np.float32)
        self.b2 = np.zeros((1, out_dim), dtype=np.float32)

        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t = 0

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        p = softmax(z2)
        cache = (X, z1, a1, z2, p)
        return p, cache

    def backward(self, cache, y):
        X, z1, a1, z2, p = cache
        N = X.shape[0]

        # gradient of softmax+cross entropy
        dz2 = p.copy()
        dz2[np.arange(N), y] -= 1
        dz2 /= N

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)  # ReLU backprop

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for name, param in self.params().items():
            g = grads[name]
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g * g)

            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)

            update = lr * m_hat / (np.sqrt(v_hat) + eps)

            if name == "W1": self.W1 -= update
            if name == "b1": self.b1 -= update
            if name == "W2": self.W2 -= update
            if name == "b2": self.b2 -= update

    def predict(self, X):
        p, _ = self.forward(X)
        return np.argmax(p, axis=1)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        d = np.load(path)
        self.W1, self.b1, self.W2, self.b2 = d["W1"], d["b1"], d["W2"], d["b2"]