import numpy as np


class ReLU:
    def forward(self, x):
        self.mask = (x > 0).astype(x.dtype)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


class Sigmoid:
    def forward(self, x):
        x = np.clip(x, -500, 500)
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1.0 - self.out ** 2)


ACTIVATIONS = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
}


def softmax(logits):
    logits = np.clip(logits, -500, 500)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, labels):
    n = labels.shape[0]
    clipped = np.clip(probs[np.arange(n), labels], 1e-12, 1.0)
    log_probs = -np.log(clipped)
    return log_probs.mean()


class ThreeLayerMLP:
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, activation='relu'):
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim

        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden1_dim)
        scale3 = np.sqrt(2.0 / hidden2_dim)

        self.W1 = np.random.randn(input_dim, hidden1_dim) * scale1
        self.b1 = np.zeros((1, hidden1_dim))
        self.W2 = np.random.randn(hidden1_dim, hidden2_dim) * scale2
        self.b2 = np.zeros((1, hidden2_dim))
        self.W3 = np.random.randn(hidden2_dim, output_dim) * scale3
        self.b3 = np.zeros((1, output_dim))

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(ACTIVATIONS.keys())}")
        self.act1 = ACTIVATIONS[activation]()
        self.act2 = ACTIVATIONS[activation]()
        self.activation_name = activation

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.act1.forward(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.act2.forward(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.probs = softmax(self.Z3)
        return self.probs

    def backward(self, labels, weight_decay=0.0, grad_clip=5.0):
        n = labels.shape[0]

        dZ3 = self.probs.copy()
        dZ3[np.arange(n), labels] -= 1.0
        dZ3 /= n

        self.dW3 = self.A2.T @ dZ3 + weight_decay * self.W3
        self.db3 = dZ3.sum(axis=0, keepdims=True)

        dA2 = dZ3 @ self.W3.T
        dZ2 = self.act2.backward(dA2)

        self.dW2 = self.A1.T @ dZ2 + weight_decay * self.W2
        self.db2 = dZ2.sum(axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = self.act1.backward(dA1)

        self.dW1 = self.X.T @ dZ1 + weight_decay * self.W1
        self.db1 = dZ1.sum(axis=0, keepdims=True)

        if grad_clip > 0:
            for attr in ('dW1', 'db1', 'dW2', 'db2', 'dW3', 'db3'):
                g = getattr(self, attr)
                norm = np.linalg.norm(g)
                if norm > grad_clip:
                    setattr(self, attr, g * (grad_clip / norm))

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def get_params(self):
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy(),
            'W3': self.W3.copy(), 'b3': self.b3.copy(),
        }

    def set_params(self, params):
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
        self.W3 = params['W3'].copy()
        self.b3 = params['b3'].copy()

    def save(self, filepath):
        params = self.get_params()
        params['config'] = {
            'input_dim': self.input_dim,
            'hidden1_dim': self.hidden1_dim,
            'hidden2_dim': self.hidden2_dim,
            'output_dim': self.output_dim,
            'activation': self.activation_name,
        }
        np.savez(filepath, **params)

    @classmethod
    def load(cls, filepath):
        data = np.load(filepath, allow_pickle=True)
        config = data['config'].item()
        model = cls(
            config['input_dim'], config['hidden1_dim'],
            config['hidden2_dim'], config['output_dim'],
            config['activation']
        )
        model.set_params({
            'W1': data['W1'], 'b1': data['b1'],
            'W2': data['W2'], 'b2': data['b2'],
            'W3': data['W3'], 'b3': data['b3'],
        })
        return model