import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ActivationFunction:
    def activate(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)

class ReLU(ActivationFunction):
    def activate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)

class ActivationFactory:
    @staticmethod
    def get_activation(name):
        if name == "sigmoid":
            return Sigmoid()
        elif name == "relu":
            return ReLU()
        else:
            raise ValueError("Activation function not supported")

class TrainingObserver:
    def update(self, epoch, loss):
        raise NotImplementedError

class LossLogger(TrainingObserver):
    def update(self, epoch, loss):
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

class TrainingMonitor:
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify(self, epoch, loss):
        for observer in self.observers:
            observer.update(epoch, loss)

class MLPClassification:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, momentum=0.9, regularization=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.monitor = TrainingMonitor()
        self.monitor.add_observer(LossLogger())

        self.weights = [np.random.randn(input_size, hidden_sizes[0]) * 0.1]
        self.biases = [np.zeros((1, hidden_sizes[0]))]

        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * 0.1)
            self.biases.append(np.zeros((1, hidden_sizes[i])))

        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.1)
        self.biases.append(np.zeros((1, output_size)))
        self.velocities_w = [np.zeros_like(w) for w in self.weights]
        self.velocities_b = [np.zeros_like(b) for b in self.biases]

        self.activation_hidden = ActivationFactory.get_activation("sigmoid")
        self.activation_output = ActivationFactory.get_activation("sigmoid")

    def forward(self, x):
        activations = [x]
        pre_activations = []

        for i in range(len(self.hidden_sizes)):
            z = activations[-1].dot(self.weights[i]) + self.biases[i]
            a = self.activation_hidden.activate(z)
            pre_activations.append(z)
            activations.append(a)

        z = activations[-1].dot(self.weights[-1]) + self.biases[-1]
        a = self.activation_output.activate(z)
        pre_activations.append(z)
        activations.append(a)
        return activations, pre_activations

    def backward(self, x, y, activations, pre_activations):
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)
        delta = activations[-1] - y
        gradients_w[-1] = activations[-2].T.dot(delta) / x.shape[0] + self.regularization * self.weights[-1]
        gradients_b[-1] = np.sum(delta, axis=0, keepdims=True) / x.shape[0]

        for i in range(len(self.hidden_sizes) - 1, -1, -1):
            delta = delta.dot(self.weights[i + 1].T) * self.activation_hidden.derivative(activations[i + 1])
            gradients_w[i] = activations[i].T.dot(delta) / x.shape[0] + self.regularization * self.weights[i]
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True) / x.shape[0]
        return gradients_w, gradients_b

    def update_parameters(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.velocities_w[i] = self.momentum * self.velocities_w[i] - self.learning_rate * gradients_w[i]
            self.weights[i] += self.velocities_w[i]
            self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.learning_rate * gradients_b[i]
            self.biases[i] += self.velocities_b[i]

    def fit(self, x, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            activations, pre_activations = self.forward(x)
            loss = np.mean((activations[-1] - y) ** 2)
            losses.append(loss)
            gradients_w, gradients_b = self.backward(x, y, activations, pre_activations)
            self.update_parameters(gradients_w, gradients_b)
            self.monitor.notify(epoch, loss)
        return losses

    def predict(self, x):
        activations, _ = self.forward(x)
        return (activations[-1] > 0.5).astype(int)

class MLPBuilder:
    def __init__(self):
        self.input_size = None
        self.hidden_sizes = []
        self.output_size = None
        self.learning_rate = 0.01

    def set_input_size(self, size):
        self.input_size = size
        return self

    def add_hidden_layer(self, size):
        self.hidden_sizes.append(size)
        return self

    def set_output_size(self, size):
        self.output_size = size
        return self

    def set_learning_rate(self, lr):
        self.learning_rate = lr
        return self

    def build(self):
        return MLPClassification(self.input_size, self.hidden_sizes, self.output_size, self.learning_rate)

data = load_iris()
x = data['data'][:, (2, 3)]
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = (data['target'] == 2).astype(int).reshape(-1, 1)

mlp = MLPBuilder().set_input_size(2).add_hidden_layer(4).add_hidden_layer(4).set_output_size(1).set_learning_rate(0.01).build()
losses = mlp.fit(x, y, epochs=1000)
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
y_pred = mlp.predict(x)
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.2f}")
