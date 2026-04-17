import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

plt.plot(z, sigmoid_values)
plt.title("Sigmoid function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid()
plt.show()