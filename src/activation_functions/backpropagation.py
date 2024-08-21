from typing import Callable, Tuple

import numpy as np
import mlflow

from relu import relu, relu_derivative
from sigmoid import sigmoid, sigmoid_derivative
from tanh import tanh, tanh_derivative


# Configurando o MLflow
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def run_backpropagation(algorithm: Tuple[Callable, Callable]):
    mlflow.set_experiment(f"Backpropagation for {algorithm[0].__name__}")
    with mlflow.start_run():
        # Exemplo de backpropagation
        x = np.array([[0.1, 0.2], [0.4, 0.5]])
        y_true = np.array([[0.3], [0.9]])
        weights = np.array([[0.1], [0.2]])

        # Forward pass
        z = np.dot(x, weights)
        y_pred = algorithm[0](z)

        error = mse_loss(y_true, y_pred)
        print("Erro inicial:", error)

        # Backward pass
        gradient = np.dot(x.T, algorithm[1](z) * (y_pred - y_true))

        # Logando o gradiente
        mlflow.log_metric("initial_error", error)
        mlflow.log_param("initial_weights", weights.tolist())
        mlflow.log_param("gradient", gradient.tolist())

        # Atualização dos pesos
        learning_rate = 0.01
        weights -= learning_rate * gradient

        print("Pesos atualizados:", weights)
        mlflow.log_param("updated_weights", weights.tolist())
    return weights


if __name__ == "__main__":
    algorithms = [(sigmoid, sigmoid_derivative), (relu, relu_derivative), (tanh, tanh_derivative)]
    for algo in algorithms:
        run_backpropagation(algo)
