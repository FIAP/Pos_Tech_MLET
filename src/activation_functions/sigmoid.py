import numpy as np
import mlflow


# Configurando o MLflow
mlflow.set_experiment("Funções de Ativação - Sigmóide")

def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))

def sigmoid_derivative(vector):
    sig = sigmoid(vector)
    return sig * (1 - sig)

with mlflow.start_run():

    # Exemplo de uso
    x = np.array(
        [
            [-1, 0, 1], [-2, 5, 2], [100, 0, -150],
            [-1, 0, 1], [-2, 5, 2], [500, 0, -100],
            [-1, 0, 1], [-2, 5, 2], [125, 0, -214]
        ]
    )
    sigmoid_output = sigmoid(x)
    sigmoid_grad = sigmoid_derivative(x)

    # Logando o desempenho e resultado
    mlflow.log_param("input", x.tolist())
    mlflow.log_metric("sigmoid_output", sigmoid_output.mean())
    mlflow.log_metric("sigmoid_derivative", sigmoid_grad.mean())

    print("Saída da Sigmóide:", sigmoid_output)
    print("Derivada da Sigmóide:", sigmoid_grad)

