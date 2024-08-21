import numpy as np
import mlflow


# Configurando o MLflow
mlflow.set_experiment("Funções de Ativação - ReLU")

def relu(vector):
    return np.maximum(0, vector)

def relu_derivative(vector):
    return np.where(vector > 0, 1, 0)


with mlflow.start_run():
    # Exemplo de uso
    x = np.array(
        [
            [-1, 0, 1], [-2, 5, 2], [100, 0, -100],
            [-1, 0, 1], [-2, 5, 2], [100, 0, -100],
            [-1, 0, 1], [-2, 5, 2], [100, 0, -100]
        ]
    )
    relu_output = relu(x)
    relu_grad = relu_derivative(x)

    # Logando o desempenho e resultado
    mlflow.log_param("input", x.tolist())
    mlflow.log_metric("relu_output", relu_output.mean())
    mlflow.log_metric("relu_derivative", relu_grad.mean())

    print("Saída da ReLU:", relu_output)
    print("Derivada da ReLU:", relu_grad)
