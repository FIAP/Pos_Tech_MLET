import numpy as np
import mlflow

# Configurando o MLflow
mlflow.set_experiment("Funções de Ativação - Tanh")


def tanh(vector):
    return np.tanh(vector)

def tanh_derivative(vector):
    return 1 - np.tanh(vector)**2


with mlflow.start_run():

    # Exemplo de uso
    x = np.array(
        [
            [0, 255, 255], [0, 150, 255], [100, 0, 150],
            [0, 255, 255], [0, 150, 255], [100, 0, 150],
            [0, 255, 255], [0, 150, 255], [100, 0, 150]
        ]
    )
    tanh_output = tanh(x)
    tanh_grad = tanh_derivative(x)

    # Logando o desempenho e resultado
    mlflow.log_param("input", x.tolist())
    mlflow.log_metric("tanh_output", tanh_output.mean())
    mlflow.log_metric("tanh_derivative", tanh_grad.mean())

    print("Saída da Tanh:", tanh_output)
    print("Derivada da Tanh:", tanh_grad)
