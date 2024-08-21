import numpy as np
import mlflow

# Configurando o MLflow
mlflow.set_experiment("Técnicas de Otimização - SGD")

with mlflow.start_run():
    # Função de ativação ReLU
    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    # Função de perda (MSE)
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Dados de entrada
    x = np.array([[0.1, 0.2], [0.4, 0.5]])
    y_true = np.array([[0.3], [0.9]])

    # Pesos iniciais
    weights = np.array([[0.1], [0.2]])

    learning_rate = 0.2
    epochs = 100

    weight_history = []

    for epoch in range(epochs):
        # Forward pass
        z = np.dot(x, weights)
        y_pred = relu(z)

        # Cálculo do erro
        error = mse_loss(y_true, y_pred)

        # Backward pass
        gradient = np.dot(x.T, relu_derivative(z) * (y_pred - y_true))

        # Atualização dos pesos - SGD
        weights -= learning_rate * gradient
        weight_history.append(weights)
        print(weights)

        # Logando o desempenho
        mlflow.log_metric(f"Current error on Epoch {epoch}", error)
        mlflow.log_metric(f"error_epoch_{epoch}", error)
    
    print(weight_history)
    print("Pesos finais com SGD:", weights)
    mlflow.log_param("final_weights_sgd", weights.tolist())
