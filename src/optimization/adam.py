import numpy as np
import mlflow

# Configurando o MLflow
mlflow.set_experiment("Técnicas de Otimização - ADAM")

with mlflow.start_run():
    # Funções de ativação e derivadas
    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    x = np.array([[0.1, 0.2], [0.4, 0.5]])
    y_true = np.array([[0.3], [0.9]])

    weights = np.array([[0.1], [0.2]])

    learning_rate = 0.2
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    t = 0
    epochs = 100

    for epoch in range(epochs):
        t += 1
        z = np.dot(x, weights)
        y_pred = relu(z)
        error = mse_loss(y_true, y_pred)
        gradient = np.dot(x.T, relu_derivative(z) * (y_pred - y_true))

        # Atualização do m e v
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # Correção de viés
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Atualização dos pesos - ADAM
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        mlflow.log_metric(f"error_epoch_{epoch}", error)

    print("Pesos finais com ADAM:", weights)
    mlflow.log_param("final_weights_adam", weights.tolist())
