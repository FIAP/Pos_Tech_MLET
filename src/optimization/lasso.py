import numpy as np
import mlflow

# Configurando o MLflow
mlflow.set_experiment("Técnicas de Regularização LASSO")

with mlflow.start_run():
    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def l1_regularization(weights, lambd):
        return lambd * np.sum(np.abs(weights))

    x = np.array([[0.1, 0.2], [0.4, 0.5]])
    y_true = np.array([[0.3], [0.9]])

    weights = np.array([[0.1], [0.2]])
    learning_rate = 0.01
    lambd = 0.01  # Coeficiente de regularização L1
    epochs = 100

    for epoch in range(epochs):
        z = np.dot(x, weights)
        y_pred = relu(z)
        error = mse_loss(y_true, y_pred)
        regularization_loss = l1_regularization(weights, lambd)
        total_loss = error + regularization_loss

        gradient = np.dot(x.T, relu_derivative(z) * (y_pred - y_true))
        weights -= learning_rate * gradient

        mlflow.log_metric(f"total_loss_epoch_{epoch}", total_loss)

    print("Pesos finais com L1:", weights)
    mlflow.log_param("final_weights_l1", weights.tolist())
