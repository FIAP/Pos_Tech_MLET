## Roteiro de Vídeo 2.2: Técnicas de Otimização e Regularização

**Tempo estimado:** 20 minutos

---

**[Abertura] (0:00 - 0:30)**

- **Apresentação:** 

  - "Olá, pessoal! Bem-vindos a mais um vídeo da nossa série sobre Machine Learning. Hoje, vamos explorar dois conceitos fundamentais: técnicas de otimização e regularização em redes neurais."
  - "Entender esses conceitos é essencial para engenheiras e engenheiros de machine learning que desejam construir modelos eficientes e que generalizam bem."

**[Parte 1: Algoritmos de Otimização] (0:30 - 10:00)**

## 1. Stochastic Gradient Descent (SGD)

```python
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

    learning_rate = 0.01
    epochs = 100

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

        # Logando o desempenho
        mlflow.log_metric(f"error_epoch_{epoch}", error)
    
    print("Pesos finais com SGD:", weights)
    mlflow.log_param("final_weights_sgd", weights.tolist())
```

**Explicação:**

- "O Stochastic Gradient Descent (SGD) é um dos algoritmos de otimização mais simples e amplamente utilizados. Ele ajusta os pesos da rede em pequenas etapas, tornando o processo de treinamento mais rápido e eficiente."
- "Neste código, implementamos o SGD com uma função de ativação ReLU e registramos o erro em cada época com o MLflow."

## 2. ADAM (Adaptive Moment Estimation)

```python
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

    learning_rate = 0.01
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
```

**Explicação:**

- "ADAM combina as vantagens do SGD com o momentum, ajustando dinamicamente a taxa de aprendizado com base em momentos de primeira e segunda ordem. Isso resulta em uma convergência mais rápida e estável."
- "No código, implementamos o algoritmo ADAM, registrando os pesos finais e o erro de cada época usando o MLflow."

**[Parte 2: Técnicas de Regularização] (10:00 - 18:00)**

## 3. Regularização L1

```python
import numpy as np
import mlflow

# Configurando o MLflow
mlflow.set_experiment("Técnicas de Regularização - L1")

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
```

**Explicação:**

- "A regularização L1 adiciona uma penalidade proporcional à soma dos valores absolutos dos pesos, incentivando a esparsidade no modelo, ou seja, a maioria dos pesos se torna zero."
- "Aqui, implementamos a regularização L1 em um modelo simples e registramos a perda total em cada época."

## 4. Regularização L2

```python
import numpy as np
import mlflow

# Configurando o MLflow
mlflow.set_experiment("Técnicas de Regularização - L2")

with mlflow.start_run():
    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def l2_regularization(weights, lambd):
        return lambd * np.sum(weights ** 2)

    x = np.array([[0.1, 0.2], [0.4, 0.5]])
    y_true = np.array([[0.3], [0.9]])

    weights = np.array([[0.1], [0.2]])
    learning_rate = 0.01
    lambd = 0.01  # Coeficiente de regularização L2
    epochs = 100

    for epoch in range(epochs):
        z = np.dot(x, weights)
        y_pred = relu(z)
        error = mse_loss(y_true, y_pred)
        regularization_loss = l2_regularization(weights, lambd)
        total_loss = error + regularization_loss

        gradient = np.dot(x.T, relu_derivative(z) * (y_pred - y_true))
        weights -= learning_rate * gradient

        mlflow.log_metric(f"total_loss_epoch_{epoch}", total_loss)

    print("Pesos finais com L2:", weights)
    mlflow.log_param("final_weights_l2", weights.tolist())
```

**Explicação:**

- "A regularização L2 adiciona uma penalidade proporcional à soma dos quadrados dos pesos, prevenindo overfitting ao evitar que os pesos se tornem muito grandes."
- "Implementamos a regularização L2 e registramos a perda total em cada época com o MLflow."
