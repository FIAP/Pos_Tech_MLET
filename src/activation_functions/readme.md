# Roteiro de Vídeo 2.1: Funções de Ativação e Backpropagation

**Tempo estimado:** 20 minutos

---

**[Abertura] (0:00 - 0:30)**

- **Apresentação:**
  - "Olá, pessoal! Bem-vindos a mais um vídeo da nossa série sobre Machine Learning. Hoje vamos explorar as funções de ativação e o algoritmo de backpropagation, dois pilares fundamentais para o sucesso das redes neurais."

**[Parte 1: Funções de Ativação] (0:30 - 10:00)**

## 1. Função Sigmóide

```python
import numpy as np
import mlflow

# Configurando o MLflow
mlflow.set_experiment("Funções de Ativação - Sigmóide")

with mlflow.start_run():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        sig = sigmoid(x)
        return sig * (1 - sig)

    # Exemplo de uso
    x = np.array([[-1, 0, 1]])
    sigmoid_output = sigmoid(x)
    sigmoid_grad = sigmoid_derivative(x)

    # Logando o desempenho e resultado
    mlflow.log_param("input", x.tolist())
    mlflow.log_metric("sigmoid_output", sigmoid_output.mean())
    mlflow.log_metric("sigmoid_derivative", sigmoid_grad.mean())

    print("Saída da Sigmóide:", sigmoid_output)
    print("Derivada da Sigmóide:", sigmoid_grad)
```

**Explicação:**

- "A função sigmóide é usada para mapear entradas para valores entre 0 e 1. Isso é útil em problemas de classificação binária."
- "Aqui, implementamos a função e sua derivada. Usamos o MLflow para registrar parâmetros de entrada, saídas e derivadas médias, o que ajuda a monitorar o desempenho durante o treinamento."

## 2. Função Tanh

```python
import mlflow

# Configurando o MLflow
mlflow.set_experiment("Funções de Ativação - Tanh")

with mlflow.start_run():
    def tanh(x):
        return np.tanh(x)

    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

    # Exemplo de uso
    x = np.array([[-1, 0, 1]])
    tanh_output = tanh(x)
    tanh_grad = tanh_derivative(x)

    # Logando o desempenho e resultado
    mlflow.log_param("input", x.tolist())
    mlflow.log_metric("tanh_output", tanh_output.mean())
    mlflow.log_metric("tanh_derivative", tanh_grad.mean())

    print("Saída da Tanh:", tanh_output)
    print("Derivada da Tanh:", tanh_grad)
```

**Explicação:**

- "A função `tanh` mapeia os valores de entrada para o intervalo entre -1 e 1, centralizando os dados e facilitando o aprendizado em redes neurais."
- "Registramos os dados com MLflow, o que permite comparar a performance de diferentes funções de ativação ao longo do treinamento."

## 3. Função ReLU

```python
import mlflow

# Configurando o MLflow
mlflow.set_experiment("Funções de Ativação - ReLU")

with mlflow.start_run():
    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    # Exemplo de uso
    x = np.array([[-1, 0, 1]])
    relu_output = relu(x)
    relu_grad = relu_derivative(x)

    # Logando o desempenho e resultado
    mlflow.log_param("input", x.tolist())
    mlflow.log_metric("relu_output", relu_output.mean())
    mlflow.log_metric("relu_derivative", relu_grad.mean())

    print("Saída da ReLU:", relu_output)
    print("Derivada da ReLU:", relu_grad)
```

**Explicação:**

- "A ReLU é uma função de ativação simples e eficaz, ideal para redes profundas. Ela resolve o problema do gradiente desaparecendo, mas pode causar neurônios mortos."
- "Com o MLflow, podemos acompanhar como a ReLU está funcionando em comparação com outras funções de ativação."

**[Parte 2: Backpropagation] (10:00 - 19:00)**

## Exemplo Completo de Backpropagation

```python
import numpy as np
import mlflow

# Configurando o MLflow
mlflow.set_experiment("Backpropagation")

with mlflow.start_run():
    # Funções de ativação e suas derivadas
    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    # Função de perda (MSE)
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Exemplo de backpropagation
    x = np.array([[0.1, 0.2], [0.4, 0.5]])
    y_true = np.array([[0.3], [0.9]])

    weights = np.array([[0.1], [0.2]])

    # Forward pass
    z = np.dot(x, weights)
    y_pred = relu(z)

    error = mse_loss(y_true, y_pred)
    print("Erro inicial:", error)

    # Backward pass
    gradient = np.dot(x.T, relu_derivative(z) * (y_pred - y_true))

    # Logando o gradiente
    mlflow.log_metric("initial_error", error)
    mlflow.log_param("initial_weights", weights.tolist())
    mlflow.log_param("gradient", gradient.tolist())

    # Atualização dos pesos
    learning_rate = 0.01
    weights -= learning_rate * gradient

    print("Pesos atualizados:", weights)
    mlflow.log_param("updated_weights", weights.tolist())
```

**Explicação:**

- "Neste código, vemos o backpropagation em ação. A ReLU é utilizada como função de ativação, e os pesos são ajustados com base no gradiente calculado."
- "Usamos o MLflow para registrar o erro inicial, os gradientes e os pesos antes e depois da atualização, facilitando o rastreamento do processo de otimização."

**[Conclusão] (19:00 - 20:00)**

- **Resumo e Importância:**
  - "Hoje, vimos a importância das funções de ativação e como o algoritmo de backpropagation ajusta os pesos das redes neurais, permitindo que elas aprendam."
  - "Esses conceitos são essenciais para qualquer engenheiro de machine learning, e entender seu funcionamento detalhado é crucial para construir modelos robustos."

- **Encerramento:**
  - "Espero que esse conteúdo tenha sido útil. No próximo vídeo, continuaremos a aprofundar nossos conhecimentos em redes neurais. Até a próxima!"

---

Esse roteiro fornece uma abordagem prática e completa para os conceitos de funções de ativação e backpropagation, com a implementação de código comentado e o uso do MLflow para monitorar e registrar os experimentos.
