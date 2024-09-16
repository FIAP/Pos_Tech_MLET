# Aula 3.4: Consulta e Uso de Features em Modelos de ML

# Importação das bibliotecas necessárias
import pandas as pd
from datetime import datetime
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier

# Passo 1: Preparação do DataFrame de Entidades para Treinamento
# DataFrame com as entidades e timestamps para treinamento
entity_df = pd.DataFrame({
    "customer_id": [1001, 1002, 1003],
    "event_timestamp": [datetime.now()] * 3,  # Usando o timestamp atual
})

# Passo 2: Consulta das Features para Treinamento
# Carregue o Feature Store
store = FeatureStore(repo_path=".")

# Defina as features que deseja obter
feature_list = [
    "customer_transactions:transaction_amount",
    "customer_transactions:transaction_count",
]

# Obtenha as features históricas
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=feature_list,
).to_df()

# Passo 3: Preparação dos Dados para Treinamento
# Suponha que temos as labels correspondentes
training_df["label"] = [1, 0, 1]  # 1 para churn, 0 para não churn

# Separe as features (X) e o target (y)
X_train = training_df[["transaction_amount", "transaction_count"]]
y_train = training_df["label"]

# Passo 4: Treinamento do Modelo de Machine Learning
# Inicialize o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treine o modelo
model.fit(X_train, y_train)

# Passo 5: Preparação das Entidades para Predição Online
# DataFrame com as entidades para predição online
entity_df_online = pd.DataFrame({
    "customer_id": [1001, 1002],
})

# Passo 6: Consulta das Features em Tempo Real
# Obtenha as features online
online_features = store.get_online_features(
    features=feature_list,
    entity_rows=entity_df_online.to_dict(orient="records"),
).to_df()

# Passo 7: Realização de Predições com o Modelo
# Prepare os dados para predição
X_online = online_features[["transaction_amount", "transaction_count"]]

# Realize as predições
predictions = model.predict(X_online)

# Exiba as predições
for customer_id, prediction in zip(entity_df_online["customer_id"], predictions):
    resultado = 'Churn' if prediction == 1 else 'No Churn'
    print(f"Customer ID {customer_id}: {resultado}")
