# Importação das bibliotecas necessárias
import pandas as pd
from datetime import datetime
from feast import FeatureStore

# Passo 1: Preparação das entidades para predição
entity_df_online = pd.DataFrame({
    "customer_id": [1001, 1002, 1005],  # Incluindo um ID inexistente para monitoramento
})

# Passo 2: Carregamento da Feature Store
store = FeatureStore(repo_path=".")

# Definição das features a serem obtidas
feature_list = [
    "customer_transactions:transaction_amount",
    "customer_transactions:transaction_count",
]

# Passo 3: Obtenção das features online
online_features = store.get_online_features(
    features=feature_list,
    entity_rows=entity_df_online.to_dict(orient="records"),
).to_df()

# Passo 4: Verificação e monitoramento das features obtidas
# Identificação de valores ausentes ou inconsistentes
missing_values = online_features.isnull().sum()
print("Monitoramento de Features:")
print(missing_values)

# Passo 5: Servindo as features para um modelo de ML
# (Simulação de carregamento do modelo e realização de predições)
from sklearn.ensemble import RandomForestClassifier

# Dados de treinamento simulados (normalmente você treinaria o modelo separadamente)
X_train = pd.DataFrame({
    "transaction_amount": [250.75, 89.60, 120.00, 310.40],
    "transaction_count": [5, 2, 3, 7],
})
y_train = [1, 0, 1, 0]

# Treinamento do modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Preparação dos dados para predição
X_online = online_features[["transaction_amount", "transaction_count"]].fillna(0)

# Realização das predições
predictions = model.predict(X_online)

# Exibição dos resultados
for customer_id, prediction in zip(entity_df_online["customer_id"], predictions):
    resultado = 'Churn' if prediction == 1 else 'No Churn'
    print(f"Customer ID {customer_id}: {resultado}")
