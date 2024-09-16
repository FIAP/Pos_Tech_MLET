# Importação das bibliotecas necessárias
import pandas as pd
from datetime import datetime, timedelta
from feast import FeatureStore, Entity, FeatureView, FileSource, Field
from feast.types import Int64, Float32, String

# Passo 1: Criação de múltiplos DataFrames de origem
# Dados de transações de clientes
transactions_data = {
    "customer_id": [1001, 1002, 1003, 1004],
    "transaction_amount": [250.75, 89.60, 120.00, 310.40],
    "transaction_count": [5, 2, 3, 7],
    "event_timestamp": [datetime.now() - timedelta(hours=i) for i in range(4)],
}
transactions_df = pd.DataFrame(transactions_data)
transactions_df.to_parquet("data/customer_transactions.parquet")

# Dados de perfil de clientes
profile_data = {
    "customer_id": [1001, 1002, 1003, 1004],
    "customer_name": ["Alice", "Bob", "Charlie", "Diana"],
    "loyalty_score": [8.5, 7.2, 9.1, 6.8],
    "event_timestamp": [datetime.now() - timedelta(hours=i) for i in range(4)],
}
profile_df = pd.DataFrame(profile_data)
profile_df.to_parquet("data/customer_profiles.parquet")

# Passo 2: Definição das fontes de dados
transactions_source = FileSource(
    path="data/customer_transactions.parquet",
    event_timestamp_column="event_timestamp",
)

profiles_source = FileSource(
    path="data/customer_profiles.parquet",
    event_timestamp_column="event_timestamp",
)

# Passo 3: Definição das entidades
customer = Entity(name="customer_id", join_keys=["customer_id"])

# Passo 4: Definição das Feature Views
# Feature View para transações
customer_transactions_view = FeatureView(
    name="customer_transactions",
    entities=["customer_id"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="transaction_amount", dtype=Float32),
        Field(name="transaction_count", dtype=Int64),
    ],
    online=True,
    source=transactions_source,
    tags={"team": "data_engineering"},
)

# Feature View para perfis
customer_profiles_view = FeatureView(
    name="customer_profiles",
    entities=["customer_id"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="customer_name", dtype=String),
        Field(name="loyalty_score", dtype=Float32),
    ],
    online=True,
    source=profiles_source,
    tags={"team": "marketing"},
)

# Passo 5: Aplicação das definições na Feature Store
store = FeatureStore(repo_path=".")
store.apply([customer, customer_transactions_view, customer_profiles_view])

# Passo 6: Materialização das features
store.materialize_incremental(end_date=datetime.now())
