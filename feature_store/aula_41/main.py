# Importação das bibliotecas necessárias
import pandas as pd
from datetime import datetime, timedelta
from feast import FeatureStore, Entity, FeatureView, FileSource, Field
from feast.types import Int64, Float32

# Passo 1: Criação do DataFrame de origem
# Simulação de dados de transações de clientes
data = {
    "customer_id": [1001, 1002, 1003, 1004],
    "transaction_amount": [250.75, 89.60, 120.00, 310.40],
    "transaction_count": [5, 2, 3, 7],
    "event_timestamp": [
        datetime.now() - timedelta(hours=3),
        datetime.now() - timedelta(hours=2),
        datetime.now() - timedelta(hours=1),
        datetime.now(),
    ],
}

df = pd.DataFrame(data)

# Salva os dados em um arquivo Parquet para simular uma fonte de dados
df.to_parquet("data/customer_transactions.parquet")

# Passo 2: Definição da fonte de dados
customer_source = FileSource(
    path="data/customer_transactions.parquet",
    event_timestamp_column="event_timestamp",
)

# Passo 3: Definição da entidade
customer = Entity(name="customer_id", join_keys=["customer_id"])

# Passo 4: Definição da Feature View com transformação
from feast.transforms import OverWindow

customer_transactions_view = FeatureView(
    name="customer_transactions",
    entities=["customer_id"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="transaction_amount", dtype=Float32),
        Field(name="transaction_count", dtype=Int64),
    ],
    online=True,
    source=customer_source,
    tags={"team": "data_science"},
)

# Passo 5: Aplicação das definições na Feature Store
store = FeatureStore(repo_path=".")
store.apply([customer, customer_transactions_view])

# Passo 6: Materialização das features
store.materialize_incremental(end_date=datetime.now())
