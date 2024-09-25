from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64, String
from feast.infra.offline_stores.file_source import FileSource


# Defina a fonte de dados
customer_source = FileSource(
    path="data/customer_transactions.parquet",
    event_timestamp_column="event_timestamp",
)

# Defina a entidade
Entity(name="customer_id", value_type=2, join_keys=["customer_id"])

# Defina a Feature View
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
)
