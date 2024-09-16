import pandas as pd
from datetime import datetime
import pyarrow.parquet as pq

# Dados de exemplo
data = {
    "customer_id": [1001, 1002, 1003],
    "transaction_amount": [250.75, 89.60, 120.00],
    "transaction_count": [5, 2, 3],
    "event_timestamp": [datetime.now()] * 3,
}

df = pd.DataFrame(data)

# Salve em formato Parquet
df.to_parquet("data/customer_transactions.parquet")
