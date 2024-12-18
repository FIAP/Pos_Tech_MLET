import pandas as pd
from datetime import datetime
import pyarrow.parquet as pq
from scipy.stats import kstest, zscore


# Dados de exemplo
data = {
    "customer_id": [1001, 1002, 1003],
    "transaction_amount": [250.75, 89.60, 120.00],
    "transaction_amount_with_tax": [352.75, 159.61, 180.32],
    "transaction_count": [5, 2, 3],
    "event_timestamp": [datetime.now()] * 3,
}

ks_score = kstest(data.get("transaction_amount", []), data.get("transaction_amount_with_tax", []))
data["ks_statistic"] = [ks_score.statistic for i in range(len(data.get("transaction_amount", [])))]
data["zscore_trans_amount"] = zscore(data.get("transaction_amount", []))
data["zscore_trans_amount_tax"] = zscore(data.get("transaction_amount_with_tax", []))

df = pd.DataFrame(data)

# Salve em formato Parquet
df.to_parquet("data/customer_transactions.parquet")
