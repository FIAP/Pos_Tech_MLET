import pandas as pd
from embedders import embedding

from feast import FeatureStore
from datetime import datetime


new_data = {
    "customer_id": [1001, 1002, 1003],
    "customer_age": [18, 36, 53],
    "customer_profile": [
        "Curte pagode e churrasco aos sábados",
        "Curte techno e balada às quintas-feiras",
        "Curte sertanejo e moda de viola aos domingos"
    ],
    "customer_profile_image": [
        b"https://someweb.com/image-customer-1001.png",
        b"https://someweb.com/image-customer-1002.png",
        b"https://someweb.com/image-customer-1003.png"
    ]
}

# Carregue o Feature Store
store = FeatureStore(repo_path=".")

# Atualize as features re-materializando os dados
previous_data = store.materialize_incremental(end_date=datetime.now())
df = pd.from_parquet(previous_data)
new_df = pd.DataFrame(new_data)

df.concat(new_df, by="customer_id")
df.to_parquet("data/customer_transactions.parquet")

#executa rotina em customer_features.py
previous_data = store.materialize_incremental(end_date=datetime.now())
df = pd.from_parquet(previous_data)


df['embedding_customer_profile'] = embedding(df['customer_profile'])
df.to_parquet("data/customer_transactions.parquet")
#executa rotina em customer_features.py