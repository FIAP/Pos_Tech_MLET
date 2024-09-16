from feast import FeatureStore
from datetime import datetime

# Carregue o Feature Store
store = FeatureStore(repo_path=".")

# Atualize as features re-materializando os dados
store.materialize_incremental(end_date=datetime.now())
