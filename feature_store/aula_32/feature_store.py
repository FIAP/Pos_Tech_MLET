from feast import FeatureStore

# Carregue o Feature Store
store = FeatureStore(repo_path=".")

# Aplique as definições das features
store.apply()

# Ingestão dos dados offline
store.materialize_incremental(end_date=datetime.now())
