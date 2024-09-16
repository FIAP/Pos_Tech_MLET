# DAG do Airflow para integração com a Feature Store
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from feast import FeatureStore

default_args = {
    "owner": "airflow",
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "feature_store_pipeline",
    default_args=default_args,
    schedule_interval="0 * * * *",  # Executa a cada hora
)

def materialize_features():
    store = FeatureStore(repo_path=".")
    store.materialize_incremental(end_date=datetime.now())

def train_model():
    # Código para treinamento do modelo utilizando features da Feature Store
    store = FeatureStore(repo_path=".")
    entity_df = ...  # DataFrame de entidades para treinamento
    feature_list = [...]
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_list,
    ).to_df()
    # Treinamento do modelo
    ...

materialize_task = PythonOperator(
    task_id="materialize_features",
    python_callable=materialize_features,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

materialize_task >> train_model_task
