from datetime import timedelta


VOLUME = "/home/vkorennoy/airflow_ml_dags/data:/data"
RAW_PATH = "data/raw/{{ ds }}"
PREPROCESSED_PATH = "data/preprocessed/{{ ds }}"
SPLITTED_PATH = "data/splitted/{{ ds }}"
MODEL_PATH = "data/models"
MODEL_NAME = "model_{{ ds }}"
METRICS_PATH = "data/metrics"
METRICS_NAME = "metrics_{{ ds }}"


DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["vkorennoj@gmail.com"],
    'email_on_failure': True,
    'email_on_retry': False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
