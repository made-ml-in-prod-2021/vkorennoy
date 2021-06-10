from datetime import timedelta


VOLUME = "C:\\Users\\And_then_i_woke_up\\Desktop\\MADE\\!! Учеба\\2 семестр\\ML prod\\vkorennoy\\HW-3\\data:/data"
RAW_PATH = "/data/raw/{{ ds }}"
PREPROCESSED_PATH = "/data/preprocessed/{{ ds }}"
SPLITTED_PATH = "/data/splitted/{{ ds }}"
MODEL_PATH = "/data/models"
MODEL_NAME = "model_{{ ds }}"
BASIC_MODEL_NAME = "basic_model"
METRICS_PATH = "/data/metrics"
METRICS_NAME = "metrics_{{ ds }}"


DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["vkorennoj@example.com"],
    'email_on_failure': True,
    'email_on_retry': False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
