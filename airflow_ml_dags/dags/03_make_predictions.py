from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utils import (
    VOLUME, DEFAULT_ARGS, SPLITTED_PATH,
    MODEL_PATH, MODEL_NAME, METRICS_PATH,
    METRICS_NAME,
)


with DAG(
    "3_make_predictions",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(0),
) as dag:
    predict = DockerOperator(
        image="airflow-predict",
        command=f"--data-path {SPLITTED_PATH} --model-path {MODEL_PATH} "
                f"--model-name {MODEL_NAME} --metrics-path {METRICS_PATH} "
                f"--metrics-name {METRICS_NAME}",

        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[VOLUME],
    )
