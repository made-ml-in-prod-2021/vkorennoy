from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utils import (
    VOLUME, RAW_PATH, PREPROCESSED_PATH,
    MODEL_PATH, METRICS_PATH, SPLITTED_PATH,
    MODEL_NAME, BASIC_MODEL_NAME,
    METRICS_NAME, DEFAULT_ARGS,
)


with DAG(
        "2_data_pipeline",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(5),
        catchup=True,
) as dag:
    generate_data = DockerOperator(
        task_id="docker-airflow-generate-data",
        image="airflow-generate-data",
        command=f"--output-dir {RAW_PATH}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {RAW_PATH} --output-dir {PREPROCESSED_PATH}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--input-dir {PREPROCESSED_PATH} --output-dir {SPLITTED_PATH}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--input-dir {SPLITTED_PATH} --model-path {MODEL_PATH}"
                f" --model-name {MODEL_NAME}",  # --basic-model-name {BASIC_MODEL_NAME}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--data-path {SPLITTED_PATH} --model-path {MODEL_PATH} "
                f"--model-name {MODEL_NAME} --metrics-path {METRICS_PATH} "
                f"--metrics-name {METRICS_NAME}",

        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    generate_data >> preprocess >> split >> train >> validate
