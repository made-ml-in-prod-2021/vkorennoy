import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators import email
from airflow.utils.dates import days_ago

from utils import (
    VOLUME, RAW_PATH, PREPROCESSED_PATH,
    MODEL_PATH, METRICS_PATH, SPLITTED_PATH,
    MODEL_NAME, METRICS_NAME, DEFAULT_ARGS,
)


with DAG(
        "2_data_pipeline",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(3),
        catchup=True,
) as dag:
    start_task = DummyOperator(task_id='start_pipeline')
    wait_train_data = FileSensor(
        task_id="wait_for_train_data",
        poke_interval=10,
        retries=100,
        filepath=os.path.join(RAW_PATH, "data.csv"),
    )

    wait_train_target = FileSensor(
        task_id="wait_for_train_target",
        poke_interval=10,
        retries=100,
        filepath=os.path.join(RAW_PATH, "target.csv"),
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
                f" --model-name {MODEL_NAME}",
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

    email_summary = email.EmailOperator(
        task_id='email_summary',
        to='vkorennoj@gmail.com',
        subject='Airflow pipeline',
        html_content="""Pipeline built successfully"""
    )

    start_task >> [wait_train_data, wait_train_target] >> preprocess >> split >> \
        train >> validate >> email_summary
