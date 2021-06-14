from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utils import VOLUME, RAW_PATH, DEFAULT_ARGS


with DAG(
    "1_generate_data",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(3),
) as dag:
    generate_data = DockerOperator(
        task_id="docker-airflow-generate-data",
        image="airflow-generate-data",
        command=f"--output-dir {RAW_PATH}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME],
    )
