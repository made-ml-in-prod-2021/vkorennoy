import sys
import pytest
from airflow.models import DagBag


sys.path.append("dags")


@pytest.fixture()
def dag_bag():
    return DagBag()


def test_dag_imported_ok(dag_bag):
    assert dag_bag.dags is not None
    assert dag_bag.import_errors == {}


def test_generate_data_launched_ok(dag_bag):
    assert "1_generate_data" in dag_bag.dags
    assert len(dag_bag.dags["1_generate_data"].tasks) == 1


def test_pipeline_launched_ok(dag_bag):
    assert "2_data_pipeline" in dag_bag.dags
    assert len(dag_bag.dags["2_data_pipeline"].tasks) == 5


def test_pipeline_steps_ok(dag_bag):
    steps = {
        "docker-airflow-generate-data": ["docker-airflow-preprocess"],
        "docker-airflow-preprocess": ["docker-airflow-split"],
        "docker-airflow-split": ["docker-airflow-train"],
        "docker-airflow-train": ["docker-airflow-validate"],
        "docker-airflow-validate": [],
    }
    dag = dag_bag.dags["2_data_pipeline"]
    dag.assertDagDictEqual(steps)


def test_predictions_dag_loaded(dag_bag):
    assert "3_make_predictions" in dag_bag.dags
    assert len(dag_bag.dags["3_make_predictions"].tasks) == 1
