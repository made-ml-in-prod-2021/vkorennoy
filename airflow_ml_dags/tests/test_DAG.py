import sys
import pytest
from airflow.models import DagBag


sys.path.append('dags')


def assert_dag_dict_equal(source, dag):
    assert dag.task_dict.keys() == source.keys()
    for task_id, downstream_list in source.items():
        assert dag.has_task(task_id), (
            "Missing task_id: {} in dag".format(task_id)
        )

        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list), (
            "unexpected downstream link in {}".format(task_id)
        )


@pytest.fixture
def dag_bag():
    return DagBag(dag_folder='dags/', include_examples=False)


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
    assert_dag_dict_equal(steps, dag)


def test_predictions_dag_loaded(dag_bag):
    assert "3_make_predictions" in dag_bag.dags
    assert len(dag_bag.dags["3_make_predictions"].tasks) == 1
