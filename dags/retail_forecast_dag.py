#Weekly forecast pipeline: prepare data, check it, evaluate the model, and publish only if the gate passes
import os
from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

#Where the project and its Python environment live (can be changed with environment variables)
PROJECT_DIR = os.environ.get("RETAIL_FORECAST_DIR", "/mnt/c/dev/retail-forecast")
PYTHON = os.path.expanduser(os.environ.get("RETAIL_FORECAST_PYTHON", "~/venvs/retail/bin/python"))


def step(name: str, retries: int = 1) -> BashOperator:
    """One pipeline step = one Airflow task that runs src/pipeline.py <name>."""
    return BashOperator(
        task_id=name,
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/pipeline.py {name}",
        env={"MODEL_OBJECTIVE": "{{ params.model_objective }}"},
        append_env=True,
        retries=retries,
        retry_delay=timedelta(minutes=5),
    )


with DAG(
    dag_id="retail_forecast_weekly",
    description="Prepare data, check it, backtest the model and publish forecasts if the validation gate passes",
    schedule="0 6 * * 1",  #every Monday 06:00, before the weekly orders
    start_date=datetime(2026, 11, 1),
    catchup=False,
    params={"model_objective": "poisson"},
    tags=["forecasting"],
):
    prepare = step("prepare")
    check_data = step("check_data", retries=0)
    evaluate = step("evaluate")
    gate = step("gate", retries=0)  #a failed gate is a decision, not a glitch, so no retry
    publish = step("publish")

    prepare >> check_data >> evaluate >> gate >> publish
