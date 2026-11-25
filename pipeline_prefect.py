import subprocess  # nosec B404 - Required to execute trusted CLI tools
import sys

from prefect import task, flow
from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)
import argparse

PYTHON_FILES = ["main.py", "model_pipeline.py", "pipeline_prefect.py"]


def run_command(command):
    """Execute trusted, predefined CLI commands."""
    subprocess.run(command, check=True)  # nosec B603 - No shell, trusted commands


@task(name="Install Dependencies")
def install_dependencies_task():
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Format Code")
def format_code_task():
    run_command(["black", *PYTHON_FILES])


@task(name="Quality Code Check")
def quality_code_check_task():
    run_command(["flake8", *PYTHON_FILES])


@task(name="Security Code Check")
def security_code_check_task():
    run_command(["bandit", *PYTHON_FILES])


@task(name="Run Unit Tests")
def unit_tests_task():
    run_command([sys.executable, "-m", "pytest", "test_pipeline.py", "-v"])


@task(name="prepare_data_task")
def prepare_data_task(date_filename):
    return prepare_data(date_filename)


@task(name="train_task")
def train_task(x_train, y_train):
    return train_model(x_train, y_train)


@task(name="save_model_task")
def save_model_task(model, filename):
    save_model(model, filename)


@task(name="load_model_task")
def load_model_task(filename):
    return load_model(filename)


@task(name="evaluate_model_task")
def evaluate_model_task(model, x_test, y_test):
    return evaluate_model(model, x_test, y_test)


@flow(name="Code_Check")
def code_check_flow():
    install_dependencies_task()
    format_code_task()
    quality_code_check_task()
    security_code_check_task()
    unit_tests_task()


@flow(name="all")
def all_flow(data_filename, model_filename):

    code_check_flow()

    x_train, x_test, y_train, y_test = prepare_data_task(data_filename)
    trained_model = train_task(x_train, y_train)
    save_model_task(trained_model, model_filename)
    evaluate_model_task(trained_model, x_test, y_test)


@flow(name="train")
def train_flow(data_filename, model_filename):
    x_train, x_test, y_train, y_test = prepare_data_task(data_filename)
    trained_model = train_task(x_train, y_train)
    save_model_task(trained_model, model_filename)


@flow(name="evaluate")
def evaluate_flow(data_filename, model_filename):
    x_train, x_test, y_train, y_test = prepare_data_task(data_filename)
    loaded_model = load_model_task(model_filename)
    evaluate_model_task(loaded_model, x_test, y_test)


def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn Machine Learning Pipeline with Prefect"
    )

    parser.add_argument(
        "--flow",
        choices=["all", "train", "evaluate", "code_check"],
        default="all",
        help=(
            "Select a workflow: "
            "'all' for the complete pipeline, "
            "'train' for training, "
            "'evaluate' for evaluation, "
            "or 'code_check' for code checks."
        ),
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow("Churn_Modelling.csv", "classifier.joblib")
    elif args.flow == "train":
        train_flow("Churn_Modelling.csv", "classifier.joblib")
    elif args.flow == "evaluate":
        evaluate_flow("Churn_Modelling.csv", "classifier.joblib")
    elif args.flow == "code_check":
        code_check_flow()


if __name__ == "__main__":
    main()
