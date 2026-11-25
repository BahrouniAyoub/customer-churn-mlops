from datetime import timedelta

from prefect import serve

from pipeline_prefect import (
    all_flow,
    train_flow,
    evaluate_flow,
    code_check_flow,
)

if __name__ == "__main__":
    all_deployment = all_flow.to_deployment(
        name="all_deployment",
        interval = timedelta(days=1),
        parameters={
            "data_filename": "Churn_Modelling.csv",
            "model_filename": "classifier.joblib",
        },
    )
    train_deployment = train_flow.to_deployment(
        name="train_deployment",
        parameters={
            "data_filename": "Churn_Modelling.csv",
            "model_filename": "classifier.joblib",
        },
    )
    evaluate_deployment = evaluate_flow.to_deployment(
        name="evaluate_deployment",
        parameters={
            "data_filename": "Churn_Modelling.csv",
            "model_filename": "classifier.joblib",
        }
    )
    
    code_deployment = code_check_flow.to_deployment(
        name="code_deployment",
    )
    
    serve(
        all_deployment,
        train_deployment,
        evaluate_deployment,
        code_deployment
    )