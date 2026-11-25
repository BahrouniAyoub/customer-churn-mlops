import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn Machine Learning Pipeline"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="Churn_Modelling.csv",
        help="Path to CSV file containing Churn_Modelling.csv",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="classifier.joblib",
        help="Path to save the trained model (default: classifier.joblib)",
    )

    parser.add_argument(
        "--mode",
        choices=["train", "load"],
        default="train",
        help=(
            "Choose 'train' to train a new model " "or 'load' to load an existing model"
        ),
    )

    args = parser.parse_args()

    x_train, x_test, y_train, y_test = prepare_data(args.data)

    if args.mode == "train":
        model = train_model(x_train, y_train)
        save_model(model, args.model)
    else:
        model = load_model(args.model)

    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
