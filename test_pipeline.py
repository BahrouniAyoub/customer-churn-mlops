

from model_pipeline import prepare_data


def test_prepare_data():
    x_train, x_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    
     # Verify that data was split correctly
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    # Verify that the target column was removed
    assert "Exited" not in x_train.columns

    # Verify that the training data isn't empty
    assert len(x_train) > 0