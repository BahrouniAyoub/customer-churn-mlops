from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib


def prepare_data(date_filename):
    df = pd.read_csv(date_filename)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    columns_to_drop = ["Surname", "Geography"]

    df = df.drop(columns_to_drop, axis=1)
    x = df.drop(["Exited"], axis=1)
    y = df["Exited"]
    x.drop(
        columns=[
            "RowNumber",
            "CustomerId",
        ],
        inplace=True,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(x_train, y_train)

    return rf


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = (y_pred == y_test).mean()

    print("Accuracy score:" + str(accuracy * 100))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", matrix)

    return accuracy


def save_model(model, filename):
    joblib.dump(model, filename)

    print("Model saved successfully!")


def load_model(filename):
    model = joblib.load(filename)
    print("Model loaded successfully!")

    return model
