"""
This module is responsible for training a machine learning model using the provided dataset.

The module uses Linear Regression from scikit-learn for model training and leverages
MLflow for experiment tracking. The data is split into training and test sets, with the
model being trained on the training set. The test data and model outputs are saved for
further evaluation and deployment.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import mlflow
import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def get_features_from_store(feature_store_name):
    """
    Retrieve features from the feature store.

    Args:
        feature_store_name (str): Name of the feature store.

    Returns:
        tuple: Pickup and dropoff features.
    """
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    pickup_features = ml_client.feature_sets.get_feature_values(
        feature_set_name="pickup_features",
        feature_store_name=feature_store_name
    )
    dropoff_features = ml_client.feature_sets.get_feature_values(
        feature_set_name="dropoff_features",
        feature_store_name=feature_store_name
    )
    return pickup_features, dropoff_features


def main(training_data, test_data, model_output, model_metadata, feature_store_name):
    """
    Read training data, split data and initiate training.

    Parameters:
      training_data (str): training data folder
      test_data (str): test data folder
      model_output (str): a folder to store model files
      model_metadata (str): a file to store information about the model
      feature_store_name (str): name of the feature store
    """
    print("Hello training world...")

    lines = [
        f"Training data path: {training_data}",
        f"Test data path: {test_data}",
        f"Model output path: {model_output}",
        f"Model metadata path: {model_metadata}",
        f"Feature store name: {feature_store_name}",
    ]

    for line in lines:
        print(line)

    print("mounted_path files: ")
    arr = os.listdir(training_data)
    print(arr)

    df_list = []
    for filename in arr:
        print("reading file: %s ..." % filename)
        input_df = pd.read_parquet((Path(training_data) / filename))
        df_list.append(input_df)

    train_data = df_list[0]
    print(train_data.columns)

    # Get features from feature store
    pickup_features, dropoff_features = get_features_from_store(feature_store_name)

    # Merge features with training data
    train_data = train_data.merge(pickup_features, on="pickup_datetime")
    train_data = train_data.merge(dropoff_features, on="dropoff_datetime")

    train_x, test_x, trainy, testy = split(train_data)
    write_test_data(test_x, testy, test_data)
    train_model(train_x, trainy, model_output, model_metadata)


def split(train_data):
    """
    Split the input data into training and testing sets.

    Parameters:
    train_data (DataFrame): The input data.

    Returns:
    trainX (DataFrame): The training data.
    testX (DataFrame): The testing data.
    trainy (Series): The training labels.
    testy (Series): The testing labels.
    """
    # Split the data into input(X) and output(y)
    y = train_data["cost"]
    x = train_data[
        [
            "distance",
            "dropoff_latitude",
            "dropoff_longitude",
            "passengers",
            "pickup_latitude",
            "pickup_longitude",
            "store_forward",
            "vendor",
            "pickup_weekday",
            "pickup_month",
            "pickup_monthday",
            "pickup_hour",
            "pickup_minute",
            "pickup_second",
            "dropoff_weekday",
            "dropoff_month",
            "dropoff_monthday",
            "dropoff_hour",
            "dropoff_minute",
            "dropoff_second",
        ]
    ]

    # Split the data into train and test sets
    train_x, test_x, trainy, testy = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    print(train_x.shape)
    print(train_x.columns)

    return train_x, test_x, trainy, testy


def train_model(train_x, trainy, model_output, model_metadata):
    """
    Train a Linear Regression model and save the model and its metadata.

    Parameters:
    trainX (DataFrame): The training data.
    trainy (Series): The training labels.

    Returns:
    None
    """
    mlflow.autolog()
    # Train a Linear Regression Model with the train set
    with mlflow.start_run() as run:
        model = LinearRegression().fit(train_x, trainy)
        print(model.score(train_x, trainy))

        # Output the model, metadata and test data
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_data = {"run_id": run.info.run_id, "run_uri": model_uri}
        with open(model_metadata, "w") as json_file:
            json.dump(model_data, json_file, indent=4)

        pickle.dump(model, open((Path(model_output) / "model.sav"), "wb"))


def write_test_data(test_x, testy, test_data):
    """
    Write the testing data to a CSV file.

    Parameters:
    testX (DataFrame): The testing data.
    testy (Series): The testing labels.

    Returns:
    None
    """
    test_x["cost"] = testy
    print(test_x.shape)
    test_x.to_csv((Path(test_data) / "test_data.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--model_metadata", type=str, help="Path of model metadata")
    parser.add_argument("--feature_store_name", type=str, help="Name of the feature store")

    args = parser.parse_args()

    main(args.training_data, args.test_data, args.model_output, args.model_metadata, args.feature_store_name)
