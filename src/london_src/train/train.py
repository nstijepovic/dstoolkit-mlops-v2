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
# from azureml.featurestore import FeatureSet, FeatureStoreClient
# from azureml.featurestore.feature_source import CsvFeatureSource


def main(training_data, test_data, model_output, model_metadata, feature_store_name, subscription_id, resource_group_name):
    """
    Read training data, split data, enrich data with features from the feature store, and initiate training.

    Parameters:
      training_data (str): training data folder
      test_data (str): test data folder
      model_output (str): a folder to store model files
      model_metadata (str): a file to store information about the model
      feature_store_name (str): name of the feature store
      subscription_id (str): Azure subscription ID
      resource_group_name (str): Azure resource group name
    """
    print("Starting training...")

    # Load training data
    df_list = []
    for filename in os.listdir(training_data):
        print(f"Reading file: {filename}")
        input_df = pd.read_csv((Path(training_data) / filename))
        df_list.append(input_df)

    train_data = df_list[0]

    # Enrich data with features from the feature store
    enriched_train_data = get_enriched_data(train_data, feature_store_name, subscription_id, resource_group_name)

    # Split data into training and testing
    train_x, test_x, trainy, testy = split(enriched_train_data)

    # Write the test data to the provided folder
    write_test_data(test_x, testy, test_data)

    # Train the model
    train_model(train_x, trainy, model_output, model_metadata)


def get_enriched_data(train_data, feature_store_name, subscription_id, resource_group_name):
    """
    Enrich the training data by pulling features from the feature store.

    Parameters:
    train_data (pd.DataFrame): The original training data.
    feature_store_name (str): The name of the feature store.
    subscription_id (str): Azure subscription ID.
    resource_group_name (str): Azure resource group name.

    Returns:
    pd.DataFrame: The enriched training data.
    """
    # Initialize Feature Store client
    fs_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        feature_store_name=feature_store_name
    )

    # Example logic to enrich data - Retrieve feature set and apply transformations
    feature_set = fs_client.feature_sets.get(name=feature_store_name, version="latest")

    # Assuming 'vendorID' is the key column to join on
    enriched_train_data = feature_set.join_on(train_data, join_column="vendorID")

    return enriched_train_data


def split(train_data):
    """
    Split the input data into training and testing sets.

    Parameters:
    train_data (pd.DataFrame): The input data.

    Returns:
    trainX (pd.DataFrame): The training data.
    testX (pd.DataFrame): The testing data.
    trainy (pd.Series): The training labels.
    testy (pd.Series): The testing labels.
    """
    y = train_data["cost"]
    x = train_data.drop(["cost", "timestamp", "accountID"], axis=1)

    train_x, test_x, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=42)

    return train_x, test_x, trainy, testy


def train_model(train_x, trainy, model_output, model_metadata):
    """
    Train a Linear Regression model and save the model and its metadata.

    Parameters:
    trainX (pd.DataFrame): The training data.
    trainy (pd.Series): The training labels.
    model_output (str): Path to save the model.
    model_metadata (str): Path to save the model metadata.

    Returns:
    None
    """
    mlflow.autolog()
    with mlflow.start_run() as run:
        model = LinearRegression().fit(train_x, trainy)
        print(f"Model Score: {model.score(train_x, trainy)}")

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
    testX (pd.DataFrame): The testing data.
    testy (pd.Series): The testing labels.
    test_data (str): The path to save the test data.

    Returns:
    None
    """
    test_x["cost"] = testy
    test_x.to_csv((Path(test_data) / "test_data.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--model_metadata", type=str, help="Path of model metadata")
    parser.add_argument("--feature_store_name", type=str, help="Name of the feature store")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription ID")
    parser.add_argument("--resource_group_name", type=str, help="Azure resource group name")

    args = parser.parse_args()

    main(
        args.training_data, args.test_data, args.model_output,
        args.model_metadata, args.feature_store_name,
        args.subscription_id, args.resource_group_name
    )
