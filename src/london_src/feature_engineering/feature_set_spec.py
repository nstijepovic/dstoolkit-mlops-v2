from azure.identity import DefaultAzureCredential
from azureml.featurestore import FeatureStoreClient
from azureml.featurestore import get_offline_features
import pandas as pd


def get_enriched_data(training_data, feature_store_name):
    """
    Retrieve features from the feature store and enrich the training data.

    Parameters:
    training_data (str): Path to the training data
    feature_store_name (str): Name of the feature store

    Returns:
    DataFrame: Enriched training data
    """
    # Read training data
    train_data = pd.read_parquet(training_data)

    # Initialize feature store client
    featurestore = FeatureStoreClient(
        credential=DefaultAzureCredential(),
        feature_store_name=feature_store_name,
    )

    # Retrieve features from the feature store
    feature_set = featurestore.feature_sets.get("transactions", "1")
    features = [
        feature_set.get_feature("transaction_amount_7d_sum"),
        feature_set.get_feature("transaction_amount_7d_avg"),
        feature_set.get_feature("transaction_3d_count"),
        feature_set.get_feature("transaction_amount_3d_avg"),
    ]

    # Generate enriched dataframe by using feature data and observation data
    enriched_train_data = get_offline_features(
        features=features,
        observation_data=train_data,
        timestamp_column="timestamp",
    )

    return enriched_train_data
