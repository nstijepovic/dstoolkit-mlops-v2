"""
Transform taxi data for training, including feature definition and engineering.

This class is responsible for transforming and preparing taxi data.
It transforms the input DataFrame and ensures proper data types, feature extraction, and normalization.
Additionally, it includes logic to define features for feature store registration
and enrichment with existing features.
"""

import argparse
import pandas as pd
import numpy as np
from azureml.featurestore.feature_source import CsvFeatureSource
from azureml.featurestore.contracts import (
    DateTimeOffset,
    TransformationCode,
    Column,
    ColumnType,
    TimestampColumn,
)
from azureml.featurestore import create_feature_set_spec
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


class TaxiDataTransformer(TransformationCode):
    """
    Transform taxi data for feature store and machine learning.

    This class implements data cleaning, feature engineering, and normalization for
    the input DataFrame, as well as defining features for feature store registration.
    """

    def __init__(self, config):
        """
        Initialize TaxiDataTransformer with a configuration object.

        Args:
            config: The configuration object containing feature store settings.
        """
        self.config = config

    def transform(self, df, clean_data_path, transformation_code_path):
        """
        Apply transformations to the input DataFrame.

        Args:
            df (pandas.DataFrame): Input DataFrame to transform.
            clean_data_path (str): Path to the cleaned data.
            transformation_code_path (str): Path to the transformation code.

        Returns:
            pandas.DataFrame: Transformed DataFrame ready for machine learning.
            FeatureSetSpecification: Feature set specification for registration.
        """
        df = self._clean_and_transform_data(df)
        feature_set_spec = self._define_features(clean_data_path, transformation_code_path)
        return df, feature_set_spec

    def _clean_and_transform_data(self, df):
        """
        Clean and transform the data by filtering, renaming, and feature engineering.

        Args:
            df (pandas.DataFrame): The DataFrame to transform.

        Returns:
            pandas.DataFrame: The transformed DataFrame.
        """
        df = df.astype({
            "pickup_longitude": "float64",
            "pickup_latitude": "float64",
            "dropoff_longitude": "float64",
            "dropoff_latitude": "float64",
        })

        df = df[
            (df.pickup_longitude <= -73.72) & (df.pickup_longitude >= -74.09) &
            (df.pickup_latitude <= 40.88) & (df.pickup_latitude >= 40.53) &
            (df.dropoff_longitude <= -73.72) & (df.dropoff_longitude >= -74.72) &
            (df.dropoff_latitude <= 40.88) & (df.dropoff_latitude >= 40.53)
        ]

        df.reset_index(drop=True, inplace=True)
        df["store_forward"] = df["store_forward"].replace("0", "N").fillna("N")
        df["distance"] = df["distance"].replace(".00", 0).fillna(0).astype("float64")

        df = self._add_datetime_features(df)
        df["store_forward"] = np.where(df["store_forward"] == "N", 0, 1)
        df = df[(df["distance"] > 0) & (df["cost"] > 0)]

        df.reset_index(drop=True, inplace=True)
        return df

    def _add_datetime_features(self, df):
        """Add date and time features to the DataFrame."""
        pickup_temp = pd.DatetimeIndex(df["pickup_datetime"], dtype="datetime64[ns]")
        df["pickup_weekday"] = pickup_temp.dayofweek
        df["pickup_month"] = pickup_temp.month
        df["pickup_monthday"] = pickup_temp.day
        df["pickup_hour"] = pickup_temp.hour
        df["pickup_minute"] = pickup_temp.minute
        df["pickup_second"] = pickup_temp.second

        dropoff_temp = pd.DatetimeIndex(df["dropoff_datetime"], dtype="datetime64[ns]")
        df["dropoff_weekday"] = dropoff_temp.dayofweek
        df["dropoff_month"] = dropoff_temp.month
        df["dropoff_monthday"] = dropoff_temp.day
        df["dropoff_hour"] = dropoff_temp.hour
        df["dropoff_minute"] = dropoff_temp.minute
        df["dropoff_second"] = dropoff_temp.second

        df.drop(["dropoff_datetime"], axis=1, inplace=True)
        return df

    def _define_features(
        self,
        clean_data,
        transformation_code_path,
        subscription_id,
        resource_group_name,
        feature_store_name
    ):
        """
        Define the features for feature store registration.

        Args:
            clean_data (str): Path to the cleaned data.
            transformation_code_path (str): Path to the transformation code.
            subscription_id (str): Azure subscription ID.
            resource_group_name (str): Azure resource group name.
            feature_store_name (str): Name of the feature store.

        Returns:
            FeatureSetSpecification: The feature set specification for registration.
        """
        fs_client = MLClient(
            DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            feature_store_name=feature_store_name
        )

        feature_set_spec = create_feature_set_spec(
            source=CsvFeatureSource(
                path=clean_data,
                timestamp_column=TimestampColumn(name="pickup_datetime"),
                source_delay=DateTimeOffset(days=0, hours=0, minutes=20),
            ),
            transformation_code=TransformationCode(
                path=transformation_code_path,
                transformer_class="TaxiDataTransformer",
            ),
            index_columns=[Column(name="vendorID", type=ColumnType.string)],
            source_lookback=DateTimeOffset(days=7, hours=0, minutes=0),
            temporal_join_lookback=DateTimeOffset(days=1, hours=0, minutes=0),
            infer_schema=True,
        )

        poller = fs_client.feature_sets.begin_create_or_update(
            name="london_taxi_features",
            feature_set_spec=feature_set_spec,
            entity_name="taxi_trip",
        )
        feature_set = poller.result()

        return feature_set_spec, feature_set


def get_enriched_data(transformed_data, feature_store_name, subscription_id, resource_group_name):
    """
    Enrich the transformed data by pulling features from the Azure Feature Store.

    Parameters:
    transformed_data (pd.DataFrame): The transformed DataFrame that needs to be enriched.
    feature_store_name (str): The name of the feature store.
    subscription_id (str): Azure subscription ID.
    resource_group_name (str): Azure resource group name.

    Returns:
    pd.DataFrame: The enriched DataFrame with additional features from the feature store.
    """
    # Initialize the Azure ML client
    fs_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name
    )

    # Retrieve the feature set from the feature store
    feature_set = fs_client.feature_sets.get(name=feature_store_name, version="latest")

    # Assuming that "vendorID" is the key column to join the feature store data with the transformed data
    enriched_data = feature_set.join_on(transformed_data, join_column="vendorID")

    return enriched_data


def main(
    clean_data,
    transformation_code_path,
    transformed_data,
    feature_store_name,
    subscription_id,
    resource_group_name
):
    """
    Main entry point for transforming and enriching taxi data.

    Args:
        clean_data (str): Path to the cleaned data.
        transformation_code_path (str): Path to the transformation code.
        transformed_data (str): Path to save the transformed data.
        feature_store_name (str): Name of the feature store.
        subscription_id (str): Azure subscription ID.
        resource_group_name (str): Azure resource group name.
    """
    df = pd.read_csv(clean_data)

    # Transform the data
    transformer = TaxiDataTransformer(config={})
    transformed_df, feature_set_spec = transformer.transform(df, clean_data, transformation_code_path)

    # Enrich the data from the feature store
    transformed_data_with_features = get_enriched_data(
        transformed_df, feature_store_name, subscription_id, resource_group_name)

    # Save the final transformed and enriched data
    transformed_data_with_features.to_csv(transformed_data, index=False)

    return transformed_data_with_features, feature_set_spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser("transform")
    parser.add_argument("--clean_data", type=str, help="Path to prepped data")
    parser.add_argument("--transformation_code_path", type=str, help="Path to the transformation code")
    parser.add_argument("--transformed_data_with_features", type=str, help="Path of output data")
    parser.add_argument("--feature_store_name", type=str, help="Name of the feature store")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription ID")
    parser.add_argument("--resource_group_name", type=str, help="Azure resource group name")

    args = parser.parse_args()
    main(
        args.clean_data,
        args.transformation_code_path,
        args.transformed_data_with_features,
        args.feature_store_name,
        args.subscription_id,
        args.resource_group_name
    )
