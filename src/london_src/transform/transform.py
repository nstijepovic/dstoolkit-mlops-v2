"""
Transform and prepare taxi data for training, including feature definition and engineering.

This class is responsible for transforming and preparing taxi data.
It transforms the input DataFrame and ensures proper data types, feature extraction, and normalization.
Additionally, it includes logic to define features for feature store registration.
"""

import pandas as pd
import numpy as np
from azureml.featurestore.contracts import Transformation
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
from mlops.common.config_utils import MLOpsConfig


class TaxiDataTransformer(Transformation):
    """
    Transform taxi data for feature store and machine learning.

    This class implements data cleaning, feature engineering, and normalization for
    the input DataFrame, as well as defining features for feature store registration.
    """

    def __init__(self, config):
        self.config = config

    def transform(self, df, clean_data_path, transformation_code_path):
        """
        Apply transformations to the input DataFrame.

        Parameters:
            df (pandas.DataFrame): Input DataFrame to transform.
            clean_data_path (str): Path to the cleaned data.
            transformation_code_path (str): Path to the transformation code.

        Returns:
            pandas.DataFrame: Transformed DataFrame ready for machine learning.
            FeatureSetSpecification: The feature set specification for registration.
        """
        # Clean and transform the data
        df = self._clean_and_transform_data(df)

        # Define the features and register them in the feature store
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
        # Ensure correct data types for lat/long fields
        df = df.astype({
            "pickup_longitude": "float64",
            "pickup_latitude": "float64",
            "dropoff_longitude": "float64",
            "dropoff_latitude": "float64",
        })

        # Filter out coordinates outside the city border
        df = df[
            (df.pickup_longitude <= -73.72) & (df.pickup_longitude >= -74.09)
            & (df.pickup_latitude <= 40.88) & (df.pickup_latitude >= 40.53)
            & (df.dropoff_longitude <= -73.72) & (df.dropoff_longitude >= -74.72)
            & (df.dropoff_latitude <= 40.88) & (df.dropoff_latitude >= 40.53)
        ]

        df.reset_index(inplace=True, drop=True)

        # Replace undefined values and rename columns to meaningful names
        df["store_forward"] = df["store_forward"].replace("0", "N").fillna("N")
        df["distance"] = df["distance"].replace(".00", 0).fillna(0)
        df = df.astype({"distance": "float64"})

        # Date and time feature engineering
        df = self._add_datetime_features(df)

        # Change 'store_forward' to binary values
        df["store_forward"] = np.where(df["store_forward"] == "N", 0, 1)

        # Filter out rows where distance or cost is zero
        df = df[(df["distance"] > 0) & (df["cost"] > 0)]

        df.reset_index(inplace=True, drop=True)
        return df

    def _add_datetime_features(self, df):
        """
        Add date and time features to the DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to transform.

        Returns:
            pandas.DataFrame: DataFrame with new date/time features.
        """
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

    def _define_features(self, clean_data_path, transformation_code_path):
        """
        Define the features for feature store registration.

        Args:
            clean_data_path (str): Path to the cleaned data.
            transformation_code_path (str): Path to the transformation code.

        Returns:
            FeatureSetSpecification: The feature set specification for registration.
        """
        # Load the configuration
        config = MLOpsConfig()

        # Initialize the feature store client
        fs_client = MLClient(
            DefaultAzureCredential(),
            config.feature_store_config["subscription_id"],
            config.feature_store_config["resource_group_name"],
            feature_store_name=config.feature_store_config["name"]
        )

        # Define the feature set spec
        feature_set_spec = create_feature_set_spec(
            source=CsvFeatureSource(
                path=clean_data_path,
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

        # Register the feature set with the feature store
        feature_set_name = "london_taxi_features"
        entity_name = "taxi_trip"
        poller = fs_client.feature_sets.begin_create_or_update(
            name=feature_set_name,
            feature_set_spec=feature_set_spec,
            entity_name=entity_name,
        )
        feature_set = poller.result()

        return feature_set_spec, feature_set
