"""
Transform taxi data for training, including feature definition and engineering.

This module is responsible for transforming and preparing taxi data.
It transforms the input DataFrame and ensures proper data types, feature extraction,
and normalization. Additionally, it includes logic to define features for feature
store registration and enrichment with existing features.
"""

import argparse
import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.featurestore import create_feature_set_spec
from azureml.featurestore.feature_source import CsvFeatureSource
from azureml.featurestore.contracts import (
    DateTimeOffset,
    TransformationCode,
    Column,
    ColumnType,
    TimestampColumn,
)


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
        self.spark = SparkSession.builder.getOrCreate()

    def transform(
        self,
        df,
        clean_data_path,
        transformation_code_path,
        subscription_id,
        resource_group_name,
        feature_store_name
    ):
        """
        Apply transformations to the input DataFrame.

        Args:
            df (pyspark.sql.DataFrame): Input DataFrame to transform.
            clean_data_path (str): Path to the cleaned data.
            transformation_code_path (str): Path to the transformation code.
            subscription_id (str): Azure subscription ID.
            resource_group_name (str): Azure resource group name.
            feature_store_name (str): Name of the feature store.

        Returns:
            pyspark.sql.DataFrame: Transformed DataFrame ready for machine learning.
            FeatureSetSpecification: Feature set specification for registration.
        """
        spark_df = self._clean_and_transform_data(df)

        # Define the features and register them in the feature store
        feature_set_spec = self._define_features(
            clean_data_path, transformation_code_path,
            subscription_id, resource_group_name, feature_store_name, spark_df
        )
        return spark_df, feature_set_spec

    def _clean_and_transform_data(self, df):
        """
        Clean and transform the data by filtering, renaming, and feature engineering.

        Args:
            df (pyspark.sql.DataFrame): The DataFrame to transform.

        Returns:
            pyspark.sql.DataFrame: The transformed DataFrame.
        """
        df = df.filter(
            (F.col("pickup_longitude") <= -73.72)
            & (F.col("pickup_longitude") >= -74.09)
            & (F.col("pickup_latitude") <= 40.88)
            & (F.col("pickup_latitude") >= 40.53)
            & (F.col("dropoff_longitude") <= -73.72)
            & (F.col("dropoff_longitude") >= -74.72)
            & (F.col("dropoff_latitude") <= 40.88)
            & (F.col("dropoff_latitude") >= 40.53)
        )

        df = df.withColumn(
            "store_forward",
            F.when(F.col("store_forward") == "0", "N").otherwise(F.col("store_forward"))
        )
        df = df.withColumn(
            "store_forward",
            F.when(F.col("store_forward").isNull(), "N").otherwise(F.col("store_forward"))
        )
        df = df.withColumn(
            "distance",
            F.when(F.col("distance") == ".00", 0).otherwise(F.col("distance"))
        )
        df = df.withColumn("distance", F.col("distance").cast("float"))

        df = self._add_datetime_features(df)
        df = df.withColumn(
            "store_forward",
            F.when(F.col("store_forward") == "N", 0).otherwise(1)
        )
        df = df.filter((F.col("distance") > 0) & (F.col("cost") > 0))

        return df

    def _add_datetime_features(self, df):
        """Add date and time features to the DataFrame."""
        df = df.withColumn("pickup_datetime", F.to_timestamp("pickup_datetime"))
        df = df.withColumn("pickup_weekday", F.dayofweek("pickup_datetime"))
        df = df.withColumn("pickup_month", F.month("pickup_datetime"))
        df = df.withColumn("pickup_monthday", F.dayofmonth("pickup_datetime"))
        df = df.withColumn("pickup_hour", F.hour("pickup_datetime"))
        df = df.withColumn("pickup_minute", F.minute("pickup_datetime"))
        df = df.withColumn("pickup_second", F.second("pickup_datetime"))

        df = df.withColumn("dropoff_datetime", F.to_timestamp("dropoff_datetime"))
        df = df.withColumn("dropoff_weekday", F.dayofweek("dropoff_datetime"))
        df = df.withColumn("dropoff_month", F.month("dropoff_datetime"))
        df = df.withColumn("dropoff_monthday", F.dayofmonth("dropoff_datetime"))
        df = df.withColumn("dropoff_hour", F.hour("dropoff_datetime"))
        df = df.withColumn("dropoff_minute", F.minute("dropoff_datetime"))
        df = df.withColumn("dropoff_second", F.second("dropoff_datetime"))

        df = df.drop("dropoff_datetime")
        return df

    def _define_features(
        self,
        clean_data,
        transformation_code_path,
        subscription_id,
        resource_group_name,
        feature_store_name,
        spark_df
    ):
        """
        Define the features for feature store registration.

        Args:
            clean_data (str): Path to the cleaned data.
            transformation_code_path (str): Path to the transformation code.
            subscription_id (str): Azure subscription ID.
            resource_group_name (str): Azure resource group name.
            feature_store_name (str): Name of the feature store.
            spark_df (pyspark.sql.DataFrame): The transformed Spark DataFrame.

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


def get_enriched_data(
    transformed_data, feature_store_name, subscription_id, resource_group_name
):
    """
    Enrich the transformed data by pulling features from the Azure Feature Store.

    Parameters:
    transformed_data (pyspark.sql.DataFrame): The transformed DataFrame to be enriched.
    feature_store_name (str): The name of the feature store.
    subscription_id (str): Azure subscription ID.
    resource_group_name (str): Azure resource group name.

    Returns:
    pyspark.sql.DataFrame: The enriched DataFrame with additional features.
    """
    # Initialize the Azure ML client
    fs_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name
    )

    # Retrieve the feature set from the feature store
    feature_set = fs_client.feature_sets.get(name=feature_store_name, version="latest")

    # Assuming "vendorID" is the key column to join with the transformed data
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
    Run the transformation and enrichment process for taxi data.

    Args:
        clean_data (str): Path to the cleaned data.
        transformation_code_path (str): Path to the transformation code.
        transformed_data (str): Path to save the transformed data.
        feature_store_name (str): Name of the feature store.
        subscription_id (str): Azure subscription ID.
        resource_group_name (str): Azure resource group name.
    """
    spark = SparkSession.builder.getOrCreate()

    # Check if the input is a directory, and read all CSV files
    if os.path.isdir(clean_data):
        # Read CSV files using Spark
        df = spark.read.csv(clean_data, header=True, inferSchema=True)
    else:
        df = spark.read.csv(clean_data, header=True, inferSchema=True)

    # Transform the data
    transformer = TaxiDataTransformer(config={})
    transformed_df, feature_set_spec = transformer.transform(
        df, clean_data, transformation_code_path,
        subscription_id, resource_group_name, feature_store_name
    )

    # Enrich the data from the feature store
    transformed_data_with_features = get_enriched_data(
        transformed_df, feature_store_name, subscription_id, resource_group_name
    )

    # Save the final transformed and enriched data
    transformed_data_with_features.write.csv(transformed_data, header=True, mode="overwrite")

    return transformed_data_with_features, feature_set_spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser("transform")
    parser.add_argument("--clean_data", type=str, help="Path to prepped data")
    parser.add_argument(
        "--transformation_code_path", type=str, help="Path to the transformation code"
    )
    parser.add_argument(
        "--transformed_data_with_features", type=str, help="Path of output data"
    )
    parser.add_argument("--feature_store_name", type=str, help="Name of the feature store")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription ID")
    parser.add_argument("--resource_group_name", type=str, help="Azure resource group name")
    parser.add_argument(
        "--feature_set_specification",
        type=str,
        help="Path to the feature set specification",
        required=False
    )
    parser.add_argument(
        "--registered_feature_set",
        type=str,
        help="Path to the registered feature set",
        required=False
    )

    args = parser.parse_args()
    main(
        args.clean_data,
        args.transformation_code_path,
        args.transformed_data_with_features,
        args.feature_store_name,
        args.subscription_id,
        args.resource_group_name
    )
