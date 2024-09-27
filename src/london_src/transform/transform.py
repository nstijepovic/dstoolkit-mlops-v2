"""
Transform taxi data for training, including feature definition and engineering.

This module is responsible for transforming and preparing taxi data.
It transforms the input DataFrame and ensures proper data types, feature extraction,
and normalization. Additionally, it includes logic to define features for feature
store registration and enrichment with existing features.
"""

import argparse
import os
import datetime
from pathlib import Path
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.featurestore import create_feature_set_spec
from azureml.featurestore.feature_source import CsvFeatureSource
from azure.ai.ml.entities import (
    FeatureStore,
    FeatureStoreEntity,
    FeatureSet,
)
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

    def __init__(self, config=None):
        """
        Initialize TaxiDataTransformer with a configuration object.

        Args:
            config: The configuration object containing feature store settings.
        """
        self.config = config or {}
        self.spark = SparkSession.builder.getOrCreate()

    def transform(self, df):
        """
        Clean and transform the input DataFrame.

        Args:
            df (pyspark.sql.DataFrame): The DataFrame to transform.

        Returns:
            pyspark.sql.DataFrame: Transformed DataFrame.
        """
        return self._clean_and_transform_data(df)

    def _clean_and_transform_data(self, df):
        """
        Clean and transform the data by filtering, renaming, and feature engineering.

        Args:
            df (pyspark.sql.DataFrame): The DataFrame to transform.

        Returns:
            pyspark.sql.DataFrame: The transformed DataFrame.
        """
        df = df.filter(
            (f.col("pickup_longitude") <= -73.72)
            & (f.col("pickup_longitude") >= -74.09)
            & (f.col("pickup_latitude") <= 40.88)
            & (f.col("pickup_latitude") >= 40.53)
            & (f.col("dropoff_longitude") <= -73.72)
            & (f.col("dropoff_longitude") >= -74.72)
            & (f.col("dropoff_latitude") <= 40.88)
            & (f.col("dropoff_latitude") >= 40.53)
        )

        df = df.withColumn(
            "store_forward",
            f.when(f.col("store_forward") == "0", "N").otherwise(f.col("store_forward"))
        )
        df = df.withColumn(
            "store_forward",
            f.when(f.col("store_forward").isNull(), "N").otherwise(f.col("store_forward"))
        )
        df = df.withColumn(
            "distance",
            f.when(f.col("distance") == ".00", 0).otherwise(f.col("distance"))
        )
        df = df.withColumn("distance", f.col("distance").cast("float"))

        df = self._add_datetime_features(df)
        df = df.withColumn(
            "store_forward",
            f.when(f.col("store_forward") == "N", 0).otherwise(1)
        )
        df = df.filter((f.col("distance") > 0) & (f.col("cost") > 0))

        return df

    def _add_datetime_features(self, df):
        """Add date and time features to the DataFrame."""
        df = df.withColumn("pickup_datetime", f.to_timestamp("pickup_datetime"))
        df = df.withColumn("pickup_weekday", f.dayofweek("pickup_datetime"))
        df = df.withColumn("pickup_month", f.month("pickup_datetime"))
        df = df.withColumn("pickup_monthday", f.dayofmonth("pickup_datetime"))
        df = df.withColumn("pickup_hour", f.hour("pickup_datetime"))
        df = df.withColumn("pickup_minute", f.minute("pickup_datetime"))
        df = df.withColumn("pickup_second", f.second("pickup_datetime"))

        df = df.withColumn("dropoff_datetime", f.to_timestamp("dropoff_datetime"))
        df = df.withColumn("dropoff_weekday", f.dayofweek("dropoff_datetime"))
        df = df.withColumn("dropoff_month", f.month("dropoff_datetime"))
        df = df.withColumn("dropoff_monthday", f.dayofmonth("dropoff_datetime"))
        df = df.withColumn("dropoff_hour", f.hour("dropoff_datetime"))
        df = df.withColumn("dropoff_minute", f.minute("dropoff_datetime"))
        df = df.withColumn("dropoff_second", f.second("dropoff_datetime"))

        df = df.drop("dropoff_datetime")
        return df

    def register_features(
        self,
        clean_data,
        transformation_code_path,
    ):
        """
        Register the features from the transformed DataFrame to the Azure Feature Store.

        Args:
            clean_data (str): Path to the cleaned data.
            transformation_code_path (str): Path to the transformation code.

        Returns:
            FeatureSetSpecification: The feature set specification for registration.
        """
        subscription_id = "a1e2f839-e403-4d66-9a4b-96f6c743606f"
        resource_group_name = "mlopsv2-rg"
        ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
        )

        try:
            # Create the feature store
            fs = FeatureStore(name="LondonFS", location="eastus")
            print("Creating feature store...")
            fs_poller = ml_client.feature_stores.begin_create(fs)
            feature_store_result = fs_poller.result()  # Ensure that the creation completes
            print(f"Feature store creation result: {feature_store_result}")

            # Define feature set specification
            feature_set_spec = create_feature_set_spec(
                source=CsvFeatureSource(
                    path=clean_data,
                    timestamp_column=TimestampColumn(name="pickup_datetime"),
                    source_delay=DateTimeOffset(days=0, hours=0, minutes=20),
                ),
                transformation_code=TransformationCode(
                    path=transformation_code_path,
                    transformer_class="transform.TaxiDataTransformer",
                ),
                index_columns=[Column(name="vendorID", type=ColumnType.string)],
                source_lookback=DateTimeOffset(days=7, hours=0, minutes=0),
                temporal_join_lookback=DateTimeOffset(days=1, hours=0, minutes=0),
                infer_schema=True,
            )

            # Register the feature set
            print("Registering feature set...")
            poller = ml_client.feature_sets.begin_create_or_update(
                name="london_taxi_features",
                feature_set_spec=feature_set_spec,
                entity_name="taxi_trip",
            )
            result = poller.result()  # Ensure that the registration completes
            print(f"Feature set registration result: {result}")
            return result

        except Exception as e:
            print(f"Error during feature store creation or feature registration: {e}")
            raise


def get_enriched_data(
    transformed_data,
    feature_store_name
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
    subscription_id = "a1e2f839-e403-4d66-9a4b-96f6c743606f"
    resource_group_name = "mlopsv2-rg"
    fs_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name
    )

    # Retrieve the feature set from the feature store
    feature_set = fs_client.feature_sets.get(name=feature_store_name, version="latest")

    # check all feature stores and feature sets
    feature_stores = fs_client.feature_stores.list()
    if not feature_stores:
        print("No feature stores found.")
        return
    print("Available feature stores:")
    for feature_store in feature_stores:
        print(f"- {feature_store.name}")

        # List all feature sets within the feature store
        feature_sets = fs_client.feature_sets.list(feature_store_name=feature_store.name)

        if not feature_sets:
            print(f"  No feature sets found in store '{feature_store.name}'.")
            continue

        for feature_set in feature_sets:
            print(f"  Feature set: {feature_set.name}")

    # Assuming "vendorID" is the key column to join with the transformed data
    enriched_data = feature_set.join_on(transformed_data, join_column="vendorID")

    return enriched_data


def main(
    clean_data,
    transformation_code_path,
    feature_store_name
):
    """
    Run the transformation and enrichment process for taxi data.

    Args:
        clean_data (str): Path to the cleaned data.
        transformation_code_path (str): Path to the transformation code.
        feature_store_name (str): Name of the feature store.
    """
    # Initialize Spark session
    spark = (SparkSession.builder
             .appName("TaxiDataTransformer")
             .config("spark.driver.memory", "4g")
             .config("spark.executor.memory", "4g")
             .getOrCreate()
             )

    # Check if the input is a directory, and read all CSV files
    if os.path.isdir(clean_data):
        df = spark.read.csv(clean_data, header=True, inferSchema=True).drop("_c0")
    else:
        df = spark.read.csv(clean_data, header=True, inferSchema=True).drop("_c0")

    # Transform the data
    transformer = TaxiDataTransformer(config={})
    transformed_df = transformer.transform(df)

    # Optionally register features in the Azure Feature Store
    if feature_store_name:
        try:
            feature_set_spec = transformer.register_features(
                transformed_df, clean_data, transformation_code_path
            )
            print("Feature Set Registered:", feature_set_spec)
        except Exception as e:
            print(f"Error during feature registration: {e}")

    # Add timestamp to the output path and save transformed data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    transformed_data_with_timestamp = Path(args.transformed_data_with_features) / f"transformed_data_{timestamp}"

    try:
        transformed_df.write.csv(str(transformed_data_with_timestamp), header=True, mode="overwrite")
        print(f"Successfully wrote data to {transformed_data_with_timestamp}")
    except Exception as e:
        print(f"Error writing data: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("transform")
    parser.add_argument("--clean_data", type=str, help="Path to prepped data")
    parser.add_argument("--transformation_code_path", type=str, help="Path to the transformation code")
    parser.add_argument("--transformed_data_with_features", type=str, help="Path of output data")
    parser.add_argument("--feature_store_name", type=str, help="Name of the feature store")

    args = parser.parse_args()
    main(
        args.clean_data,
        args.transformation_code_path,
        args.feature_store_name
    )
