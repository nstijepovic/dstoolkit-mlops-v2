"""
Transform and prepare taxi data for training.

This class is responsible for transforming and preparing taxi data.
It transforms the input DataFrame and ensures proper data types, feature extraction, and normalization.
"""

import pandas as pd
import numpy as np
from azureml.featurestore.contracts import Transformation


class TaxiDataTransformer(Transformation):
    """
    Transform taxi data for feature store.

    This class implements data cleaning, feature engineering, and normalization for
    the input DataFrame to prepare it for machine learning models.
    """

    def transform(self, df):
        """
        Apply transformations to the input DataFrame.

        Parameters:
            df (pandas.DataFrame): Input DataFrame to transform.

        Returns:
            pandas.DataFrame: Transformed DataFrame ready for machine learning.
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

        Extract additional features like day of the week, month, and time components
        from the pickup and dropoff datetime fields.

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

        # Drop unnecessary datetime columns
        df.drop(["pickup_datetime", "dropoff_datetime"], axis=1, inplace=True)

        return df
