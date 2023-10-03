import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig
from typing import Tuple
from sklearn.preprocessing import LabelEncoder

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def handle_null_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle null values in the data."""
        logger.info("Handling Null Values...")
        return data.fillna(0, inplace=True)
    
    def drop_columns(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Drop specified columns."""
        logger.info(f"Dropping Columns: {', '.join(columns)}")
        return data.drop(columns, axis=1, inplace=True)

    def handle_outliers(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Handle outliers in a specified column."""
        logger.info(f"Handling Outliers in Column: {column}")
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        filter = (data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)
        return data.loc[filter]

    def encode_labels(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Label encode specified columns."""
        logger.info(f"Encoding Labels for Columns: {', '.join(columns)}")
        label_encoder = LabelEncoder()
        for col in columns:
            data[col] = label_encoder.fit_transform(data[col])
        return data
    
    def dummy_encode(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Dummy encode specified columns."""
        logger.info(f"Dummy Encoding Columns: {', '.join(columns)}")
        for col in columns:
            dummies = pd.get_dummies(data[col], prefix=f'dummy_', drop_first=True)
            dummies = dummies.astype(int)
            data = pd.concat([data, dummies], axis=1)
            data = data.drop(columns=[col])
        return data

    
    def split_and_save(self, data: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets and save them as CSV files."""
        logger.info(f"Splitting Data with Test Size: {test_size}")
        train, test = train_test_split(data, test_size=test_size)
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        logger.info("Data has been split and saved.")
        return train, test

    def perform_data_transformation(self):
        """Perform data transformations and split data into train and test."""
        logger.info("Starting Data Transformation Process...")
        data = pd.read_csv(self.config.data_path)
        self.handle_null_values(data)
        self.drop_columns(data, ['sl_no', 'ssc_b', 'hsc_b'])
        data = self.handle_outliers(data, 'hsc_p')
        data = self.encode_labels(data, ['gender', 'workex', 'specialisation', 'status'])
        data = self.dummy_encode(data, ['hsc_s', 'degree_t'])
        train, test = self.split_and_save(data, test_size=0.25)
        logger.info("Data Transformation Process Completed.")