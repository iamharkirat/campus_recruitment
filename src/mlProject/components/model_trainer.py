import os
from mlProject import logger
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"No data: {e}")
            raise
        
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]].values.ravel()
        test_y = test_data[[self.config.target_column]].values.ravel()

        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            max_depth=self.config.max_depth,
            random_state=42
        )

        try:
            model.fit(train_x, train_y)
        except Exception as e:
            logger.error(f"Failed to train the model: {e}")
            raise

        try:
            joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
            logger.info("Model saved successfully!")
        except Exception as e:
            logger.error(f"Failed to save the model: {e}")
            raise