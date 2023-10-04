import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from mlProject import logger
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        accuracy = accuracy_score(actual, pred)
        logger.info(f"Evaluation Metrics - RMSE: {rmse}, MAE: {mae}, R2 Score: {r2}, Accuracy: {accuracy}")
        return rmse, mae, r2, accuracy

    def save_metrics_locally(self, scores):
        # Ensure you have a save_json function or use an appropriate method to save the scores as JSON.
        save_json(path=Path(self.config.metric_file_name), data=scores)

    def log_into_mlflow(self):
        logger.info("Model evaluation started.")
        logger.info("Starting MLflow logging...")

        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Split data into features and target
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            # Evaluate metrics
            rmse, mae, r2, accuracy = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics locally and log parameters and metrics to MLflow
            scores = {"rmse": rmse, "mae": mae, "r2": r2, "accuracy": accuracy}
            self.save_metrics_locally(scores)
            logger.info("Metrics saved locally.")
            
            logger.info("Logging parameters and metrics to MLflow.")
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(scores)

            # Log model to MLflow
            logger.info("Logging model to MLflow.")
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifier")
                logger.info("Model registered in MLflow.")
            else:
                mlflow.sklearn.log_model(model, "model")
                logger.info("Model logged to MLflow.")
        
        logger.info("Model evaluation completed.")
