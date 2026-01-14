# src/training/register_model.py

import mlflow

RUN_ID = "5f0a6a6775c74c549fadf90e962eadb3"
MODEL_NAME = "fraud-detection-model"

model_uri = f"runs:/{RUN_ID}/model"

print(f"Registering model from: {model_uri}")

mlflow.register_model(model_uri, MODEL_NAME)

print("Model registered successfully.")
