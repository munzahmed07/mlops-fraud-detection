# 🚨 End-to-End Fraud Detection System with MLOps

A production-style **Machine Learning + MLOps** project that demonstrates the full lifecycle of an ML system — from raw data ingestion to deployment, monitoring, and automated retraining.

This project is built to reflect **real-world engineering practices**, not just notebook experiments.

---

## 📌 What this project shows

* How to build a **complete ML pipeline**
* How to track experiments with **MLflow**
* How to deploy models with **FastAPI**
* How to log predictions in production
* How to detect **data drift**
* How to trigger **automated retraining**
* How to manage ML projects the **industry way**

---

## 🏗️ System Architecture

```
Raw Data
   │
   ▼
Data Ingestion  ──▶ Cleaning ──▶ Feature Engineering
                                   │
                                   ▼
                             Model Training
                         (LogReg, RF, XGBoost)
                                   │
                                   ▼
                           MLflow Tracking
                           + Model Registry
                                   │
                                   ▼
                           FastAPI Inference
                               /predict
                                   │
                                   ▼
                         Prediction Logging
                                   │
                                   ▼
                         Drift Detection
                               (Evidently)
                                   │
                                   ▼
                       Automated Retraining
```

---

## 🧠 Models Used

* Logistic Regression
* Random Forest
* XGBoost

Evaluation metrics:

* ROC-AUC
* Precision
* Recall
* F1-score

---

## 📁 Project Structure

```
mlops-fraud-detection/
│
├── src/
│   ├── ingestion/        # Data loading & validation
│   ├── preprocessing/   # Cleaning & missing value handling
│   ├── features/        # Feature engineering
│   ├── training/        # Model training + MLflow
│   ├── serving/         # FastAPI app
│   ├── monitoring/      # Drift detection
│   └── retraining/      # Automated retrain pipeline
│
├── data/                # (ignored in git)
├── logs/                # Prediction logs
├── reports/             # Drift reports
├── README.md
└── .gitignore
```

---

## ⚙️ Tech Stack

* **Python**
* **scikit-learn**
* **XGBoost**
* **MLflow**
* **FastAPI**
* **Evidently AI**
* **Git + GitHub**

---

## 🚀 How to Run Locally

### 1️⃣ Create environment

```bash
conda create -n fraud python=3.10
conda activate fraud
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧩 Pipeline Execution

### Data Ingestion

```bash
python src/ingestion/ingest.py
```

### Cleaning

```bash
python src/preprocessing/clean.py
```

### Feature Engineering

```bash
python src/features/build_features.py
```

### Train Models

```bash
python src/training/train_with_mlflow.py
```

### Register Best Model

```bash
python src/training/register_model.py
```

---

## 🔍 MLflow UI

```bash
mlflow ui
```

Open:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🌐 Run API

```bash
uvicorn src.serving.api:app --reload
```

API will be live at:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

Docs:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📮 Example Prediction

POST `/predict`

```json
{
  "features": [0.13, 0.44, 1.02, ...]
}
```

Response:

```json
{
  "fraud_probability": 0.73,
  "fraud_prediction": 1,
  "latency_sec": 0.42
}
```

---

## 📊 Monitoring & Drift Detection

```bash
python src/monitoring/drift_check.py
```

Output:

```
reports/drift_report.txt
```

Contains:

* Drift detected or not
* Feature checked
* Monitoring status

---

## 🔁 Automated Retraining

```bash
python src/retraining/retrain_pipeline.py
```

What it does:

1. Reads drift report
2. If drift detected → retrains models
3. Logs new run in MLflow
4. Registers new model version

---

## 🧑‍💻 What I learned

* How production ML differs from notebooks
* How to structure real ML projects
* How MLflow works in practice
* How to serve ML models reliably
* Why monitoring & retraining matter
* How real MLOps pipelines are built

---

## 🎯 Resume-ready Description

> Built an end-to-end fraud detection system with full MLOps lifecycle including data pipelines, MLflow-based experiment tracking, FastAPI deployment, prediction logging, drift detection using Evidently AI, and an automated retraining pipeline.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!
