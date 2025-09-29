Project Documentation — Predictive Maintenance MLOps Pipeline
1. Overview

This project implements a Predictive Maintenance (PdM) pipeline using TFX (TensorFlow Extended) as the orchestration framework.
The pipeline is designed to reach MLOps maturity level 2 (pseudo-L2): enabling reproducible training, continuous integration of data and code, model validation, and automated deployment to serving.

The system ingests preprocessed tabular data (CSV + pandas feature engineering), validates its schema, trains ML models, validates them against business metrics, and automatically promotes validated models into serving.

2. Architecture Overview
🔹 Key stages

Data Ingestion (CsvExampleGen)
Reads CSVs containing engineered features and labels generated in preprocessing (preprocess.py with pandas).

Data Validation (StatisticsGen + SchemaGen + ExampleValidator)
Ensures input data matches expected schema, detects anomalies, missing values, and drift in incoming batches.

Transform (Transform)
In this demo, a passthrough preprocessing_fn is used (features are already engineered in pandas).
In production, this stage could handle normalization, vocabularies, and bucketization with tf.Transform.

Training (Trainer)
Trains an ML model (classification/regression depending on the PdM target). Training is defined in trainer.py.
Continuous Training (CT) is achieved by rerunning this stage when new validated data arrives.

Model Validation (Evaluator)
Evaluates trained models against baseline metrics (accuracy, precision, recall, F1, business KPIs).
Only models passing the thresholds are "blessed" for serving.

Model Pushing (Pusher)
Deploys validated models into a serving environment (local for demo, cloud-ready in future).
This achieves Continuous Deployment (CD).

3. MLOps Maturity — Level 2 (Pseudo)
Aspect	Implementation in this Project
Data & Model Validation	Automated schema check, anomaly detection, and model evaluation thresholds.
CT (Continuous Training)	Pipeline retrains automatically when new validated CSV data is available.
CD (Continuous Deployment)	Only validated ("blessed") models are pushed to serving.
Pipeline Orchestration	Managed by TFX with pluggable orchestrators (Airflow, Kubeflow, or local runner).
Reproducibility	Data versioning (via ExampleGen) and pipeline artifact tracking (ML Metadata).
Demo mode	CSV ingestion with pandas preprocessing passthrough; easy to migrate to production.
4. Tech Stack

Programming Language: Python 3.10

Framework: TensorFlow Extended (TFX ≥1.15)

Preprocessing: Pandas (offline), tf.Transform passthrough (online)

Experiment Tracking: ML Metadata (built into TFX)

Dependency Management: uv + pyproject.toml

Containerization: Docker (Linux x86_64 base, compatible with M1/M2 via emulation)

Orchestration: Local DAG runner (demo); extensible to Airflow or Kubeflow

5. Workflow Diagram
           ┌───────────────┐
           │   CSV Data    │
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │ Data Validation│
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │   Transform   │  (passthrough, features already engineered in pandas)
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │   Trainer     │  (Continuous Training)
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │   Evaluator   │  (model validation)
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │    Pusher     │  (Continuous Deployment)
           └───────────────┘

6. Future Extensions

Data Drift Detection: Extend ExampleValidator to trigger retraining when schema drifts.

Model Monitoring: Track inference stats in production to detect concept drift.

Automated Retraining: Schedule pipeline with Airflow/Kubeflow.

Cloud Deployment: Export Pusher to TensorFlow Serving / Vertex AI / Sagemaker.