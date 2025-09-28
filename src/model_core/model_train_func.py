import pandas as pd
import numpy as np
import xgboost as xgb
import joblib # Library for saving and loading models efficiently
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta

# --- GLOBAL CONFIGURATION (for formality and centralization) ---
MODEL_FILE_TEMPLATE = "model_snapshot_{}.joblib"
METADATA_FILE_TEMPLATE = "metadata_snapshot_{}.joblib"
TARGET_COLUMN = 'failure'
EXCLUDE_COLS = ['datetime', 'machineID', TARGET_COLUMN]


# --- 1. DATA PREPARATION FUNCTION (Modified for Clarity and Formal Naming) ---

def prepare_time_series_split(
    data_frame: pd.DataFrame, 
    train_end_date: datetime, 
    inference_start_date: datetime, 
    inference_end_date: datetime,
    encoder: LabelEncoder
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Splits the labeled feature set into training and inference (test) sets, 
    handling feature alignment and label encoding.

    Args:
        data_frame: The full DataFrame containing all features and the target label.
        train_end_date: The exclusive cutoff date for the training data (X_train, y_train).
        inference_start_date: The inclusive start date for the inference/test data.
        inference_end_date: The exclusive end date for the inference/test data.
        encoder: The fitted LabelEncoder object.

    Returns:
        Tuple[X_train, y_train, X_inference, y_inference]
    """
    
    # 2. Leave a month of data for inference/testing
    train_data = data_frame.loc[data_frame['datetime'] < train_end_date]
    inference_data = data_frame.loc[
        (data_frame['datetime'] >= inference_start_date) & 
        (data_frame['datetime'] < inference_end_date)
    ]
    
    # Process features (X) by dropping identifiers and one-hot encoding
    train_X = train_data.drop(EXCLUDE_COLS, axis=1).pipe(pd.get_dummies)
    inference_X = inference_data.drop(EXCLUDE_COLS, axis=1).pipe(pd.get_dummies)
    
    # Align columns between train and inference sets (critical step)
    train_X, inference_X = train_X.align(inference_X, join='outer', axis=1, fill_value=0)
    
    # Ensure all NaN values introduced by outer join are treated as 0 for model input
    train_X = train_X.fillna(0)
    inference_X = inference_X.fillna(0)
    
    # Process labels (y)
    train_y = encoder.transform(train_data[TARGET_COLUMN])
    inference_y = encoder.transform(inference_data[TARGET_COLUMN])
    
    return train_X, train_y, inference_X, inference_y


# --- 3 & 5. TRAINING FUNCTION (Saves Model and Returns Metrics) ---

def train_and_validate_model(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    X_inference: pd.DataFrame, 
    y_inference: np.ndarray, 
    model_version: str, 
    n_classes: int,
    le: LabelEncoder
) -> Dict[str, Any]:
    """
    Initializes, trains, saves the model, and calculates performance metrics.
    
    Args:
        ... [arguments defined above]
        model_version: A string identifier (e.g., date) for saving the model.

    Returns:
        Dict[str, Any]: Dictionary containing all evaluation metrics.
    """
    
    # Initialize XGBoost Classifier (formal naming: Classifier Instance)
    clf_instance = xgb.XGBClassifier(
        objective='multi:softmax', 
        num_class=n_classes, 
        n_estimators=100,
        use_label_encoder=False, 
        eval_metric='mlogloss',
        random_state=42
    )
    
    print(f"[{model_version}] Starting model training...")
    clf_instance.fit(X_train, y_train)

    # 3. Save the model for future inference
    model_path = MODEL_FILE_TEMPLATE.format(model_version)
    joblib.dump(clf_instance, model_path)
    print(f"[{model_version}] Model successfully saved to {model_path}")

    # Predict and Evaluate
    predictions = clf_instance.predict(X_inference)
    
    # 5. Calculate and return metrics
    metrics = {
        'version': model_version,
        'f1_weighted': f1_score(y_inference, predictions, average='weighted'),
        'accuracy': accuracy_score(y_inference, predictions),
        'classes': list(le.classes_),
        'report': classification_report(
            y_inference, 
            predictions, 
            target_names=le.classes_, 
            output_dict=True
        )
    }
    
    return metrics

def main_pipeline_training(labeled_features: pd.DataFrame, le: LabelEncoder):
    """Orchestrates the time-series cross-validation, training, and result logging."""
    
    # 2. Modified THRESHOLD_DATES for clean, one-month validation windows
    # The structure is now: (End of Train, Start of Inference, End of Inference)
    CV_WINDOWS = [
        (datetime(2015, 7, 31, 1), datetime(2015, 8, 1, 1), datetime(2015, 9, 1, 1)),
        (datetime(2015, 8, 31, 1), datetime(2015, 9, 1, 1), datetime(2015, 10, 1, 1)),
        (datetime(2015, 9, 30, 1), datetime(2015, 10, 1, 1), datetime(2015, 11, 1, 1))
    ]
    
    all_metrics: List[Dict[str, Any]] = []
    
    for i, (train_end, infer_start, infer_end) in enumerate(CV_WINDOWS):
        version = f"v_{i+1}_{train_end.strftime('%Y%m%d')}"
        
        # 1. Prepare Data Split
        X_train, y_train, X_infer, y_infer = prepare_time_series_split(
            labeled_features, train_end, infer_start, infer_end, le
        )
        
        # 2. Train, Save Model, and Get Metrics
        metrics = train_and_validate_model(
            X_train, y_train, X_infer, y_infer, version, len(le.classes_), le
        )
        all_metrics.append(metrics)


    return all_metrics

        