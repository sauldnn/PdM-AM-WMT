import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report

from consts import (
    FIELDS, ERROR_FIELDS, COMPONENT_FIELDS, COMPONENTS,
    START_DATE, TARGET_COLUMN
)
from src.model_core.feature_dev import (
    generate_telemetry_features,
    process_error_data,
    calculate_rolling_error_count,
    process_component_repairs,
    calculate_time_since_replacement, 
    merge_all_features, 
    apply_labels_and_cleanup,
)
from src.model_core.model_train_func import main_pipeline_training


def preprocessing_fn(inputs, schema):
    """Feature engineering entrypoint for TFX Transform."""
    return inputs


def run_fn(fn_args):
    """Custom Trainer for TFX pipeline."""
    # 1. Load training data from fn_args
    train_data = pd.read_csv(fn_args.train_files[0])
    eval_data = pd.read_csv(fn_args.eval_files[0])

    # 2. Feature Engineering
    telemetry_feat = generate_telemetry_features(train_data, FIELDS)
    error_count = process_error_data(train_data, eval_data, ERROR_FIELDS)
    error_count_rolling = calculate_rolling_error_count(error_count, ERROR_FIELDS)
    comp_rep = process_component_repairs(train_data, eval_data, COMPONENT_FIELDS)
    comp_rep_tslr = calculate_time_since_replacement(comp_rep, COMPONENTS, START_DATE)

    final_feat = merge_all_features(telemetry_feat, error_count, comp_rep, eval_data)
    labeled_features = apply_labels_and_cleanup(final_feat, eval_data)

    le = LabelEncoder()
    le.fit(labeled_features[TARGET_COLUMN])

    # 3. Train Model
    results = main_pipeline_training(labeled_features, le)
    print("Metrics:", results)

    # 4. Save model
    model = xgb.XGBClassifier()
    X = labeled_features.drop(columns=["datetime", "machineID", TARGET_COLUMN])
    y = le.transform(labeled_features[TARGET_COLUMN])
    model.fit(X, y)

    joblib.dump(model, fn_args.serving_model_dir + "/model.joblib")
