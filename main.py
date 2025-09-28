from sklearn.preprocessing import LabelEncoder
from consts import DATA_DIR, FIELDS, ERROR_FIELDS, COMPONENT_FIELDS, COMPONENTS, START_DATE, TARGET_COLUMN
from src.model_core.feature_dev import (
    load_pdm_data_simple,
    generate_telemetry_features,
    process_error_data,
    calculate_rolling_error_count,
    process_component_repairs,
    calculate_time_since_replacement, 
    merge_all_features, 
    apply_labels_and_cleanup,
    )

from src.model_core.model_train_func import main_pipeline_training

telemetry, errors, maint, failures, machines = load_pdm_data_simple(DATA_DIR)
telemetry_feat = generate_telemetry_features(telemetry, FIELDS)
error_count = process_error_data(telemetry, errors, ERROR_FIELDS)
error_count_rolling = calculate_rolling_error_count(error_count, ERROR_FIELDS)
comp_rep = process_component_repairs(telemetry, maint, COMPONENT_FIELDS)
comp_rep_tslr = calculate_time_since_replacement(comp_rep, COMPONENTS, START_DATE)
final_feat = merge_all_features(
    telemetry_feat, 
    error_count, 
    comp_rep, 
    machines
)
labeled_features = apply_labels_and_cleanup(final_feat, failures)
le = LabelEncoder()
# Fit the encoder once on the entire set of possible labels from the final dataset.
le.fit(labeled_features[TARGET_COLUMN])

results = main_pipeline_training(labeled_features, le)

