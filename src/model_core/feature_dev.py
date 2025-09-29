import pandas as pd
import numpy as np


import xgboost as xgb # <--- New Import!
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


def load_pdm_data_simple(data_directory):
    """
    Loads all required CSV files for the Predictive Maintenance (PdM) project.

    Args:
        data_directory (str): The directory path where the CSV files are located.

    Returns:
        tuple: A tuple containing the five DataFrames: 
               (telemetry, errors, maint, failures, machines)
    """
    
    # Load each file directly using a clear path construction
    telemetry = pd.read_csv(os.path.join(data_directory, 'PdM_telemetry.csv'))
    errors = pd.read_csv(os.path.join(data_directory, 'PdM_errors.csv'))
    maint = pd.read_csv(os.path.join(data_directory, 'PdM_maint.csv'))
    failures = pd.read_csv(os.path.join(data_directory, 'PdM_failures.csv'))
    machines = pd.read_csv(os.path.join(data_directory, 'PdM_machines.csv'))
    
    return telemetry, errors, maint, failures, machines


def _calculate_single_feature_set(df, fields, agg_func, suffix):
    """
    Core function to calculate a single set of 24-period rolling features (mean or std).
    This function is now marked with an underscore (_) as an internal helper.
    """
    
    def calculate_rolling_feature(col):
        # Apply data transformation pipeline: pivot -> resample -> unstack -> rolling -> agg
        feature_series = (
            pd.pivot_table(df, index='datetime', columns='machineID', values=col)
            .resample('3H', closed='left', label='right')
            .first()
            .unstack()
            .rolling(window=24, center=False)
            .agg(agg_func)
        )
        return feature_series

    # Apply the logic to all fields and concatenate
    temp_dfs = [calculate_rolling_feature(col) for col in fields]
    
    result_df = pd.concat(temp_dfs, axis=1)
    
    # Rename columns and reset index
    new_cols = [f'{i}{suffix}' for i in fields]
    result_df.columns = new_cols
    result_df.reset_index(inplace=True)
    
    # Filter out initial NaN values from the rolling window 
    filter_col = new_cols[0]
    result_df = result_df.loc[result_df[filter_col].notnull()]
    
    return result_df

# New top-level function to handle the full feature engineering pipeline
def generate_telemetry_features(telemetry_df, fields, verbose=True):
    """
    Generates, combines, cleans, and describes the 24-period rolling mean and 
    standard deviation features.

    Args:
        telemetry_df (pd.DataFrame): The input telemetry DataFrame.
        fields (list): List of column names to process.
        verbose (bool): If True, prints the descriptive statistics.

    Returns:
        pd.DataFrame: The combined and cleaned feature DataFrame.
    """
    
    # 1. Calculate Mean features
    telemetry_mean_24h = _calculate_single_feature_set(
        telemetry_df, fields, 'mean', 'mean_24h'
    )

    # 2. Calculate Standard Deviation features
    telemetry_sd_24h = _calculate_single_feature_set(
        telemetry_df, fields, 'std', 'sd_24h'
    )
    
    # The first two columns are 'datetime' and 'machineID', so we select columns 2:6 (the 4 features)
    FEATURE_COLS_SLICE = slice(2, 6)

    # 3. Combine Mean and Standard Deviation features
    telemetry_feat = pd.concat([
        telemetry_mean_24h,
        telemetry_sd_24h.iloc[:, FEATURE_COLS_SLICE]
    ], axis=1).dropna()
    
    # 4. Analyze the new feature set
    if verbose:
        print("\n--- Combined Feature Set Description ---")
        print(telemetry_feat.describe())
        
    return telemetry_feat

def process_error_data(telemetry_df, errors_df, error_fields, fill_value=0.0):
    """
    Processes raw error data to create a time-aligned, one-hot encoded error count
    DataFrame, merged with the main telemetry timeline.

    Args:
        telemetry_df (pd.DataFrame): The main telemetry DataFrame (used for its timeline).
        errors_df (pd.DataFrame): The raw errors DataFrame with 'datetime', 'machineID', 'errorID'.
        error_fields (list): The desired column names for the one-hot encoded errors.
        fill_value (float): The value to use for periods with no errors.

    Returns:
        pd.DataFrame: DataFrame with 'datetime', 'machineID', and time-aligned error columns.
    """
    
    # --- Step 1: One-Hot Encode and Aggregate Error Counts ---
    # The errors DataFrame should already have 'datetime' and 'machineID'
    error_count_processed = (
        errors_df
        .set_index('datetime')
        # One-hot encode the 'errorID' column and combine with the rest of the index/columns
        .pipe(pd.get_dummies, columns=['errorID'])
        .reset_index()
    )

    # --- Step 2: Clean up Column Names ---
    # Assuming 'machineID' and all 'errorID_' dummies are the desired output columns.
    # The first two columns should be 'datetime' and 'machineID'
    new_columns = ['datetime', 'machineID'] + error_fields
    error_count_processed.columns = new_columns

    # --- Step 3: Align Error Data with Telemetry Timeline (Outer Join for the timeline) ---
    # Use the telemetry timeline as the left side to ensure every machineID/datetime
    # combination from the main data exists.
    # We only need the machine and time columns from telemetry for alignment.
    aligned_errors = (
        telemetry_df[['datetime', 'machineID']]
        .merge(
            error_count_processed, 
            on=['machineID', 'datetime'], 
            how='left'
        )
        # --- Step 4: Fill Missing Values ---
        # Fill all NaN values (where no error occurred) with the specified fill_value
        .fillna(fill_value)
    )

    return aligned_errors

ERROR_FIELDS = [f'error{i}' for i in range(1, 6)]

def calculate_rolling_error_count(error_df, fields, window=24, freq='3H'):
    """
    Calculates the 24-period rolling sum of errors for each error type and machine.

    Args:
        error_df (pd.DataFrame): DataFrame with error counts, 'datetime', and 'machineID'.
        fields (list): List of error column names to process (e.g., ['error1', 'error2']).
        window (int): The rolling window size (in 3-hour periods, e.g., 24 periods = 3 days).
        freq (str): The resampling frequency (e.g., '3H').

    Returns:
        pd.DataFrame: DataFrame with 'datetime', 'machineID', and rolling error sum columns.
    """
    
    temp_dfs = []

    for col in fields:
        # Core pipeline: Pivot -> Resample -> Unstack -> Rolling Sum
        rolling_sum_series = (
            pd.pivot_table(error_df,
                           index='datetime',
                           columns='machineID',
                           values=col)
            .resample(freq, closed='left', label='right')
            .first()
            .unstack()
            .rolling(window=window, center=False)
            .sum() # Key difference from mean/std: using sum()
        )
        temp_dfs.append(rolling_sum_series)

    # Combine all error rolling sums
    result_df = pd.concat(temp_dfs, axis=1)

    # Rename and clean up
    result_df.columns = [f'{i}count' for i in fields]
    result_df.reset_index(inplace=True)
    
    # Drop rows that are NaN due to the start of the rolling window
    result_df = result_df.dropna() 

    return result_df

def process_component_repairs(telemetry_df, maint_df, component_fields, fill_value=0):
    """
    Processes raw maintenance data, aligns it to the telemetry timeline,
    and returns a time-aligned, one-hot encoded DataFrame of component replacements.

    Args:
        telemetry_df (pd.DataFrame): The main telemetry DataFrame (used for its timeline).
        maint_df (pd.DataFrame): The raw maintenance/repair DataFrame.
        component_fields (list): The desired column names for the one-hot encoded repairs.
        fill_value (int): The value to use for periods with no repair (0 is typical).

    Returns:
        pd.DataFrame: DataFrame with time-aligned component repair flags.
    """
    
    # 1. One-Hot Encode and rename the repair data
    comp_rep_processed = (
        maint_df
        .set_index('datetime')
        # One-hot encode the 'comp' column (assuming a 'comp' column in maint)
        # Note: We must infer the column to get dummies from. Assuming 'comp' is the column name.
        # If your column is named 'compID' or 'component', adjust the .pipe() call.
        .pipe(pd.get_dummies, columns=['comp']) # Assuming 'comp' is the column in maint
        .reset_index()
    )
    
    # Align the new columns to the required names. 
    # The first two columns are 'datetime' and 'machineID'.
    new_columns = ['datetime', 'machineID'] + component_fields
    comp_rep_processed.columns = new_columns

    # 2. Merge the processed repairs onto the master timeline (telemetry_df)
    # The merge ensures we have an entry for every (datetime, machineID) pair.
    # An 'outer' merge is technically correct for joining two sparse time series, 
    # but since the goal is to align to the *telemetry* timeline, a 'left' merge 
    # using telemetry as the left base is often cleaner. The original code used 
    # 'outer', so we'll stick with that for functional equivalence.
    comp_rep_final = (
        telemetry_df[['datetime', 'machineID']]
        .merge(
            comp_rep_processed,
            on=['datetime', 'machineID'],
            how='outer'  # Use outer to ensure all maintenance events are captured
        )
        # 3. Cleanup: Fill NaN values (periods with no repair) and sort
        .fillna(fill_value)
        .sort_values(by=['machineID', 'datetime'])
    )
    
    return comp_rep_final



def calculate_time_since_replacement(df, components, start_date):
    """
    Calculates the 'Time Since Last Replacement' in days for specified components.

    This function performs three main steps:
    1. Transforms repair flags (1s) into the actual repair date ('datetime').
    2. Forward-fills the repair dates, propagating the last replacement date forward.
    3. Calculates the difference between the current date and the last repair date in days.

    Args:
        df (pd.DataFrame): The component repair DataFrame.
        components (list): List of component columns to process.
        start_date (str): The date to filter the final results (removes initial data).

    Returns:
        pd.DataFrame: The DataFrame with the component columns replaced by 
                      "Time Since Last Replacement" in days.
    """
    
    # Work on a copy to avoid SettingWithCopyWarning and modify the DataFrame in place
    df_transformed = df.copy()

    # --- Step 1: Prepare Repair Dates and Forward-Fill ---
    for comp in components:
        # a) Replace '0' flags with NaN to isolate the '1' flags (indicating a repair)
        #    Rows where comp == 1 will remain as 1 for now.
        df_transformed.loc[df_transformed[comp] < 1, comp] = np.nan
        
        # b) Replace the '1' flag with the actual repair date from the 'datetime' column
        #    We are using loc[] to target the rows where the original value was not NaN (i.e., was 1)
        #    Note: The original code used a complex boolean filter; a cleaner way is often just isna()
        is_not_nan = df_transformed[comp].notna() # True for rows where repair happened (was 1)
        df_transformed.loc[is_not_nan, comp] = df_transformed.loc[is_not_nan, 'datetime']
        
        # c) Forward-fill the repair date
        #    This propagates the last repair date down to subsequent rows.
        df_transformed[comp] = df_transformed[comp].fillna(method='ffill')
    
    # --- Step 2: Filter out initial data before a clean slate ---
    # This ensures the first non-NaN date is valid after ffill.
    df_transformed = df_transformed.loc[df_transformed['datetime'] > pd.to_datetime(start_date)].copy()


    # --- Step 3: Calculate Time Since Replacement in Days ---
    for comp in components:
        # Calculate time difference and convert the timedelta result to a floating-point number of days
        df_transformed[comp] = (
            df_transformed["datetime"] - pd.to_datetime(df_transformed[comp])
        ) / np.timedelta64(1, "D")
        
    return df_transformed


def merge_all_features(telemetry_feat, error_count, comp_rep, machines):
    """
    Performs a sequence of left merges to combine all engineered features and static
    machine data onto the telemetry timeline.

    Args:
        telemetry_feat (pd.DataFrame): Rolling mean/std features (primary timeline).
        error_count (pd.DataFrame): Rolling error count features.
        comp_rep (pd.DataFrame): Time since last repair features.
        machines (pd.DataFrame): Static machine properties.

    Returns:
        pd.DataFrame: The final, merged feature set.
    """
    
    # Start the feature pipeline with the main time-aligned data
    final_feat = telemetry_feat.copy()
    
    # Use method chaining with .merge() to build the final DataFrame
    final_feat = (
        final_feat
        # 1. Merge rolling error counts onto the telemetry timeline
        .merge(
            error_count, 
            on=['datetime', 'machineID'], 
            how='left'
        )
        # 2. Merge time since last repair features
        .merge(
            comp_rep, 
            on=['datetime', 'machineID'], 
            how='left'
        )
        # 3. Merge static machine features (only requires 'machineID')
        .merge(
            machines, 
            on=['machineID'], 
            how='left'
        )
    )
    
    return final_feat

def apply_labels_and_cleanup(features_df, failures_df, bfill_limit=7, final_fill_value='none'):
    """
    Merges the feature set with the failure labels and performs final imputation steps.

    Args:
        features_df (pd.DataFrame): The DataFrame containing all engineered features.
        failures_df (pd.DataFrame): The DataFrame containing failure labels ('datetime', 'machineID', 'failure_type').
        bfill_limit (int): The limit for backward fill (bfill), typically used to propagate a 
                           failure label backward in time for a few days.
        final_fill_value (str): The value to use for any remaining NaN values.

    Returns:
        pd.DataFrame: The final, labeled, and cleaned DataFrame ready for modeling.
    """

    # 1. Merge the failure labels onto the feature set
    labeled_features = (
        features_df
        .merge(
            failures_df, 
            on=['datetime', 'machineID'], 
            how='left'
        )
        # 2. Backward Fill (bfill) with a limit
        # This step is critical for time-series labeling, often used to label the
        # period *leading up to* a failure event.
        .fillna(
            method='bfill', 
            limit=bfill_limit
        )
        # 3. Final Imputation of remaining NaN values
        # All remaining NaNs (periods where no failure occurred or was propagated) 
        # are filled with a specified value (e.g., 'none').
        .fillna(final_fill_value)
    )
    
    return labeled_features



def prepare_data_split(df, last_train_date, first_test_date, exclude_cols=['datetime', 'machineID', 'failure']):
    """
    Splits the main DataFrame into separate, processed X and y training and testing sets.
    """
    
    # --- Split Data ---
    train_data = df.loc[df['datetime'] < last_train_date]
    test_data = df.loc[(df['datetime'] >= first_test_date) & (df['datetime'] < last_train_date.replace(month=last_train_date.month + 1, day=1))] # Simplified test period definition

    # --- Preprocessing (Label Encoding for Target) ---
    # Convert 'failure' labels (like 'none', 'comp1', etc.) to integers (0, 1, 2, ...)
    le = LabelEncoder()
    le.fit(df['failure']) 

    # --- Feature Engineering and Separation ---
    
    # Process features (X) for training and testing
    train_X = train_data.drop(exclude_cols, axis=1).pipe(pd.get_dummies)
    test_X = test_data.drop(exclude_cols, axis=1).pipe(pd.get_dummies)
    
    # Align columns between train and test (critical step after get_dummies)
    # This ensures both X sets have the exact same columns, filling missing categories with 0
    common_cols = list(set(train_X.columns) & set(test_X.columns))
    train_X, test_X = train_X.align(test_X, join='outer', axis=1, fill_value=0)
    
    # Process labels (y)
    train_y = le.transform(train_data['failure'])
    test_y = le.transform(test_data['failure'])
    
    # Return everything needed for the loop
    return train_X, train_y, test_X, test_y, pd.concat([train_X, pd.Series(train_y, name='failure')], axis=1) # Return the df for the train_dfs list
