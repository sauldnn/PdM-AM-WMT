import pandas as pd
DATA_DIR = '../input/microsoft-azure-predictive-maintenance'
FIELDS = ['volt', 'rotate', 'pressure', 'vibration']
ERROR_FIELDS = [f'error{i}' for i in range(1, 6)]
COMPONENT_FIELDS = ['comp1', 'comp2', 'comp3', 'comp4']
COMPONENTS = ['comp1', 'comp2', 'comp3', 'comp4']
START_DATE = '2015-01-01'
THRESHOLD_DATES = [
    (pd.to_datetime('2015-07-31 01:00:00'), pd.to_datetime('2015-08-01 01:00:00')),
    (pd.to_datetime('2015-08-31 01:00:00'), pd.to_datetime('2015-09-01 01:00:00')),
    (pd.to_datetime('2015-09-30 01:00:00'), pd.to_datetime('2015-10-01 01:00:00'))
]
TARGET_COLUMN = "failure"