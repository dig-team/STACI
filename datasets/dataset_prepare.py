import pandas as pd
import numpy as np



# Sick constants

SICK_PATH = './sick.data'
SICK_HAS_HEADER = True
SICK_CAT_FEATURES = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant',
                     'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                     'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured', 'TT4_measured',
                     'T4U_measured', 'FTI_measured', 'TBG_measured', 'referral_source']

sick_dataset = pd.read_csv('sick.data', header=0)


sick_dataset = sick_dataset.replace({'target': {r"^neg.*": 0, r"sick.*": 1}}, regex=True)
sick_dataset = sick_dataset.replace({'?': np.nan})
sick_dataset = sick_dataset.drop("TBG", axis=1)
print(sick_dataset)
numerical = []
for col in sick_dataset.columns:
    if col not in SICK_CAT_FEATURES and col != 'target':
        numerical.append(col)
print(numerical)
for col in numerical:
    column = sick_dataset[col].astype('float')
    med = column.median(skipna=True)
    sick_dataset = sick_dataset.fillna(value={col: med})
# sick_dataset = pd.get_dummies(sick_dataset, columns=SICK_CAT_FEATURES)
print(sick_dataset)


sick_dataset.to_csv("sick_ready.csv", index=False)

# Hypothyroid constants

HYPOTHYROID_PATH = './hypothyroid.data'
HYPOTHYROID_HAS_HEADER = True
HYPOTHYROID_CAT_FEATURES = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant',
                            'thyroid_surgery', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor',
                            'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured',
                            'TBG_measured']


sick_dataset = pd.read_csv('hypothyroid.data', header=0)

sick_dataset = sick_dataset.replace({'?': np.nan})
print(sick_dataset)
numerical = []
for col in sick_dataset.columns:
    if col not in HYPOTHYROID_CAT_FEATURES and col != 'target':
        numerical.append(col)
print(numerical)
for col in numerical:
    column = sick_dataset[col].astype('float')
    med = column.median(skipna=True)
    sick_dataset = sick_dataset.fillna(value={col: med})
# sick_dataset = pd.get_dummies(sick_dataset, columns=HYPOTHYROID_CAT_FEATURES)

sick_dataset.to_csv("hypothyroid_ready.csv", index=False)

