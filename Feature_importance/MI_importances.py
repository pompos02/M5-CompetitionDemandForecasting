import pandas as pd
df_master = pd.read_pickle('/path/to/pklFiles/df_master.pkl')
df_master = df_master.sort_values(['id', 'date'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Assume df_master is your DataFrame and 'sales' is the target variable.
smoothing = 10  # You can adjust this parameter as needed

# Compute the global mean of the target variable
global_mean = df_master['sales'].mean()

# List of columns to target encode
columns_to_encode = ['event_name_1','event_type_1', 'event_name_2', 'event_type_2']

# Loop through each column and compute the target encoding
for col in columns_to_encode:
    # Aggregate target statistics for each category in the column
    agg = df_master.groupby(col)['sales'].agg(['mean', 'count'])
    
    # Compute the smoothed target encoding
    agg['target_enc'] = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
    
    # Map the smoothed target encoding back to the original DataFrame
    new_column_name = col + '_enc'
    df_master[new_column_name] = df_master[col].map(agg['target_enc'])
    
    # (Optional) Print out the first few rows for inspection
    print(f"Encoding for {col}:")

    # List of columns to be deleted
columns_to_delete = [
    'event_name_1', 
    'event_type_1', 
    'event_name_2', 
    'event_type_2'
]

# Dropping the specified columns
df_master = df_master.drop(columns=columns_to_delete)

# List of columns to be deleted
columns_to_delete = [
    'id',
    'item_id', 
    'dept_id', 
    'cat_id', 
    'store_id',
    'state_id',
    'date',
    'price_momentum',
    'event_name_1_enc', 
    'event_type_1_enc',
    'event_name_2_enc',
    'event_type_2_enc' 

]

# Dropping the specified columns
df_master_without_static = df_master.drop(columns=columns_to_delete)
del df_master
print(df_master_without_static.columns)

# Sample 10,000 rows from the dataset for MI calculation (adjust n as needed)
df_sample = df_master_without_static.sample(n=2000000, random_state=42)

X_sample = df_sample.drop('sales', axis=1)
y_sample = df_sample['sales']

from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X_sample, y_sample, n_jobs=-1)
mi_scores_series = pd.Series(mi_scores, index=X_sample.columns).sort_values(ascending=False)
print("Mutual Information Scores (on sample):")
print(mi_scores_series)

# Plot the MI scores using a horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=mi_scores_series.values, y=mi_scores_series.index, orient='h')
plt.title('Mutual Information Scores')
plt.xlabel('MI Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()