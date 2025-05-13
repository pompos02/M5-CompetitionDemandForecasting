import pandas as pd
df_master = pd.read_pickle('/path/to/pklFiles/df_master.pkl')
df_master = df_master.sort_values(['id', 'date'])
import pandas as pd
import numpy as np

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
    'price_momentum',
    'date',

]

# Dropping the specified columns
df_master_without_static = df_master.drop(columns=columns_to_delete)
del df_master

# Dropping the specified columns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

X = df_master_without_static.drop('sales', axis=1)
y = df_master_without_static['sales']


rf = RandomForestRegressor(n_estimators=20, random_state=42,verbose=2,max_samples=0.1,n_jobs=-1)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print("Random Forest Feature Importances:")
print(importances)

# Plotting the importances
importances.plot(kind='bar', figsize=(12,6))
plt.title("Feature Importances from Random Forest")
plt.show()