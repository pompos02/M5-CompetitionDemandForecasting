import pandas as pd
import numpy as np

df_master = pd.read_pickle('/path/to/pklFiles/df_master.pkl')
df_master = df_master.sort_values(['id', 'date'])

smoothing = 10  # Adjust as needed

# Compute the global mean of the target variable
global_mean = df_master['sales'].mean()

# List of columns to target encode
columns_to_encode = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
encoded_columns = []  # To store the names of the new columns

# Loop through each column and compute the target encoding
for col in columns_to_encode:
    # Aggregate target statistics for each category in the column
    agg = df_master.groupby(col)['sales'].agg(['mean', 'count'])
    
    # Compute the smoothed target encoding
    agg['target_enc'] = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
    
    # Map the smoothed target encoding back to the original DataFrame
    new_column_name = col + '_enc'
    df_master[new_column_name] = df_master[col].map(agg['target_enc'])
    encoded_columns.append(new_column_name)

# Convert the encoded columns to float first
df_master[encoded_columns] = df_master[encoded_columns].astype(np.float32)

# Now replace any NaN values with 0
df_master[encoded_columns] = df_master[encoded_columns].fillna(0)
    
# List of columns to be deleted
columns_to_delete = [
    'event_name_1', 
    'event_type_1', 
    'event_name_2', 
    'event_type_2'
]

# Dropping the specified columns
df_master = df_master.drop(columns=columns_to_delete)












from tqdm import tqdm
from darts import TimeSeries
# Dictionaries for storing the series
target_series_dict = {}

# Group by 'id' to isolate each item+store
for i_id, product_df in tqdm (df_master.groupby('id', sort=False, observed=False), total=df_master['id'].nunique()):
    # Sort within the group by date (just to be safe)
    product_df = product_df.sort_values('date')
    
    state_id = product_df['state_id'].iloc[0]
    snap_col = f"snap_{state_id}"

    # --- TARGET SERIES (sales) ---
    target_series = TimeSeries.from_dataframe(
        df=product_df,
        time_col='date',       # The column that holds your dates
        value_cols='sales',    # The column(s) you want as your target
        freq='D'               # Daily frequency (adjust if your data is weekly, etc.)
    )
    
    # --- Static Covariates ---
    static_covs = product_df[["item_id", "dept_id", "cat_id", "release", "price_max", "price_min","price_std",
                        "price_mean", "price_norm", "price_nunique", "item_nunique", 
                        "price_momentum_m","price_momentum_y"]].iloc[0].to_dict()
    target_series = target_series.with_static_covariates(pd.Series(static_covs))

    covariates_df = product_df[[
            "date",  
            "sell_price", 
            snap_col,
            "tm_d","tm_m","tm_y", "tm_w_end", "tm_dw", "tm_wm",
            "event_name_1_enc", 
            "event_type_1_enc", 
            "event_name_2_enc", 
            "event_type_2_enc"
        ]]
        
    covariate_series = TimeSeries.from_dataframe(
        df=covariates_df,
        time_col="date",
        value_cols=covariates_df.columns.drop("date"),  # Exclude "date"
        freq="D"
    )

    # Store in the dictionaries
    target_series_dict[i_id] = ((target_series, covariate_series))

print(f"Created {len(target_series_dict)} target series")

import pickle

# Specify the file path where you want to save the dictionary
file_path = "/path/to/pklFiles/low_series_dict.pkl"

# Save the dictionary to a .pkl file
with open(file_path, "wb") as file:
    pickle.dump(target_series_dict, file)