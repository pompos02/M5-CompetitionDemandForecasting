import pandas as pd
import numpy as np

df_master = pd.read_pickle('/path/to/pklFiles/df_master.pkl')
df_master = df_master.sort_values(['id', 'date'])



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
    

    covariates_df = product_df[[
            "date",  
            "sell_price", "price_mean","item_nunique", "price_momentum_m",
            "tm_w_end", "tm_dw",
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
file_path = "/path/to/pklFiles/low_series_dict_small.pkl"

# Save the dictionary to a .pkl file
with open(file_path, "wb") as file:
    pickle.dump(target_series_dict, file)