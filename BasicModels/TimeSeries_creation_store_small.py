import pandas as pd
import numpy as np

df_master = pd.read_pickle('/path/to/pklFiles/df_master.pkl')
df_master = df_master.sort_values(['id', 'date'])

from darts import TimeSeries

from tqdm import tqdm


# Wrap the groupby iterator with tqdm
store_series_dict = {}

for store_id, store_df in tqdm(df_master.groupby("store_id", sort=False, observed=False), 
                                 total=df_master['store_id'].nunique()):


    product_series_list = []
    
    for product_id, product_df in store_df.groupby("item_id", sort=False, observed=False):
        product_df = product_df.sort_values("date")
        
        # --- Target Series (Sales) ---
        target_series = TimeSeries.from_dataframe(
            df=product_df,
            time_col="date",
            value_cols="sales",
            freq="D"
        )
        
        # --- Static Covariates ---
        # Only keep "item_id", "dept_id", "price_mean"
        static_covs = product_df[["item_id", "dept_id", "price_mean"]].iloc[0].to_dict()
        target_series = target_series.with_static_covariates(pd.Series(static_covs))
        
        # --- Future Covariates ---
        # Only keep "sell_price", "item_nunique", "price_momentum_m", "tm_dw", "tm_w_end"
        covariates_df = product_df[[
            "date",  
            "sell_price", 
            "item_nunique", 
            "price_momentum_m",
            "tm_dw",
            "tm_w_end"
        ]]
        
        covariate_series = TimeSeries.from_dataframe(
            df=covariates_df,
            time_col="date",
            value_cols=covariates_df.columns.drop("date"),
            freq="D"
        )
        
        product_series_list.append((target_series, covariate_series))
    
    store_series_dict[store_id] = product_series_list

import pickle

# Specify the file path where you want to save the dictionary
file_path = "/path/to/pklFiles/store_series_dict_small.pkl"
# Save the dictionary to a .pkl file
with open(file_path, "wb") as file:
    pickle.dump(store_series_dict, file)

print(f"Dictionary saved as {file_path}")
