# %%
# %%
import pickle


with open("/path/to/pklFiles/cat_series_dict.pkl","rb") as file:
    cat_series_dict=pickle.load(file)


# %%
from darts.models import LightGBMModel
from darts.dataprocessing.transformers import  StaticCovariatesTransformer
import numpy as np
import sklearn
from darts.metrics import rmse,mae
from darts import TimeSeries, concatenate
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
train_series_cat = {}


test_series_cat = {}
future_covs_cat = {}
#series_cat = {}


MIN_TRAIN_LENGTH = 150# for example


for store_id, cat_dict in cat_series_dict.items():
    train_series_cat[store_id]= {}
    test_series_cat[store_id]= {}
    future_covs_cat[store_id]= {}
    #series_cat[store_id]= {}

    for cat_id, ts_list in cat_dict.items():
        train_series = []
        test_series = []
        future_covariates = []
        for target, covs in ts_list:
            
            train_target = target[:-28].astype(np.float32)
            test_target = target[-28:].astype(np.float32)  
            # series_target = target.astype(np.float32)

            # filtered_covs = covs[["sell_price", "tm_w_end", "tm_dw", "event_name_1_enc"]]

            train_series.append(train_target)
            test_series.append(test_target)
            future_covariates.append(covs.astype(np.float32))
            # series.append(series_target)
        train_series_cat[store_id][cat_id] = train_series
        test_series_cat[store_id][cat_id] = test_series
        future_covs_cat[store_id][cat_id] = future_covariates
        #series_cat[store_id][cat_id] = series

        static_scaler = StaticCovariatesTransformer(transformer_num=None, cols_num=None) #cols_cat=["item_id", "dept_id", "cat_id"])
        train_series_cat[store_id][cat_id] = static_scaler.fit_transform(train_series_cat[store_id][cat_id])
        test_series_cat[store_id][cat_id] = static_scaler.transform(test_series_cat[store_id][cat_id])
        #series_cat[store_id][cat_id] = static_scaler.transform(series_cat[store_id][cat_id])


# %% [markdown]
# # 1st Model

# %%
def train_model(train_series,covariates_series):

    model = LightGBMModel(
    # Choose your lags (number of past timesteps to use). For example:
        lags=28,  # or range(1,29) if you want t-1 to t-28
        output_chunk_length=28,
        lags_future_covariates=list(range(-28,0)),
        # All LightGBM hyperparams:
        boosting_type="gbdt",
        objective="tweedie",
        tweedie_variance_power=1.1,
        metric='mse',
        n_jobs=-1,
        random_state=42,  # "seed" is deprecated in newer LightGBM; use random_state
        learning_rate=0.2,
        bagging_fraction=0.85,
        bagging_freq=1,
        colsample_bytree=0.85,    # or "feature_fraction=0.85" is also valid
        colsample_bynode=0.85,    # or "feature_fraction_bynode=0.85"
        lambda_l1=0.5,
        lambda_l2=0.5,
        verbose = 0,
        categorical_future_covariates = ["event_name_1_enc", "event_type_1_enc", "event_name_2_enc", "event_type_2_enc"],
        categorical_static_covariates= ["item_id", "dept_id"]
    )

    model.fit(series=train_series,
                future_covariates=covariates_series,
                )
    
    return model

# %%
for store_id, cat_dict in tqdm(cat_series_dict.items(), desc="Stores"):
    for cat_id in tqdm(list(cat_dict.keys()), desc=f"Store {store_id} Categories", leave=False):
        model = train_model(train_series=train_series_cat[store_id][cat_id],
                            covariates_series=future_covs_cat[store_id][cat_id])
        model.save(f"/path/to/BasicModels/lgbm/Category_Forecasting/Cat_outchunk/{store_id}_{cat_id}")
