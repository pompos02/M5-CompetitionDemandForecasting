import pickle
from darts.dataprocessing.transformers import  StaticCovariatesTransformer
import numpy as np
from darts import TimeSeries, concatenate

from darts.models import LightGBMModel
from tqdm import tqdm


import pickle


with open("/path/to/pklFiles/store_series_dict.pkl","rb") as file:
    store_series_dict=pickle.load(file)

train_series_store = {}
test_series_store = {}
future_covs_store = {}

store_static_scalers = {}


for store_id, product_series_list in store_series_dict.items():
    # Preprocess data 
    # Split into train/val (last 28 days for validation)
    train_series = []
    val_series = []
    test_series = []
    forecast_series = []
    future_covariates = []
    series = []

    for target, covs in product_series_list:

        train_target = target[:-28].astype(np.float32)
        test_target = target[-28:].astype(np.float32)  

        train_series.append(train_target)
        test_series.append(test_target)
        future_covariates.append(covs.astype(np.float32))

    train_series_store[store_id] = train_series
    test_series_store[store_id] = test_series
    future_covs_store[store_id] = future_covariates

    static_scaler = StaticCovariatesTransformer(transformer_num=None, cols_num=None) #cols_cat=["item_id", "dept_id", "cat_id"])
    store_static_scalers[store_id] = static_scaler.fit(train_series)



def train_model_store(train_series, covariates_series):

    model = LightGBMModel(
    # Choose your lags (number of past timesteps to use). For example:
        lags=28,  # or range(1,29) if you want t-1 to t-28
        #lags_past_covariates=28,
        lags_future_covariates=list(range(-28,0)),
        output_chunk_length=28,
        # All LightGBM hyperparams:
        boosting_type="gbdt",
        objective="tweedie",
        tweedie_variance_power=1.1,
        metric='mse',
        n_jobs=4,
        random_state=42,  # "seed" is deprecated in newer LightGBM; use random_state
        learning_rate=0.2,
        bagging_fraction=0.85,
        bagging_freq=1,
        colsample_bytree=0.85,    # or "feature_fraction=0.85" is also valid
        colsample_bynode=0.85,    # or "feature_fraction_bynode=0.85"
        lambda_l1=0.5,
        lambda_l2=0.5,
        verbose = 10,
        use_static_covariates =True,
        categorical_future_covariates = [#"tm_d","tm_m","tm_y", "tm_w_end", "tm_dw", "tm_wm",
                                    "event_name_1_enc", "event_type_1_enc", "event_name_2_enc", "event_type_2_enc"],
        categorical_static_covariates= ["item_id", "dept_id", "cat_id"]
    )

    model.fit(series=train_series,
            future_covariates=covariates_series,
    )
    
    return model

for store_id, _ in tqdm(store_series_dict.items()):

    train_series = store_static_scalers[store_id].transform(train_series_store[store_id])

    model = train_model_store(train_series, future_covs_store[store_id])

    model.save(f"/path/to/BasicModels/lgbm/Store_Forecasting/store_models/model_{store_id}")

