from tqdm import tqdm
from darts.models import LinearRegressionModel
import pickle


with open("/path/to/pklFiles/low_series_dict.pkl","rb") as file:
    target_series_dict=pickle.load(file)

import numpy as np

train_target_dict = {}
test_target_dict = {}
future_covariates_dict = {}
all_ids = []


for i_id in  (target_series_dict.keys()):
    full_target_series, covariates = target_series_dict[i_id]
    all_ids.append(i_id)
    train_target, test_target = full_target_series[:-28], full_target_series[-28:]

    # --- CAST HERE to float32 ---
    train_target = train_target.astype(np.float32)
    test_target = test_target.astype(np.float32)

    train_target_dict[i_id] = train_target
    test_target_dict[i_id] = test_target
    future_covariates_dict[i_id] = covariates.astype(np.float32)



print(f"Number of series: {len(all_ids)}")
print(f"Example training series length: {train_target_dict['CA_1_FOODS_1_001'].n_timesteps}")
print(f"Example test series length: {test_target_dict['CA_1_FOODS_1_001'].n_timesteps}")
print(f"Example covariates series length: {future_covariates_dict['CA_1_FOODS_1_001'].n_timesteps}")

forecasts_dict = {}
for i_id in tqdm(target_series_dict.keys(), total=len(target_series_dict.keys()), desc="Forecasting series"):
    
    model = LinearRegressionModel(
        lags=[-1,-2,-7],
        output_chunk_length=1,
        use_static_covariates=False
    )
    model.fit(train_target_dict[i_id])

    forecast = model.predict(n=28)
    
    # Store the forecast
    forecasts_dict[i_id] = forecast


file_path = "/path/to/BasicModels/ARIMA/forecasts_dict.pkl"
# Save the dictionary to a .pkl file
with open(file_path, "wb") as file:
    pickle.dump(forecasts_dict, file)

print(f"Dictionary saved as {file_path}")