from darts.dataprocessing.transformers import Scaler
import numpy as np
from darts.models import RNNModel
import torch
from pytorch_lightning.callbacks import EarlyStopping
import pickle
import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
with open("/path/to/pklFiles/low_series_dict_small.pkl","rb") as file:
        
    target_series_dict=pickle.load(file)

early_stop_callback = EarlyStopping(
    monitor='val_loss',  # or another appropriate metric
    min_delta=0.000,
    patience=7,         # number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='min'
)

import numpy as np

train_target_dict = {}
val_target_dict = {}
test_target_dict = {}
future_covariates_dict = {}
all_ids = []
item_scalers = {}



for i_id in  (target_series_dict.keys()):
    full_target_series, covariates = target_series_dict[i_id]

    train_target, val_target = full_target_series[:-128], full_target_series[-128:-28]

    all_ids.append(i_id)
    # --- CAST HERE to float32 ---
    train_target = train_target.astype(np.float32)
    val_target = val_target.astype(np.float32)

    scaler_target = Scaler()
    scaler_covs = Scaler()
    item_scalers[i_id] = {
        'target': scaler_target,
        'covariates': scaler_covs,
    }

    train_target_scaled = item_scalers[i_id]['target'].fit_transform(train_target)
    val_target_scaled = item_scalers[i_id]['target'].transform(val_target)
    covariates_scaled = item_scalers[i_id]['covariates'].fit_transform(covariates)

    train_target_dict[i_id] = train_target_scaled.astype(np.float32)
    val_target_dict[i_id] = val_target_scaled.astype(np.float32)
    future_covariates_dict[i_id] = covariates_scaled.astype(np.float32)


print(f"Number of series: {len(all_ids)}")
print(f"Example training series length: {train_target_dict[i_id].n_timesteps}")
print(f"Example validation series length: {val_target_dict[i_id].n_timesteps}")
print(f"Example covariates series length: {future_covariates_dict[i_id].n_timesteps}")

def train_model(train_series,validation_series,covariates_series,item_id):
    model = RNNModel(
            input_chunk_length=28,
            training_length=35,
            model='LSTM',
            hidden_dim=128,
            n_rnn_layers=4,
            dropout=0.3,
            batch_size=512,
            n_epochs=50,
            loss_fn=torch.nn.MSELoss(),
            random_state=42,
            force_reset=True,
            save_checkpoints=True,
            model_name="global_low_small",
            work_dir="/path/to/BasicModels/LSTM",
            optimizer_kwargs={'lr': 1e-4},
            pl_trainer_kwargs={
                "accelerator": "gpu",
                'callbacks': [early_stop_callback]
            }
    )

    model.fit(
        series=train_series,
        future_covariates=covariates_series,
        val_series=validation_series,
        val_future_covariates=covariates_series,
        verbose = True
    )
    return model

train_series_list = [train_target_dict[i_id] for i_id in all_ids]
val_series_list = [val_target_dict[i_id] for i_id in all_ids]
covariates_series_list = [future_covariates_dict[i_id] for i_id in all_ids]


model = train_model(train_series_list,val_series_list,covariates_series_list,i_id)


file_path = "/path/to/BasicModels/LSTM/pkl_files/item_scalers.pkl"
with open(file_path, "wb") as file:
    pickle.dump(item_scalers, file)
print(f"item_scalers dictionary saved as {file_path}")
