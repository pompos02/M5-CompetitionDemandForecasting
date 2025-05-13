import pickle
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
import numpy as np
import sklearn

early_stop_callback = EarlyStopping(
    monitor='val_loss',  # or another appropriate metric
    min_delta=0.00,
    patience=5,         # number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='min'
)

with open("/path/to/pklFiles/store_series_dict.pkl","rb") as file:
    store_series_dict=pickle.load(file)


def train_model_for_store(train_series, future_covariate, val_series):
    
    model = RNNModel(
    model="LSTM",
    hidden_dim=32,
    n_rnn_layers=3,
    dropout=0.2,
    batch_size=512,
    n_epochs=30,
    random_state=42,
    training_length=35,
    force_reset =True,
    input_chunk_length=28,
    optimizer_kwargs = {'lr': 1e-3},
    pl_trainer_kwargs = {
        "accelerator": "gpu", 'callbacks': [early_stop_callback]
    }
)

    # Train with future_covariates
    model.fit(
        series=train_series,
        future_covariates=future_covariate,
        val_series=val_series,
        val_future_covariates=future_covariate,
        verbose = True
    )
    return model



store_models = {}
store_scalers = {}

for store_id, product_series_list in store_series_dict.items():
    # Preprocess data 
    # Split into train/val (last 28 days for validation)
    train_series = []
    val_series = []
    future_covariates = []
    for target, covs in product_series_list:
        train_target = target[:-64].astype(np.float32)
        val_target = target[-64:].astype(np.float32)  
        series = target.astype(np.float32)
        #train_covs = covs[:-28]
        #val_covs = covs[-28:]
        train_series.append(train_target)
        val_series.append(val_target)
        future_covariates.append(covs.astype(np.float32))

    # Scale target and covariates
    scaler_target = Scaler()
    scaler_covs = Scaler()
    static_scaler = StaticCovariatesTransformer() # for numeric static covariates
                                #transformer_cat=sklearn.preprocessing.OrdinalEncoder())

    
    train_series_scaled = scaler_target.fit_transform(train_series)
    val_series_scaled = scaler_target.transform(val_series)
    future_covariates_scaled = scaler_covs.fit_transform(future_covariates)
    # Store scalers for inverse transformation later
    store_scalers[store_id] = {
        'target': scaler_target,
        'covariates': scaler_covs,
    }
    # Train the model
    model = train_model_for_store(train_series_scaled, future_covariates_scaled, val_series_scaled)
    print(f"Successfully trained model for {store_id}")
    
    model.save(f"/path/to/BasicModels/LSTM/models_saved/model_{store_id}")
    break

file_path = "/path/to/BasicModels/LSTM/pkl_files/store_scalers.pkl"

# Save the dictionary to a .pkl file
with open(file_path, "wb") as file:
    pickle.dump(store_scalers, file)

print(f"scalers dictionary saved as {file_path}")