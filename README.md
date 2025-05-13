# Demand Forecasting in Retail (Πρόβλεψη Ζήτησης στην Λιανική)

This project, based on the thesis "Demand Forecasting in Retail" by Γιάννης Καραβέλλας, explores various machine learning models for predicting demand in the retail sector. The primary dataset used is the M5-Competition Dataset, which contains sales data from Walmart.

## Table of Contents

*   [Motivation](#motivation)
*   [Dataset](#dataset)
*   [Technical Overview](#technical-overview)
    *   [Core Libraries and Dependencies](#core-libraries-and-dependencies)
    *   [Data Processing Pipeline](#data-processing-pipeline)
    *   [Model Implementation Details](#model-implementation-details)
    *   [Feature Importance Analysis](#feature-importance-analysis)
    *   [Model Evaluation](#model-evaluation)
    *   [Overall Workflow Summary](#overall-workflow-summary)
*   [Project Structure](#project-structure)
*   [Setup and Installation](#setup-and-installation)
*   [Usage](#usage)
*   [Results Summary](#results-summary)
*   [Author](#author)
*   [License](#license)
*   [Acknowledgements](#acknowledgements)

## Motivation

Effective demand forecasting is a critical challenge for retail businesses. It directly influences inventory management, operational costs, and customer satisfaction. While traditional statistical methods have been used for decades, the increasing availability and complexity of data necessitate more sophisticated approaches, with Machine Learning (ML) being a key research direction. This project aims to investigate and compare the performance of various ML models in this domain, providing a detailed account of their implementation and evaluation.

## Dataset

The project utilizes the [M5-Competition Dataset](https://www.kaggle.com/c/m5-forecasting-accuracy). This dataset comprises hierarchical sales time series data from Walmart in the USA. It includes detailed records for thousands of products categorized by department, category, and store, along with exogenous variables such as calendar-related information (holidays, special events like SNAP days) and pricing data.

The raw data files (`calendar.csv`, `sales_train_validation.csv`, `sell_prices.csv`, `sales_test_validation.csv`, `sales_test_evaluation.csv`, and `ExtraFiles/weights_validation.csv`) are expected to be located in the `./data/` directory.

## Technical Overview

This section provides a detailed technical breakdown of the project, including the libraries used, data processing steps, model architectures, and evaluation methodologies.

### Core Libraries and Dependencies
The project relies on a suite of Python libraries for data manipulation, machine learning, and visualization. Key dependencies, as listed in `requirements.txt` and observed from the codebase, include:

*   **Data Manipulation:** `pandas`, `numpy`
*   **Machine Learning (General):** `scikit-learn` (for  performance metrics, and data preprocessing)
*   **Gradient Boosting:** `lightgbm` (the core library for LightGBM models)
*   **Time Series & Deep Learning:** `darts` (central for LSTM and Exponential Smoothing models, as well as general time series manipulation), `torch`, `pytorch_lightning` (underlying frameworks for Darts-based deep learning models, handling training loops and checkpointing)
*   **Statistical Models:** `statsmodels` ( used for statistical analyses)
*   **Visualization:** `matplotlib`, `seaborn`
*   **Utilities:** `tqdm` (for progress bar visualization), `joblib` or `pickle` (for serializing and deserializing Python objects like trained models and processed data).

### Data Processing Pipeline
The data processing pipeline is a multi-stage process designed to transform raw M5 competition data into a format suitable for model training and evaluation:

1.  **Initial Data Loading and Merging (`df_master_creation.ipynb`):**
    *   Raw CSV files (`sales_train_validation.csv`, `calendar.csv`, `sell_prices.csv, pklFiles/grid_part_*.pkl`) are loaded.
    *   These files are merged to create a comprehensive master DataFrame.
    *   Basic data type conversions and initial data cleaning steps are performed.

2.  **Feature Engineering (Primarily in `df_master_creation.ipynb` and `Feature_importance/`):**
    *   A rich set of features is engineered to capture various aspects of the sales data:
        *   **Time-based features:** Lagged sales values (from previous days, weeks), rolling window statistics (mean, standard deviation, min, max of sales over different periods), and date components (day of the week, week of the year, month, year, day of the month).
        *   **Price-related features:** Product price, price changes over time, price relative to average historical prices, and price momentum indicators.
        *   **Event/Calendar features:** Information from `calendar.csv` regarding holidays, special events (e.g., SNAP purchase days) is extracted and incorporated as binary or categorical features.
    *   Categorical features (such as store IDs, item IDs, department IDs) are encoded using techniques like one-hot encoding or label encoding.

3.  **Time Series Creation (`BasicModels/TimeSeries_creation_*.py` scripts):**
    *   Dedicated scripts (`TimeSeries_creation_cat.py` for category level, `TimeSeries_creation_low.py` for item level, `TimeSeries_creation_store.py` for store level) process the master DataFrame.
    *   They transform the tabular data into time series objects compatible with the Darts library or other model input formats.
    *   This typically involves pivoting the data to have time series as rows or columns and creating Darts `TimeSeries` objects for each individual series (e.g., per item-store combination for low-level forecasting, or aggregated series for category/store levels).
    *   The generated time series data is usually saved as dictionaries in `.pkl` files within the `pklFiles/` directory (e.g., `cat_series_dict.pkl`, `low_series_dict.pkl`, `store_series_dict.pkl`).

### Model Implementation Details
The project explores and implements several forecasting models:

#### a. Exponential Smoothing (`BasicModels/ExponentialSmoothing/`)
*   **Notebooks:** `ES_Holts.ipynb`, `ES_Holts_RMSSE.ipynb`, `ExponentialSmoothingTesting.ipynb`.
*   Implements Holt\'s Exponential Smoothing method, a common technique for data with trends.
*   The implementation  uses the Darts library\'s `ExponentialSmoothing` model.
*   The `ES_Holts_RMSSE.ipynb` notebook suggests specific experimentation or tuning focused on the WRMSSE metric.

#### b. LSTM (Long Short-Term Memory) (`BasicModels/LSTM/`)
*   **Notebook:** `LSTM_Forecast_low.ipynb` (focuses on low-level, item-specific forecasting).
*   **Scripts:** `training_low.py`, `training.py` (general training scripts).
*   Utilizes the Darts library\'s `LSTM` model, which is a PyTorch-based recurrent neural network implementation.
*   The training scripts (e.g., `training_low.py`) manage model instantiation, fitting the model to the time series data, and saving training checkpoints (evident from the `darts_logs/global_low_small/checkpoints/` directory structure).
*   The LSTM models use past & feature covariates (such as lagged sales, engineered calendar features) 
*   Hyperparameters for the LSTM architecture (e.g., hidden layer dimensions, number of layers, dropout rates, learning rate for the optimizer) are defined and configured within the training scripts or notebooks.

#### c. Linear Regression (`BasicModels/LinearRegression/`)
*   **Notebook:** `Forecast_Linear.ipynb`.
*   **Script:** `Training.py`.
*   Implements a standard Linear Regression model,  using Darts
*   The `Training.py` script is responsible for fitting the linear regression model to the prepared feature set and saving the trained model.

#### d. LightGBM (`BasicModels/lgbm/`)
*   This model is a significant component of the project, with implementations for different aggregation levels of the sales hierarchy:
    *   **Category Forecasting (`Category_Forecasting/`):** Contains `LGBM_Forecast_cat.ipynb` and `training_cat.py`. Trained models are saved in `Cat_models/` and `Cat_short_models/`.
    *   **Item Forecasting (`Item_Forecasting/`):** Includes `LGBM_Forecast_item.ipynb`. Generated forecasts are stored in files such as `pkl_files/forecasts_dict_low.pkl`.
    *   **Store Forecasting (`Store_Forecasting/`):** Features `LGBM_Forecast_store.ipynb` and `training_store.py`. Trained models are saved, for instance, as `models_store_dict.pkl`. The `ParametersTesting.ipynb` notebook suggests that hyperparameter tuning was performed for these models.
*   Utilizes the `lightgbm` in Darts library for gradient boosted decision trees.
*   Models are trained as regressors to predict future sales quantities.
*   The feature set is critical and includes lagged sales, rolling window statistics, calendar features, price features, and potentially categorical embeddings.
*   Not only a recursive forecasting strategy is employed to generate multi-step ahead predictions, where the model predicts one step at a time and uses its own output as input for subsequent steps.

### Feature Importance Analysis (`Feature_importance/`)
*   **Notebook:** `Dataframe_manipulation.ipynb` ( used for preparing the dataset specifically for feature importance calculation).
*   **Scripts:** `MI_importances.py` (calculates feature importance using Mutual Information techniques) and `RF_importances.py` (uses a Random Forest model, liely `sklearn.ensemble.RandomForestRegressor`, to derive feature importances based on impurity reduction or permutation importance).
*   The output of this analysis includes visualizations, such as `outputs/Feature Importances RF.png`, which help in understanding which engineered features are most predictive for the models.

### Model Evaluation (`RESULTS/`)
*   **Notebook:** `results.ipynb` serves as the central hub for aggregating, analyzing, and visualizing the performance metrics of all trained models.
*   **Metrics:** The primary evaluation metrics used are:
    *   **MAE (Mean Absolute Error)**
    *   **RMSE (Root Mean Squared Error)**
    *   **WRMSSE (Weighted Root Mean Squared Scaled Error)**: This is the official evaluation metric for the M5 competition and accounts for the hierarchical nature of the data and varying sales volumes. Its calculation uses weighting factors from `ExtraFiles/weights_validation.csv`.
*   The `output/` subdirectory within `RESULTS/` contains generated plots, such as histograms and boxplots of MAE and RMSE values, to visually compare model performance.
*   The `RMSE_MAE/` subdirectory stores pickled pandas DataFrames that contain detailed metric scores for each model at different aggregation levels (e.g., `metrics_df_LGBMcat.pkl`, `metrics_df_LSTM.pkl`).

### Overall Workflow Summary
The project follows a structured workflow:

1.  **Data Ingestion & Preprocessing:** Load raw M5 competition data, merge different sources, and perform initial cleaning (primarily in `df_master_creation.ipynb`).
2.  **Feature Engineering:** Create an extensive set of time-based, price-based, and event-based features to capture underlying sales patterns.
3.  **Time Series Generation:** Convert the processed tabular data into specialized time series objects suitable for different forecasting models and aggregation levels (using `TimeSeries_creation_*.py` scripts).
4.  **Model Training:** Train the various forecasting models (Exponential Smoothing, LSTM, Linear Regression, LightGBM) on the prepared time series data. LightGBM models are trained with a hierarchical approach (category, item, store levels).
5.  **Feature Importance Analysis:** Analyze the contribution and importance of the engineered features to model predictions (within `Feature_importance/`).
6.  **Forecasting:** Generate sales predictions for the required forecast horizon.
7.  **Evaluation & Analysis:** Calculate key performance metrics (MAE, RMSE, WRMSSE), visualize results, and compare the performance of different models across various aggregation levels (consolidated in `RESULTS/`).

## Project Structure

The source code is organized as follows:

```
.
├── BasicModels
│   ├── ExponentialSmoothing  # Scripts and notebooks for Exponential Smoothing models
│   ├── lgbm                  # Scripts and notebooks for LightGBM models (category, item, store level)
│   │   ├── Category_Forecasting
│   │   ├── Item_Forecasting
│   │   └── Store_Forecasting
│   ├── LinearRegression      # Scripts and notebooks for Linear Regression models
│   └── LSTM                  # Scripts and notebooks for LSTM models
├── data
│   ├── calendar.csv
│   ├── processed             # Potentially for storing processed data files (e.g., grid_part_1.pkl)
│   ├── ReadMe.txt            # Original README for the M5 dataset
│   ├── sales_test_evaluation.csv
│   ├── sales_test_validation.csv
│   ├── sales_train_evaluation.csv
│   ├── sales_train_validation.csv
│   └── sell_prices.csv
├── df_master_creation.ipynb  # Notebook for creating the master dataframe and initial feature engineering
├── ExtraFiles
│   └── weights_validation.csv # Validation weights for WRMSSE calculation
├── Feature_importance        # Scripts and notebooks for feature importance analysis
│   ├── outputs
│   │   └── Feature Importances RF.png
│   ├── Dataframe_manipulation.ipynb
│   ├── MI_importances.py
│   └── RF_importances.py
├── pklFiles                  # Pickled files (e.g., preprocessed series, model objects, master dataframes)
├── README.md                 # This file
├── requirements.txt          # Python dependencies for the project
└── RESULTS
    ├── output                # Output plots (histograms, boxplots of MAE/RMSE)
    ├── results.ipynb         # Notebook for analyzing and visualizing results
    └── RMSE_MAE              # Pickled dataframes with MAE/RMSE metrics for each model
```

**Key Components Description:**

*   **`BasicModels/`**: Contains the core implementations of the different forecasting models. Each subdirectory typically includes Jupyter notebooks for experimentation and Python scripts for training or utility functions (e.g., `TimeSeries_creation_*.py` scripts for preparing data for Darts models).
*   **`data/`**: Stores the raw M5 competition data files.
*   **`df_master_creation.ipynb`**: A crucial Jupyter notebook for initial data loading, merging, extensive feature engineering, and saving the master dataset.
*   **`Feature_importance/`**: Includes scripts and notebooks dedicated to analyzing the importance of different engineered features for the forecasting models.
*   **`pklFiles/`**: This directory is used to store serialized Python objects (using `pickle` or `joblib`), such as preprocessed time series data (`cat_series_dict.pkl`, `low_series_dict.pkl`), the master dataframes (`df_master.pkl`), or potentially trained model objects.
*   **`RESULTS/`**: Contains notebooks and scripts for evaluating model performance, generating visualizations, and storing detailed metric results. The `results.ipynb` notebook is central for result aggregation and comparison.
*   **`requirements.txt`**: Lists all Python packages and their versions required to run the project, ensuring reproducibility.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pompos02/M5-CompetitionDemandForecasting.git
    cd M5-CompetitionDemandForecasting
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the M5-Competition Dataset:**
    *   Obtain the dataset from the [Kaggle M5 Forecasting Accuracy competition page](https://www.kaggle.com/c/m5-forecasting-accuracy/data).
    *   Place the required CSV files (`calendar.csv`, `sales_train_validation.csv`, `sell_prices.csv`, `sales_test_validation.csv`, `sales_test_evaluation.csv`) into the `./data/` directory.
    *   The `ExtraFiles/weights_validation.csv` file, also part of the M5 dataset, should be placed in the `./ExtraFiles/` directory.

## Usage

The project primarily consists of Jupyter notebooks that can be run sequentially or independently to perform data processing, train models, make predictions, and evaluate results.

1.  **Start Jupyter Lab or Jupyter Notebook:**
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
2.  **Recommended Execution Order:**
    *   Begin with `df_master_creation.ipynb` to process the raw data and generate the master dataset and initial features. This step is crucial as its outputs are used by subsequent scripts and notebooks.
    *   Run the `TimeSeries_creation_*.py` scripts located in the respective `BasicModels` subfolders if you intend to use Darts-based models (LSTM, Exponential Smoothing) to generate the required time series dictionary files.
    *   Explore notebooks within the `BasicModels/` subdirectories (e.g., `BasicModels/LSTM/LSTM_Forecast_low.ipynb`, `BasicModels/lgbm/Category_Forecasting/LGBM_Forecast_cat.ipynb`) to train and evaluate specific models.
    *   Run notebooks and scripts in `Feature_importance/` to analyze feature contributions.
    *   Finally, use `RESULTS/results.ipynb` to view aggregated performance metrics and visualizations across all models.

Follow the instructions and execute cells within each notebook. Ensure that the dataset is correctly set up as described in the [Setup and Installation](#setup-and-installation) section.

## Results Summary

The project culminates in a comprehensive evaluation of the implemented forecasting models. The `RESULTS/results.ipynb` notebook serves as the primary interface for analyzing and visualizing these outcomes. Key performance metrics, including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the M5 competition-specific Weighted Root Mean Squared Scaled Error (WRMSSE), are calculated and compared across models and aggregation levels.

The thesis research, which this codebase supports, found that the LSTM model demonstrated the best overall performance, achieving a WRMSSE value of 0.884. This result is competitive with top-ranking solutions from the M5 competition. The Exponential Smoothing model, despite its relative simplicity, also showed strong performance, particularly in terms of MAE (achieving 1.11). LightGBM models exhibited varied results across different hierarchical levels, underscoring the importance of careful feature engineering and hyperparameter tuning for tree-based models in complex forecasting tasks.

Detailed quantitative results, including MAE and RMSE metrics for each model (Exponential Smoothing, LSTM, Linear Regression, LightGBM at category, item, and store levels), are stored as pickled DataFrames in the `RESULTS/RMSE_MAE/` directory (e.g., `metrics_df_LSTM.pkl`, `metrics_df_LGBMcat.pkl`). Visualizations such as histograms and boxplots comparing model errors can be found in `RESULTS/output/`.

## Author

*   **Γιάννης Καραβέλλας** (Thesis Author)



## License

(Please specify the license for your project here, e.g., MIT, Apache 2.0. If no license is specified, it is typically assumed to be proprietary unless otherwise stated in the repository.)

## Acknowledgements

*   The organizers of the M5 Forecasting Accuracy competition for providing the dataset and a challenging benchmark.
*   The developers of the Darts, LightGBM, scikit-learn, and other open-source libraries used in this project.

