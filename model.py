import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import yaml
import os


def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val 

def preprocesing(df):
    numeric_cols= df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')  # optional, to keep columns not listed
    return preprocessor

def build_model(model_name):
    if model_name == "LinearRegression":
        return LinearRegression()
    elif model_name == "Ridge":
        return Ridge()
    elif model_name == "Lasso":
        return Lasso()
    elif model_name == "ElasticNet":
        return ElasticNet()
    elif model_name == "DecisionTreeRegressor":
        return DecisionTreeRegressor()
    elif model_name == "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_name == "GradientBoostingRegressor":
        return GradientBoostingRegressor()
    elif model_name == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor()
    elif model_name == "SVR":
        return SVR()
    elif model_name == "KNeighborsRegressor":
        return KNeighborsRegressor()
    elif model_name == "MLPRegressor":
        return MLPRegressor()
    elif model_name == "XGBoost":
        return xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    elif model_name == "CatBoost":
        return CatBoostRegressor(verbose=0, random_seed=42)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

def pipeline(preprocessor, model_name): 
    pipeline_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', build_model(model_name))
    ])
    return pipeline_model


def train(pipeline_model, X_train, y_train):
    pipeline_model.fit(X_train, y_train)
    return pipeline_model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    y_val_clipped = np.clip(y_val, a_min=0, a_max=None)
    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
    msle = mean_squared_log_error(y_val_clipped, y_pred_clipped)
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation R²: {r2:.4f}")
    print(f"Validation MSLE: {msle:.4f}")
    return mse, r2, msle
def save_results(config, mse, r2, msle, filename="result.csv"):
    # Combine config and metrics into one dict
    results = config.copy()
    results.update({
        "MSE": mse,
        "R2": r2,
        "MSLE": msle
    })
    
    # Convert to DataFrame (one row)
    df_result = pd.DataFrame([results])
    
    # If file exists, append without header
    if os.path.isfile(filename):
        df_result.to_csv(filename, mode='a', header=False, index=False)
    else:
        df_result.to_csv(filename, index=False)
    
    print(f"Results saved to {filename}")

def run_pipeline(config):
    df = load_data(config["data_path"])
    X_train, X_val, y_train, y_val = split_data(df, config["target_column"], config["test_size"], config["random_seed"])
    preprocessor = preprocesing(X_train)
    pineline_model_1 = pipeline(preprocessor, config["model_name"])
    trained_model = train(pineline_model_1, X_train, y_train)
    mse, r2, msle = evaluate_model(trained_model, X_val, y_val)
    save_results(config, mse, r2, msle, filename="result.csv")


if __name__ == "__main__":
    # Load config from 'config.yaml'
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    

    run_pipeline(config)