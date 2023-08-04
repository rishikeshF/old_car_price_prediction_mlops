import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
import pickle
from pathlib import Path
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import mlflow
from prefect import flow, task
from prefect_aws import S3Bucket

@task(name="read csv file", retries=3, retry_delay_seconds=3)
def read_data(filename:str="data/cars_24_combined.csv") -> pd.DataFrame:
    """Read data into dataframe"""
    df = pd.read_csv(filename, index_col=None).drop(columns=['Unnamed: 0'])
    df.dropna(inplace=True)
    df.Location = df.Location.str.split('-').apply(lambda x : x[0])
    df['Price'] = df.Price.apply(lambda x : x/1000)
    return df 

@task(name="Create train test set")
def preprocess(df: pd.DataFrame ):
    
    categorical_features = ['Car Name', 'Fuel', 'Location', 'Drive', 'Type']
    target_feature = 'Price'

    X = df.drop(columns=target_feature)
    y = df[target_feature].values

    oe = OrdinalEncoder()
    oe.fit(X=X[categorical_features])
    X[categorical_features] = oe.transform( X[categorical_features] )

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    with open('preprocess/preprocess.bin', 'wb') as f_out : 
        pickle.dump( (oe, scaler), f_out)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test

@task(name="Train model", log_prints=True)
def train_model(X_train, X_test, y_train, y_test):

    mlflow.xgboost.autolog()
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    

    with mlflow.start_run():
        params = {
        'max_depth': 30,
        'learning_rate': .03,
        'reg_alpha': .5,
        'reg_lambda': .5,
        'min_child_weight': .4,
        'objective': 'reg:squarederror',
        'seed': 42
        }

        booster = xgb.train(
            params=params,
            dtrain=xg_train,
            num_boost_round=1000,
            evals=[(xg_test, 'testing')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(xg_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_artifact("preprocess/preprocess.bin", artifact_path="preprocess")
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
    
    return None 

@flow(name="Main Flow")
def main_flow():
    filename = "data/cars_24_combined.csv"

    # mlflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("prefect_exp")

    # load data
    # Loads the configuration of the S3 Bucket from the S3Bucket Block
    s3_bucket_block = S3Bucket.load("s3-bucket-example")
    # Downloads the data from the S3 bucket folder name "data" to the local folder "data"
    s3_bucket_block.download_folder_to_path(from_folder="data", to_folder="data")

    df = read_data(filename)
    
    # transform data
    X_train, X_test, y_train, y_test = preprocess(df)

    # train model
    train_model(X_train, X_test, y_train, y_test)

if __name__=="__main__":
    main_flow()