import pickle
import pandas as pd
from flask import Flask, request, jsonify
import mlflow

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
RUN_ID = "d766a676d45146e3912e38740b3b5036"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

path = client.download_artifacts(run_id=RUN_ID, path='preprocess/preprocess.bin')
logged_model = f'runs:/{RUN_ID}/models_mlflow'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# loading preprocessing code
with open(path, 'rb') as f_in : 
    oe, scaler = pickle.load(f_in)

# loading xgboost model from mlflow
model = pickle.load(open("model.pkl", 'rb'))

def preprocess(X):
    X = pd.DataFrame(X)
    categorical_features = ['Car Name', 'Fuel', 'Location', 'Drive', 'Type']
    X[categorical_features] = oe.transform( X[categorical_features] )
    X = scaler.transform(X)
    return X
    
def predict(X):
    formatted_input = preprocess(X)
    predicted_price = model.predict(formatted_input)
    return float(predicted_price[0])


app = Flask('car-price-prediction')
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    car_details = request.get_json()

    pred = predict(car_details) 

    result = {
        'car_price' : pred
    }
    
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)