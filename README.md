# old_car_price_prediction_mlops
Complete end-to-end MlOps implementation for training, maintaining and monitoring a machine learning model that predicts the price of an old (second hand) car based on several different relevant factors.

### Frameworks used :  <br>
MlFlow - For Experiment Tracking, Model registry <br>
Prefect - For model orchestration <br>
Flask and Docker - For Deployment <br>
Grafana - For Drift detection / Monitoring <br>

### Environment setup : 
Create a virtual environment using the provided requirements.txt and run the code below in the same.

### Model creation/EDA :
Ipython notebooks regarding model creation are provided in 'ML-experiments' folder.

### How to use MlFlow for experiment tracking ? 
cd into the project folder where 'mlflow.db' is located. Use the command ```mlflow ui --backend-store-uri sqlite:///mlflow.db``` to start the GUI for mlflow. You can view the dashboard in chrome browser at "http://127.0.0.1:5000" (default port) A dashboard as show below will be generated. The models tab allows the user for model versioning.

![image](https://github.com/rishikeshF/old_car_price_prediction_mlops/assets/16107041/1379b4bf-cae1-4b58-a03c-73fbc288ea2f)

![image](https://github.com/rishikeshF/old_car_price_prediction_mlops/assets/16107041/3508ef71-c42c-4bc4-a540-351f5c432826)

### How to use Prefect for model orchestration ? 
Set up an api to use PREFECT using the command ```prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api```
Then, start the server using ```prefect server start```
This will start the UI at http://127.0.0.1:4200

![image](https://github.com/rishikeshF/old_car_price_prediction_mlops/assets/16107041/c13ceaed-3855-4f5d-b4d2-dd96b857056d)

![image](https://github.com/rishikeshF/old_car_price_prediction_mlops/assets/16107041/3baa9e16-5782-4c9a-8e4e-b91bf14eac0a)

Use the deployment tab to schedule runs to train or deploy the model. 

![image](https://github.com/rishikeshF/old_car_price_prediction_mlops/assets/16107041/828507ec-4f84-4917-97ef-e80366f64029)

### How to deploy the model ? 
Two types of deployment are provided. 
1. One is using simple model's pickle file, FLASK API and Docker.
2. Second way is to use the Models from Model registry and deploy it using Docker file.

Docker images are provided in both the ways which can be easily used to deploy the model at any platform of your choice. 

### Model Monitoring 
As the dataset used in this problem statement is of static nature, Drift is checked for in test dataset with respect to training dataset. The code and visualizations for the same are available in 'Monitoring' folder.

![image](https://github.com/rishikeshF/old_car_price_prediction_mlops/assets/16107041/21840032-62b1-4295-ace6-3cca5e5e8560)

![image](https://github.com/rishikeshF/old_car_price_prediction_mlops/assets/16107041/2289e074-8f5d-4b1d-83a1-7f6f23cf4bb0)

### Upcoming updates : 
1. Model deployed as a web service on Hugging Face spaces
