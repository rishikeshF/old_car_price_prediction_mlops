#import predict
import requests

sample = {'Car Name' : [ 'Maruti S PRESSO' ],
              'Year' : [ 2022.0 ], 
              'Distance' : [ 3878 ], 
              'Owner' : [ 1 ],
              'Fuel' : [ 'PETROL' ], 
              'Location' : [ 'HR' ], 
              'Drive' : [ 'Manual' ], 
              'Type' : [ 'HatchBack' ]
             }

url = "http://localhost:9696/predict"
response = requests.post(url, json=sample)
print(response.json())
""" 
predicted_price = predict.predict(sample)
print(predicted_price) #"""