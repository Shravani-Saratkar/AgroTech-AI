import pandas as pd
import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from notebooks.irrigation import linear_model
file_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')), 'simulated_irrigation_data.csv')
from sklearn.preprocessing import StandardScaler
import numpy as np

# Retrieve the simulated dataset from the CSV file
def retrieve_simulated_data(file_path):
    return pd.read_csv(file_path)

# Retrieve the simulated data from the CSV file
df_retrieved = retrieve_simulated_data(file_path)
print(df_retrieved.head())

# Prepare the input data for the model
def prepare_input_data(df):
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['Soil Type', 'Crop Type', 'Geographical Location'])
    
    # Scale continuous features
    scaler = StandardScaler()
    df_encoded[['Avg Temperature', 'Moisture Level']] = scaler.fit_transform(df_encoded[['Avg Temperature', 'Moisture Level']])
    
    # Select the relevant features
    X = df_encoded[['Avg Temperature', 'Moisture Level'] + 
                   [col for col in df_encoded.columns if col.startswith(('Soil Type', 'Crop Type', 'Geographical Location'))]].values
    return X

input_data = prepare_input_data(df_retrieved)

# Make predictions with the model
def predict_irrigation(input_data):
    prediction = linear_model.predict(input_data)
    predicted_irrigation_amount = prediction[:, 0]
    predicted_irrigation_type = np.argmax(prediction[:, 1:], axis=1)
    return predicted_irrigation_amount, predicted_irrigation_type

predicted_irrigation_amount, predicted_irrigation_type = predict_irrigation(input_data)
print("Predicted Irrigation Amount:", predicted_irrigation_amount)
print("Predicted Irrigation Type:", predicted_irrigation_type)