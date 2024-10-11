import os
import pickle

parent_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(parent_dir, 'irrigation_model.pkl')

with open(model_path, 'rb') as file:
    linear_model = pickle.load(file)

def predict_irrigation(input_data):
    prediction = linear_model.predict(input_data)
    predicted_irrigation_amount = prediction[0]
    predicted_irrigation_type = prediction[1]
    return predicted_irrigation_amount, predicted_irrigation_type