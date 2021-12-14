import joblib
import os

current_path = os.path.dirname(os.path.realpath(__file__))

final_model = joblib.load('model/final_model.sav') # importing model to predict

def predict(attributes):
    final_pred = final_model.predict(attributes)
    return final_pred
