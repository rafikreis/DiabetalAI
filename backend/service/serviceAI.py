import pickle
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

def call_models(data):
    age = data.get("idade")
    pregnancies = data.get("gestacoes")
    bmi = data.get("imc")
    glucose = data.get("glicose")
    blood_pressure = data.get("pressao")
    insulin = data.get("insulina")
    DiabetesPedigreeFunction = data.get("historico")
    print("Chamando modelos com os dados:", data)   


def load_models():
    models = {}

    with open('modelos/modelo_logistico.pkl', 'rb') as f:
        models['logistic'] = pickle.load(f)
    
    with open('modelos/modelo_random_forest.pkl', 'rb') as f:
        models['random_forest'] = pickle.load(f)
    
    models['neural'] = load_model('modelos/modelo_neural.h5')
    
    with open('modelos/scaler_logic.pkl', 'rb') as f:
        models['scaler_logic'] = pickle.load(f)
    
    with open('modelos/scaler_forest.pkl', 'rb') as f:
        models['scaler_forest'] = pickle.load(f)
    
    with open('modelos/scaler_neural.pkl', 'rb') as f:
        models['scaler_neural'] = pickle.load(f)
    
    return models

MODELS = load_models()

def prepare_input_data(data):
    age = data.get("idade")
    pregnancies = data.get("gestacoes")
    bmi = data.get("imc")
    glucose = data.get("glicose")
    blood_pressure = data.get("pressao")
    insulin = data.get("insulina")
    diabetes_pedigree = data.get("historico")
    
    return np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

def predict_logistic(data):

    input_data = prepare_input_data(data)
    scaled_data = MODELS['scaler_logic'].transform(input_data)
    
    prediction = MODELS['logistic'].predict(scaled_data)
    
    return {
        'prediction': int(prediction[0]),
        'model': 'logistic_regression'
    }

def predict_random_forest(data):

    input_data = prepare_input_data(data)
    scaled_data = MODELS['scaler_forest'].transform(input_data)
    
    prediction = MODELS['random_forest'].predict(scaled_data)
    
    return {
        'prediction': int(prediction[0]),
        'model': 'random_forest'
    }

def predict_neural_network(data):

    input_data = prepare_input_data(data)
    scaled_data = MODELS['scaler_neural'].transform(input_data)
    
    prediction = MODELS['neural'].predict(scaled_data)
    prediction_class = (prediction > 0.5).astype(int)
    
    return {
        'prediction': int(prediction_class[0][0]),
        'model': 'neural_network'
    }