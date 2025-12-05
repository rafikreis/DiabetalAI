import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model, Sequential 
from tensorflow.keras.layers import Dense, Dropout, Input 

from backend.utils.utils_ai import (
    MODELOS_DIR, DATASET_PATH, COLUNAS_TREINO,
    prepare_input_data, plot_confusion_matrix, plot_roc_curve, 
    plot_local_importance, plot_global_importance, 
    plot_real_vs_pred, plot_precision_recall
)

def train_all_models():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset não encontrado em: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna()
    X = df[COLUNAS_TREINO]
    y = df['Outcome']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=COLUNAS_TREINO)

    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train_scaled_df, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled_df, y_train)

    nn_model = Sequential([
        Input(shape=(len(COLUNAS_TREINO),)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

    joblib.dump(log_model, os.path.join(MODELOS_DIR, "modelo_logistico.pkl"))
    joblib.dump(rf_model, os.path.join(MODELOS_DIR, "modelo_random_forest.pkl"))
    nn_model.save(os.path.join(MODELOS_DIR, "modelo_neural.h5"))
    joblib.dump(scaler, os.path.join(MODELOS_DIR, "scaler_geral.pkl"))

def load_models_and_data():
    files = ["modelo_logistico.pkl", "modelo_random_forest.pkl", "modelo_neural.h5", "scaler_geral.pkl"]
    missing = [f for f in files if not os.path.exists(os.path.join(MODELOS_DIR, f))]
    
    if missing:
        train_all_models()

    try:
        models = {}
        models["logistic"] = joblib.load(os.path.join(MODELOS_DIR, "modelo_logistico.pkl"))
        models["random_forest"] = joblib.load(os.path.join(MODELOS_DIR, "modelo_random_forest.pkl"))
        models["neural"] = load_model(os.path.join(MODELOS_DIR, "modelo_neural.h5"))
        models["scaler"] = joblib.load(os.path.join(MODELOS_DIR, "scaler_geral.pkl"))

        data_test = None
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            df = df.dropna()
            X = df[COLUNAS_TREINO]
            y = df['Outcome']
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            X_test_scaled = models["scaler"].transform(X_test)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=COLUNAS_TREINO)
            
            data_test = {
                "X_test_arr": X_test_scaled,
                "X_test_df": X_test_scaled_df,
                "y_test": y_test
            }

        return models, data_test
    except Exception as e:
        print(f"Erro ao carregar modelos: {e}")
        return None, None

MODELS, DATA_TEST = load_models_and_data()

def calculate_local_impacts(model_type, model, scaled_data_df):
    impacts = {}
    vals = scaled_data_df.values[0]
    
    if model_type == 'logistic':
        coeffs = model.coef_[0]
        for i, c in enumerate(COLUNAS_TREINO): 
            impacts[c] = coeffs[i] * vals[i]
            
    elif model_type == 'random_forest':
        imps = model.feature_importances_
        for i, c in enumerate(COLUNAS_TREINO): 
            impacts[c] = imps[i] * vals[i]
            
    elif model_type == 'neural':
        w = np.mean(model.layers[0].get_weights()[0], axis=1)
        for i, c in enumerate(COLUNAS_TREINO): 
            impacts[c] = w[i] * vals[i]
            
    return impacts

def generate_graphs_for_model(model_key, model_obj, model_title, input_scaled, input_df, y_proba_input, y_test, X_test_arr, X_test_df):
    status_text = "High Risk" if y_proba_input > 0.5 else "Low Risk"
    
    if model_key == 'neural':
        imps_local = calculate_local_impacts(model_key, model_obj, pd.DataFrame(input_scaled, columns=COLUNAS_TREINO))
    else:
        imps_local = calculate_local_impacts(model_key, model_obj, input_df)

    if model_key == 'neural':
        y_proba_test = model_obj.predict(X_test_arr, verbose=0).ravel()
    elif model_key == 'logistic':
        y_proba_test = model_obj.predict_proba(X_test_df)[:, 1]
    else: 
        y_proba_test = model_obj.predict_proba(X_test_df)[:, 1]
        
    y_pred_test = (y_proba_test > 0.5).astype(int)

    g1 = plot_local_importance(imps_local, model_title, status_text)
    g2 = plot_global_importance(model_obj, model_key, model_title)
    g3 = plot_confusion_matrix(y_test, y_pred_test, model_title)
    g4 = plot_roc_curve(y_test, y_proba_test, model_title)
    g5 = plot_real_vs_pred(y_test, y_pred_test, model_title)
    g6 = plot_precision_recall(y_test, y_proba_test, model_title)

    return [g1, g2, g3, g4, g5, g6]

def call_models(data):
    global MODELS, DATA_TEST
    
    if MODELS is None:
        MODELS, DATA_TEST = load_models_and_data()
        if MODELS is None: return {"error": "Critical AI System Error."}

    input_df_raw = prepare_input_data(data)
    input_scaled = MODELS['scaler'].transform(input_df_raw)
    input_scaled_df = pd.DataFrame(input_scaled, columns=COLUNAS_TREINO)

    X_test_arr = DATA_TEST["X_test_arr"]
    X_test_df = DATA_TEST["X_test_df"]
    y_test = DATA_TEST["y_test"]

    results = []

    # 1. Logistic
    prob_log = MODELS['logistic'].predict_proba(input_scaled_df)[0][1]
    graficos_log = generate_graphs_for_model(
        'logistic', MODELS['logistic'], 'Regressão Logística', 
        input_scaled, input_scaled_df, prob_log, y_test, X_test_arr, X_test_df
    )
    results.append({
        'name': 'Regressão Logística',
        'prob': float(prob_log),
        'graficos': graficos_log
    })

    # 2. Random Forest
    prob_rf = MODELS['random_forest'].predict_proba(input_scaled_df)[0][1]
    graficos_rf = generate_graphs_for_model(
        'random_forest', MODELS['random_forest'], 'Random Forest', 
        input_scaled, input_scaled_df, prob_rf, y_test, X_test_arr, X_test_df
    )
    results.append({
        'name': 'Random Forest',
        'prob': float(prob_rf),
        'graficos': graficos_rf
    })

    # 3. Rede neural
    prob_nn = float(MODELS['neural'].predict(input_scaled, verbose=0)[0][0])
    graficos_nn = generate_graphs_for_model(
        'neural', MODELS['neural'], 'Rede Neural', 
        input_scaled, input_scaled_df, prob_nn, y_test, X_test_arr, X_test_df
    )
    results.append({
        'name': 'Rede Neural',
        'prob': float(prob_nn),
        'graficos': graficos_nn
    })

    best_res = max(results, key=lambda x: abs(x['prob'] - 0.5))
    has_diabetes = bool(best_res['prob'] > 0.5)
    
    print("\n" + "="*50)
    print(f"Modelo vencedor: {best_res['name']}")
    print("="*50 + "\n")

    if best_res['name'] == 'Regressão Logística':
        imp_final = calculate_local_impacts('logistic', MODELS['logistic'], input_scaled_df)
    elif best_res['name'] == 'Random Forest':
        imp_final = calculate_local_impacts('random_forest', MODELS['random_forest'], input_scaled_df)
    else:
        imp_final = calculate_local_impacts('neural', MODELS['neural'], pd.DataFrame(input_scaled, columns=COLUNAS_TREINO))

    sorted_imp = sorted(imp_final.items(), key=lambda x: x[1], reverse=True)
    causes = [k for k, v in sorted_imp if v > 0][:3]

    detalhes = []
    for res in results:
        detalhes.append({
            "tipo_modelo": res['name'],
            "probabilidade_percentual": float(round(res['prob'] * 100, 2)),
            "graficos": res['graficos'] 
        })

    response = {
        "diagnostico": {
            "tem_diabetes": has_diabetes,
            "probabilidade_media": float(round(np.mean([r['prob'] for r in results]) * 100, 2)),
            "modelo_mais_confiante": best_res['name'],
            "principais_causas": causes if has_diabetes else ["Normal levels"]
        },
        "detalhes_modelos": detalhes
    }
    
    return response