import os
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve

plt.switch_backend('Agg')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS_DIR = os.path.join(BASE_DIR, "modelos", "modelos_salvos")
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "diabetes_dataset.csv")

os.makedirs(MODELOS_DIR, exist_ok=True)

COLUNAS_TREINO = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

def prepare_input_data(data):
    input_dict = {
        'Pregnancies': [float(data.get("Pregnancies", 0))],
        'Glucose': [float(data.get("Glucose", 0))],
        'BloodPressure': [float(data.get("BloodPressure", 0))],
        'Insulin': [float(data.get("Insulin", 0))],
        'BMI': [float(data.get("BMI", 0))],
        'DiabetesPedigreeFunction': [float(data.get("DiabetesPedigreeFunction", 0))],
        'Age': [float(data.get("Age", 0))]
    }
    return pd.DataFrame(input_dict)

def img_to_base64(plt_figure):
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{img_b64}"

def plot_confusion_matrix(y_test, y_pred, model_name):
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'1. Confusion Matrix\n({model_name})', fontsize=10)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return img_to_base64(plt)

def plot_roc_curve(y_test, y_proba, model_name):
    plt.figure(figsize=(5, 4))
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='#2b5797', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title(f'2. ROC Curve\n({model_name})', fontsize=10)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return img_to_base64(plt)

def plot_local_importance(impacts, model_name, status_text):
    df_impact = pd.DataFrame(list(impacts.items()), columns=['Feature', 'Impact'])
    df_impact = df_impact.sort_values(by='Impact', ascending=True)

    plt.figure(figsize=(5, 4))
    colors = ['#ef5350' if x > 0 else '#66bb6a' for x in df_impact['Impact']]
    
    plt.barh(df_impact['Feature'], df_impact['Impact'], color=colors)
    plt.title(f'3. Local Impact\nStatus: {status_text}', fontsize=10)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.xlabel("Contribution to Risk")
    plt.tight_layout()
    return img_to_base64(plt)

def plot_global_importance(model, model_type, model_name):
    plt.figure(figsize=(5, 4))
    
    importances = []
    if model_type == 'logistic':
        importances = model.coef_[0]
    elif model_type == 'random_forest':
        importances = model.feature_importances_
    elif model_type == 'neural':
        importances = np.mean(np.abs(model.layers[0].get_weights()[0]), axis=1)

    df_imp = pd.DataFrame({
        'Feature': COLUNAS_TREINO,
        'Importance': np.abs(importances)
    }).sort_values('Importance', ascending=True)

    plt.barh(df_imp['Feature'], df_imp['Importance'], color='#5c6bc0')
    plt.title(f'4. Global Feature Importance\n({model_name})', fontsize=10)
    plt.xlabel('Absolute Relative Weight')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return img_to_base64(plt)

def plot_real_vs_pred(y_test, y_pred, model_name):
    plt.figure(figsize=(5, 4))
    real_counts = pd.Series(y_test).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    
    categories = ['No Diabetes', 'Diabetes']
    real_vals = [real_counts.get(0, 0), real_counts.get(1, 0)]
    pred_vals = [pred_counts.get(0, 0), pred_counts.get(1, 0)]
    
    x = np.arange(len(categories))
    width = 0.35

    plt.bar(x - width/2, real_vals, width, label='Real', color='#90caf9')
    plt.bar(x + width/2, pred_vals, width, label='Predicted', color='#ffab91')
    
    plt.title(f'5. Distribution: Real vs Predicted\n({model_name})', fontsize=10)
    plt.xticks(x, categories)
    plt.ylabel('Count (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    return img_to_base64(plt)

def plot_precision_recall(y_test, y_proba, model_name):
    plt.figure(figsize=(5, 4))
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.plot(recall, precision, color='#7e57c2', linewidth=2)
    plt.fill_between(recall, precision, alpha=0.2, color='#7e57c2')
    
    plt.title(f'6. Precision-Recall Curve\n({model_name})', fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    return img_to_base64(plt)