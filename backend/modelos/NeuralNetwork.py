import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def criar_novo_modelo():
    """Função para criar e treinar um novo modelo"""
    print("Carregando dados...")
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_arquivo = os.path.join(diretorio_atual, '..', 'datasets', 'diabetes_dataset.csv')
    df = pd.read_csv(caminho_arquivo)
    
    print("\nPrimeiras linhas do dataset:")
    print(df.head())
    
    print("\nDistribuição da variável target (Outcome):")
    print(df['Outcome'].value_counts())
    
    df_clean = df.dropna()
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def create_model():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    model = create_model()
    
    print("\nResumo do modelo:")
    model.summary()

    print("\nTreinando o modelo...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    model.save('modelo_diabetes.h5')
    import joblib
    joblib.dump(scaler, 'scaler_diabetes.pkl')
    print("\nModelo salvo como 'modelo_diabetes_novo.h5'")
    print("Scaler salvo como 'scaler_diabetes_novo.pkl'")
    
    return model, scaler, X_test_scaled, y_test, history

def carregar_modelo_salvo():
    """Função para carregar um modelo salvo"""
    print("Carregando modelo salvo...")
    
    if not os.path.exists('C:/Users/searc/Codigos/DiabetalAI/backend/modelo_diabetes.h5'):
        print("ERRO: Modelo salvo não encontrado!")
        return None, None, None, None, None

    df = pd.read_csv('C:/Users/searc/Codigos/DiabetalAI/backend/datasets/diabetes_dataset.csv')
    df_clean = df.dropna()
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    import joblib
    try:
        scaler = joblib.load('C:/Users/searc/Codigos/DiabetalAI/backend/scaler_diabetes.pkl')
        X_test_scaled = scaler.transform(X_test)
    except:
        print("Scaler não encontrado, criando novo...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    model = keras.models.load_model('C:/Users/searc/Codigos/DiabetalAI/backend/modelo_diabetes.h5')
    print("Modelo carregado com sucesso!")
    
    return model, scaler, X_test_scaled, y_test, None

def avaliar_modelo(model, X_test_scaled, y_test, history=None):
    """Função para avaliar o modelo e gerar gráficos"""

    print("\nAvaliando o modelo...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"Acurácia no teste: {test_accuracy:.4f}")
    print(f"Precisão no teste: {test_precision:.4f}")
    print(f"Recall no teste: {test_recall:.4f}")

    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("\n" + "="*50)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    # Acurácia adicional no relatório
    from sklearn.metrics import accuracy_score
    accuracy_manual = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy_manual:.4f}")

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc_roc:.4f}")

    plt.figure(figsize=(18, 12))

    # Gráfico 1: Curva de Aprendizado - Acurácia
    if history is not None:
        plt.subplot(2, 3, 1)
        plt.plot(history.history['accuracy'], label='Acurácia Treino', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Acurácia Validação', linewidth=2)
        plt.title('Curva de Aprendizado - Acurácia', fontsize=14, fontweight='bold')
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

    # Gráfico 2: Curva de Aprendizado - Loss
    if history is not None:
        plt.subplot(2, 3, 2)
        plt.plot(history.history['loss'], label='Loss Treino', linewidth=2)
        plt.plot(history.history['val_loss'], label='Loss Validação', linewidth=2)
        plt.title('Curva de Aprendizado - Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

    # Gráfico 3: Matriz de Confusão
    plt.subplot(2, 3, 3)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
    plt.xlabel('Predito', fontsize=12)
    plt.ylabel('Real', fontsize=12)

    # Gráfico 4: Curva ROC
    plt.subplot(2, 3, 4)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.4f})', linewidth=2, color='red')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('Curva ROC', fontsize=14, fontweight='bold')
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Gráfico 5: Distribuição das Probabilidades
    plt.subplot(2, 3, 5)
    plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='Não Diabético', bins=20, color='blue', edgecolor='black')
    plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='Diabético', bins=20, color='red', edgecolor='black')
    plt.title('Distribuição das Probabilidades', fontsize=14, fontweight='bold')
    plt.xlabel('Probabilidade Prevista', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Gráfico 6: Importância das Features
    plt.subplot(2, 3, 6)
    first_layer_weights = np.abs(model.layers[0].get_weights()[0])
    feature_importance = np.mean(first_layer_weights, axis=1)
    
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='green', alpha=0.7, edgecolor='black')
    plt.title('Importância das Features', fontsize=14, fontweight='bold')
    plt.xlabel('Importância', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*50)
    print("IMPORTÂNCIA DAS FEATURES (aproximada)")
    print("="*50)
    importance_df_sorted = importance_df.sort_values('Importance', ascending=False)
    print(importance_df_sorted.to_string(index=False))

def fazer_previsao(model, scaler):
    """Função para fazer previsões com dados inputados pelo usuário"""
    print("\n" + "="*50)
    print("PREVISÃO PARA NOVOS DADOS")
    print("="*50)
    
    while True:
        print("\nDigite os dados para previsão:")
        try:
            pregnancies = float(input("Pregnancies (gravidezes): "))
            glucose = float(input("Glucose (glicose): "))
            blood_pressure = float(input("BloodPressure (pressão arterial): "))
            insulin = float(input("Insulin (insulina): "))
            bmi = float(input("BMI (índice de massa corporal): "))
            diabetes_pedigree = float(input("DiabetesPedigreeFunction: "))
            age = float(input("Age (idade): "))

            exemplo_dados = np.array([[pregnancies, glucose, blood_pressure, insulin, bmi, diabetes_pedigree, age]])
            exemplo_dados_scaled = scaler.transform(exemplo_dados)
            probabilidade = model.predict(exemplo_dados_scaled, verbose=0)[0][0]
            
            print(f"\n" + "="*30)
            print("RESULTADO DA PREVISÃO")
            print("="*30)
            print(f"Probabilidade de ter diabetes: {probabilidade:.4f} ({probabilidade*100:.2f}%)")
            print(f"Previsão: {'DIABETES' if probabilidade > 0.5 else 'NÃO DIABETES'}")

            if probabilidade < 0.3:
                risco = "BAIXO RISCO"
                cor_risco = "🟢"
            elif probabilidade < 0.7:
                risco = "RISCO MODERADO"
                cor_risco = "🟡"
            else:
                risco = "ALTO RISCO"
                cor_risco = "🔴"
            print(f"Nível de risco: {cor_risco} {risco}")
            print("="*30)
            
        except ValueError:
            print("Erro: Por favor, digite apenas números.")
        except Exception as e:
            print(f"Erro durante a previsão: {e}")
        
        continuar = input("\nDeseja fazer outra previsão? (s/n): ").lower()
        if continuar != 's':
            break

def main():
    """Função principal com switch"""
    print("="*60)
    print("SISTEMA DE PREDIÇÃO DE DIABETES")
    print("="*60)
    print("\nOpções disponíveis:")
    print("1. Criar e treinar novo modelo")
    print("2. Usar modelo salvo")
    print("3. Sair")
    
    while True:
        try:
            opcao = int(input("\nEscolha uma opção (1-3): "))
            
            if opcao == 1:
                print("\n" + "="*50)
                print("CRIANDO NOVO MODELO")
                print("="*50)
                model, scaler, X_test_scaled, y_test, history = criar_novo_modelo()
                if model is not None:
                    avaliar_modelo(model, X_test_scaled, y_test, history)

                    fazer_pred = input("\nDeseja fazer previsões para novos dados? (s/n): ").lower()
                    if fazer_pred == 's':
                        fazer_previsao(model, scaler)
                    else:
                        print("Previsões para novos dados cancelada.")
                
            elif opcao == 2:
                print("\n" + "="*50)
                print("USANDO MODELO SALVO")
                print("="*50)
                model, scaler, X_test_scaled, y_test, _ = carregar_modelo_salvo()
                if model is not None:
                    avaliar_modelo(model, X_test_scaled, y_test)

                    fazer_pred = input("\nDeseja fazer previsões para novos dados? (s/n): ").lower()
                    if fazer_pred == 's':
                        fazer_previsao(model, scaler)
                    else:
                        print("Previsões para novos dados cancelada.")
                
            elif opcao == 3:
                print("Saindo...")
                break
                
            else:
                print("Opção inválida! Escolha 1, 2 ou 3.")
                
        except ValueError:
            print("Por favor, digite um número válido.")
        except KeyboardInterrupt:
            print("\nPrograma interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()