import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib  # <<< ADICIONADO PARA SALVAR O MODELO


def criar_novo_modelo_regressao():
    """Função para criar e treinar um novo modelo de regressão logística e salvar o modelo"""
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
    
    print(f"\nShape dos dados: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTreinando modelo de Regressão Logística...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("Modelo de Regressão Logística treinado com sucesso!")

    caminho_salvar = os.path.join(diretorio_atual, 'modelos_salvos')
    os.makedirs(caminho_salvar, exist_ok=True)  # cria a pasta se não existir

    joblib.dump(model, os.path.join(caminho_salvar, 'modelo_logistico.pkl'))
    joblib.dump(scaler, os.path.join(caminho_salvar, 'scaler_logic.pkl'))
    
    print(f"\n✔ Modelo salvo em: {caminho_salvar}/modelo_logistico.pkl")
    print(f"✔ Scaler salvo em: {caminho_salvar}/scaler_logic.pkl")

    y_pred_binary = model.predict(X_test_scaled)
    
    return model, scaler, X_test_scaled, y_test, y_pred_binary


def avaliar_modelo_regressao(model, X_test_scaled, y_test, y_pred_binary):
    """Função para avaliar o modelo de regressão logística e gerar gráficos"""

    accuracy = accuracy_score(y_test, y_pred_binary)

    print("\n" + "="*60)
    print("MÉTRICAS DE CLASSIFICAÇÃO - REGRESSÃO LOGÍSTICA")
    print("="*60)
    print(f"Acurácia: {accuracy:.4f}")

    print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred_binary))

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc_roc:.4f}")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_test, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
    plt.xlabel('Predito')
    plt.ylabel('Real')

    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_roc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)  # linha aleatória
    plt.title('Curva ROC', fontsize=14, fontweight='bold')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    real_counts = pd.Series(y_test).value_counts()
    predito_counts = pd.Series(y_pred_binary).value_counts()
    categories = ['Não Diabético', 'Diabético']
    real_values = [real_counts.get(0, 0), real_counts.get(1, 0)]
    predito_values = [predito_counts.get(0, 0), predito_counts.get(1, 0)]
    x = np.arange(len(categories))
    width = 0.35

    plt.bar(x - width/2, real_values, width, label='Real')
    plt.bar(x + width/2, predito_values, width, label='Predito')
    plt.xticks(x, categories)
    plt.title('Distribuição: Real vs Predito', fontsize=14, fontweight='bold')
    plt.xlabel('Categoria')
    plt.ylabel('Quantidade')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age']
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coeficiente': coefficients
    }).sort_values('Coeficiente', key=abs, ascending=True)

    plt.barh(importance_df['Feature'], importance_df['Coeficiente'], alpha=0.7)
    plt.title('Importância dos Coeficientes', fontsize=14, fontweight='bold')
    plt.xlabel('Valor do Coeficiente')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("COEFICIENTES DA REGRESSÃO LOGÍSTICA")
    print("="*60)
    coefficients_abs = pd.DataFrame({
        'Feature': feature_names,
        'Coeficiente': coefficients,
        'Absoluto': np.abs(coefficients)
    }).sort_values('Absoluto', ascending=False)

    print(coefficients_abs.to_string(index=False))
    print(f"\nIntercept: {model.intercept_[0]:.4f}")



def fazer_previsao_regressao(model, scaler):
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

            # Previsão da classe (0 ou 1)
            previsao_classe = model.predict(exemplo_dados_scaled)[0]
            
            print(f"\n" + "="*40)
            print("RESULTADO DA PREVISÃO")
            print("="*40)
            print(f"Previsão: {'DIABETES' if previsao_classe == 1 else 'NÃO DIABETES'}")
            
            if previsao_classe == 1:
                print("Nível de risco: 🔴 ALTO RISCO")
                print("💡 Recomendação: Consultar médico especialista")
            else:
                print("Nível de risco: 🟢 BAIXO RISCO") 
                print("💡 Recomendação: Manter hábitos saudáveis")
                
            print("="*40)
            
        except ValueError:
            print("Erro: Por favor, digite apenas números.")
        except Exception as e:
            print(f"Erro durante a previsão: {e}")
        
        continuar = input("\nDeseja fazer outra previsão? (s/n): ").lower()
        if continuar != 's':
            break

def main():
    """Função principal com switch para Regressão Logística"""
    print("="*60)
    print("SISTEMA DE PREDIÇÃO DE DIABETES - REGRESSÃO LOGÍSTICA")
    print("="*60)
    print("\nOpções disponíveis:")
    print("1. Criar e treinar novo modelo de Regressão Logística")
    print("2. Fazer previsão com dados inseridos")
    print("3. Sair")
    
    model = None
    scaler = None
    
    while True:
        try:
            opcao = int(input("\nEscolha uma opção (1-3): "))
            
            if opcao == 1:
                print("\n" + "="*50)
                print("CRIANDO NOVO MODELO - REGRESSÃO LOGÍSTICA")
                print("="*50)
                model, scaler, X_test_scaled, y_test, y_pred_binary = criar_novo_modelo_regressao()
                if model is not None:
                    avaliar_modelo_regressao(model, X_test_scaled, y_test, y_pred_binary)
                    
                    fazer_pred = input("\nDeseja fazer previsões para novos dados? (s/n): ").lower()
                    if fazer_pred == 's':
                        fazer_previsao_regressao(model, scaler)
                    else:
                        print("Previsões para novos dados cancelada.")
                
            elif opcao == 2:
                if model is None:
                    print("\n⚠️  Primeiro você precisa treinar um modelo (Opção 1)!")
                    continue
                    
                print("\n" + "="*50)
                print("PREVISÃO COM DADOS INSERIDOS")
                print("="*50)
                fazer_previsao_regressao(model, scaler)
                
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