import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def criar_novo_modelo_regressao():
    """Função para criar e treinar um novo modelo de regressão linear"""
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

    print("\nTreinando modelo de Regressão Linear...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    print("Modelo de Regressão Linear treinado com sucesso!")

    y_pred = model.predict(X_test_scaled)

    y_pred_binary = (y_pred > 0.5).astype(int)
    
    return model, scaler, X_test_scaled, y_test, y_pred, y_pred_binary

def avaliar_modelo_regressao(model, X_test_scaled, y_test, y_pred, y_pred_binary):
    """Função para avaliar o modelo de regressão e gerar gráficos"""

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MÉTRICAS DE REGRESSÃO")
    print("="*50)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    accuracy = np.mean(y_pred_binary == y_test)
    
    print("\n" + "="*50)
    print("MÉTRICAS DE CLASSIFICAÇÃO (Threshold = 0.5)")
    print("="*50)
    print(f"Acurácia: {accuracy:.4f}")

    print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred_binary))

    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC: {auc_roc:.4f}")

    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Valores Reais vs Preditos', fontsize=14, fontweight='bold')
    plt.xlabel('Valores Reais', fontsize=12)
    plt.ylabel('Valores Preditos', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    residuos = y_test - y_pred
    plt.scatter(y_pred, residuos, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Análise de Resíduos', fontsize=14, fontweight='bold')
    plt.xlabel('Valores Preditos', fontsize=12)
    plt.ylabel('Resíduos', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    cm = confusion_matrix(y_test, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
    plt.xlabel('Predito', fontsize=12)
    plt.ylabel('Real', fontsize=12)

    plt.subplot(2, 3, 4)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.4f})', linewidth=2, color='red')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('Curva ROC', fontsize=14, fontweight='bold')
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.hist(y_pred[y_test == 0], alpha=0.7, label='Não Diabético', bins=20, color='blue', edgecolor='black')
    plt.hist(y_pred[y_test == 1], alpha=0.7, label='Diabético', bins=20, color='red', edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold = 0.5')
    plt.title('Distribuição das Probabilidades', fontsize=14, fontweight='bold')
    plt.xlabel('Probabilidade Prevista', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    coefficients = model.coef_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coeficiente': coefficients
    }).sort_values('Coeficiente', key=abs, ascending=True)
    
    plt.barh(importance_df['Feature'], importance_df['Coeficiente'], color='purple', alpha=0.7, edgecolor='black')
    plt.title('Importância dos Coeficientes', fontsize=14, fontweight='bold')
    plt.xlabel('Valor do Coeficiente', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*50)
    print("COEFICIENTES DA REGRESSÃO LINEAR")
    print("="*50)
    coefficients_abs = pd.DataFrame({
        'Feature': feature_names,
        'Coeficiente': coefficients,
        'Absoluto': np.abs(coefficients)
    }).sort_values('Absoluto', ascending=False)
    
    print(coefficients_abs.to_string(index=False))
    print(f"\nIntercept: {model.intercept_:.4f}")

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

            probabilidade = model.predict(exemplo_dados_scaled)[0]
            
            print(f"\n" + "="*40)
            print("RESULTADO DA PREVISÃO")
            print("="*40)
            print(f"Score de regressão: {probabilidade:.4f}")
            print(f"Probabilidade de diabetes: {probabilidade:.4f} ({probabilidade*100:.2f}%)")
            print(f"Previsão: {'DIABETES' if probabilidade > 0.5 else 'NÃO DIABETES'}")
            
            if probabilidade < 0.2:
                risco = "MUITO BAIXO RISCO"
                cor_risco = "🟢"
            elif probabilidade < 0.4:
                risco = "BAIXO RISCO"
                cor_risco = "🟢"
            elif probabilidade < 0.6:
                risco = "RISCO MODERADO"
                cor_risco = "🟡"
            elif probabilidade < 0.8:
                risco = "ALTO RISCO"
                cor_risco = "🟠"
            else:
                risco = "MUITO ALTO RISCO"
                cor_risco = "🔴"
            
            print(f"Nível de risco: {cor_risco} {risco}")
            
            if probabilidade > 0.7:
                print("💡 Recomendação: Consultar médico especialista")
            elif probabilidade > 0.3:
                print("💡 Recomendação: Manter acompanhamento regular")
            else:
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
    """Função principal com switch para Regressão Linear"""
    print("="*60)
    print("SISTEMA DE PREDIÇÃO DE DIABETES - REGRESSÃO LINEAR")
    print("="*60)
    print("\nOpções disponíveis:")
    print("1. Criar e treinar novo modelo de Regressão Linear")
    print("2. Fazer previsão com dados inseridos")
    print("3. Sair")
    
    model = None
    scaler = None
    
    while True:
        try:
            opcao = int(input("\nEscolha uma opção (1-3): "))
            
            if opcao == 1:
                print("\n" + "="*50)
                print("CRIANDO NOVO MODELO - REGRESSÃO LINEAR")
                print("="*50)
                model, scaler, X_test_scaled, y_test, y_pred, y_pred_binary = criar_novo_modelo_regressao()
                if model is not None:
                    avaliar_modelo_regressao(model, X_test_scaled, y_test, y_pred, y_pred_binary)
                    
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