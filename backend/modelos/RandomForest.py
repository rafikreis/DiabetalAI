import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def criar_novo_modelo_random_forest():
    """Função para criar e treinar um novo modelo de Random Forest"""
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

    print("\nTreinando modelo de Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    print("Modelo de Random Forest treinado com sucesso!")

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba

def avaliar_modelo_random_forest(model, X_test_scaled, y_test, y_pred, y_pred_proba):
    """Função para avaliar o modelo de Random Forest e gerar gráficos"""

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MÉTRICAS DE CLASSIFICAÇÃO - RANDOM FOREST")
    print("="*50)
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred))

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc_roc:.4f}")

    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
    plt.xlabel('Predito', fontsize=12)
    plt.ylabel('Real', fontsize=12)

    plt.subplot(2, 3, 2)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.4f})', linewidth=2, color='red')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('Curva ROC', fontsize=14, fontweight='bold')
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='Não Diabético', bins=20, color='blue', edgecolor='black')
    plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='Diabético', bins=20, color='red', edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold = 0.5')
    plt.title('Distribuição das Probabilidades', fontsize=14, fontweight='bold')
    plt.xlabel('Probabilidade Prevista', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    feature_importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='green', alpha=0.7, edgecolor='black')
    plt.title('Importância das Features - Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Importância', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    metrics = ['Acurácia', 'Precisão', 'Recall', 'AUC-ROC']
    values = [accuracy, precision, recall, auc_roc]
    colors = ['blue', 'green', 'orange', 'red']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Comparação de Métricas', fontsize=14, fontweight='bold')
    plt.ylabel('Valor', fontsize=12)
    plt.ylim(0, 1)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.subplot(2, 3, 6)

    n_trees_to_plot = min(10, len(model.estimators_))
    tree_importances = []
    
    for i, tree in enumerate(model.estimators_[:n_trees_to_plot]):
        tree_importances.append(tree.feature_importances_[0]) 
    
    plt.plot(range(1, n_trees_to_plot + 1), tree_importances, 'o-', linewidth=2, markersize=8)
    plt.title('Importância da Glucose nas Primeiras Árvores', fontsize=14, fontweight='bold')
    plt.xlabel('Número da Árvore', fontsize=12)
    plt.ylabel('Importância da Glucose', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*50)
    print("IMPORTÂNCIA DAS FEATURES - RANDOM FOREST")
    print("="*50)
    importance_df_sorted = importance_df.sort_values('Importance', ascending=False)
    print(importance_df_sorted.to_string(index=False))

def fazer_previsao_random_forest(model, scaler):
    """Função para fazer previsões com dados inputados pelo usuário"""
    print("\n" + "="*50)
    print("PREVISÃO PARA NOVOS DADOS - RANDOM FOREST")
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

            probabilidade = model.predict_proba(exemplo_dados_scaled)[0][1]
            
            print(f"\n" + "="*40)
            print("RESULTADO DA PREVISÃO - RANDOM FOREST")
            print("="*40)
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
    """Função principal com switch para Random Forest"""
    print("="*60)
    print("SISTEMA DE PREDIÇÃO DE DIABETES - RANDOM FOREST")
    print("="*60)
    print("\nOpções disponíveis:")
    print("1. Criar e treinar novo modelo de Random Forest")
    print("2. Fazer previsão com dados inseridos")
    print("3. Sair")
    
    model = None
    scaler = None
    
    while True:
        try:
            opcao = int(input("\nEscolha uma opção (1-3): "))
            
            if opcao == 1:
                print("\n" + "="*50)
                print("CRIANDO NOVO MODELO - RANDOM FOREST")
                print("="*50)
                model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba = criar_novo_modelo_random_forest()
                if model is not None:
                    avaliar_modelo_random_forest(model, X_test_scaled, y_test, y_pred, y_pred_proba)
                    
                    fazer_pred = input("\nDeseja fazer previsões para novos dados? (s/n): ").lower()
                    if fazer_pred == 's':
                        fazer_previsao_random_forest(model, scaler)
                    else:
                        print("Previsões para novos dados cancelada.")
                
            elif opcao == 2:
                if model is None:
                    print("\n⚠️  Primeiro você precisa treinar um modelo (Opção 1)!")
                    continue
                    
                print("\n" + "="*50)
                print("PREVISÃO COM DADOS INSERIDOS")
                print("="*50)
                fazer_previsao_random_forest(model, scaler)
                
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