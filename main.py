# Importando as Libs utilizadas no projeto como um todo
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score, precision_score
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================
DATA_PATH = 'data/new/HR-Employee-Attrition-50.csv'
MODEL_SAVE_PATH = 'turnover_model.pkl'
COLS_SAVE_PATH = 'model_columns.pkl'
THRESHOLD_ALERT = 0.35

# ==========================
# FUNÇÕES DE PROCESSAMENTO
# ==========================
def load_data(filepath):
    """
    Carrega o dataset completo, um arquivo CSV

    :param filepath:
    :return: Um dataframe pandas
    """
    print(f"[INFO] Carregando arquivo {filepath}")
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Recebe um dataset pandas e limpa determinadas colunas, colunas com a variância 0

    :param df:
    :return: df limpo
    """
    print(f'[INFO] Iniciando a limpeza do dataset {df}')
    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df_clean = df.drop(
        columns = [c for c in drop_cols if c in df.columns], errors='ignore'
    )
    return df_clean

def enconding_data(df):
    """
    Recebe um dataframe já limpo sem colunas com variância 0, e realizar o enconding completo das variaveis categoricas
    :param df:
    :return: df com enconding feito.
    """
    print(f'[INFO] Iniciando a enconding do dataset {df}')

    df_proc = df.copy()

    # Realizando o mapeamento Binário:
    if 'Attrition' in df_proc.columns:
        df_proc['Attrition'] = df_proc['Attrition'].map({'Yes': 1, 'No': 0})

    if 'OverTime' in df_proc.columns:
        df_proc['OverTime'] = df_proc['OverTime'].map({'Yes': 1, 'No': 0})

    # One-Hot Encondig (Get Dummies)
    df_final = pd.get_dummies(df_proc, drop_first=True)

    # Convertendo os valores bool para int
    df_final = df_final.astype(int)

    return df_final


def diagnose_data(df_processed):
    """
    Executa um check-up nos dados para garantir que o modelo não vai treinar no escuro.
    :param df_processed: DataFrame JÁ CODIFICADO (só números)
    """
    print("\n" + "=" * 40)
    print("[DIAGNÓSTICO] INICIANDO CHECK-UP DOS DADOS")
    print("=" * 40)

    # 1. Verifica se o Target está vivo
    target_dist = df_processed['Attrition'].value_counts(normalize=True)
    print(f"\n1. Distribuição do Target (Attrition):")
    print(target_dist)

    if len(target_dist) < 2:
        print("⚠️ ALERTA CRÍTICO: O Target só tem uma classe! O modelo vai falhar.")
        return  # Para a função aqui

    # 2. Verifica correlações com o Target (O segredo do sucesso)
    # Se todas forem perto de 0, a base é lixo aleatório.
    print(f"\n2. Top 5 Correlações com Attrition (Fatores de Risco):")
    correlations = df_processed.corr()['Attrition'].sort_values(ascending=False)
    # Remove a própria coluna Attrition e mostra as top 5
    print(correlations.drop('Attrition').head(5))

    # 3. Verifica se colunas vitais existem e têm variação
    cols_vitais = ['MonthlyIncome', 'Age', 'TotalWorkingYears']
    print(f"\n3. Checagem de Variáveis Vitais:")
    for col in cols_vitais:
        if col in df_processed.columns:
            corr = df_processed[col].corr(df_processed['Attrition'])
            print(f"   > {col}: Correlação {corr:.4f}")
        else:
            print(f"   > ❌ ERRO: A coluna {col} SUMIU do dataset!")

    print("\n" + "=" * 40 + "\n")


# ==========================
# FUNÇÕES DE MODELAGEM
# ==========================

def find_optimal_threshold(model, X_test, y_test):
    print("\n[TUNING] Buscando o melhor ponto de corte...")
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.3, 0.75, 0.05)  # Testa de 0.30 até 0.70

    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 50)

    best_f1 = 0
    best_thresh = 0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)  # Importe precision_score do sklearn.metrics
        f1 = 2 * (prec * rec) / (prec + rec)

        print(f"{t:.2f}       | {prec:.2f}       | {rec:.2f}       | {f1:.2f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print("-" * 50)
    print(f"🏆 Melhor Threshold sugerido (pelo F1-Score): {best_thresh:.2f}")
    return best_thresh

# Modelo Anterior utilizando DecisionTree
# Realizando a troca para o XGBOOST
'''
def train_model(X_train, y_train):
    """
    Treina o modelo de DecisionTree com os hiperparâmetros campeões
    :param df_final:
    :return: modelo de DecisionTree
    """

    print(f"[INFO] Treinando modelo DecisionTree (Max_depth = 5, min_samples_leaf = 50)")

    # Hiperparâmetros definidos no estudo anterior
    model = DecisionTreeClassifier(
        max_depth = 5,
        min_samples_leaf = 50,
        random_state = 42,
        class_weight = 'balanced'
    )

    model.fit(
        X_train,
        y_train
    )

    return model
'''

def train_model(X_train, y_train):
    """
    Treina o modelo do XGBOOST com os hiperparâmetros campeões
    :param df_final:
    :return: modelo de DecisionTree
    """
    print(f"[INFO] Treinando o modelo XGBoost (n_estimators = 1000, max_depth = 5)")
    count_neg = y_train.value_counts()[0]
    count_pos = y_train.value_counts()[1]
    ratio = count_neg / count_pos
    print(f"Ratio: {ratio:.2f}")
    model = XGBClassifier(
        scale_pos_weight = 1.85,
        learning_rate = 0.1,
        n_estimators = 600,
        max_depth = 5,
        min_child_weight = 6,
        gamma = 2,
        colsample_bytree = 0.7,
        max_delta_step = 1,
        random_state = 42
    )

    model.fit(X_train,y_train )

    return model


def evaluate_model(model, X_test, y_test, threshold = 0.5):
    """
    Avalia o modelo com base no threshold customizado
    :param model: modelo de DecisionTree,
    :param X_test: Features de test
    :param y_test: Target de test
    :param threshold: Valor de alerta definido
    :return:
    y_proba a probabilidade real do turnover dos colaboradores com base das features
    """

    print(f"[INFO] Avaliando o modelo com Threshold de {threshold:.2f}")

    # Probabilidades da classe 1 (Sair)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Aplicando a régua de corte
    y_pred_custom = (y_proba >= threshold).astype(int)

    # Métricas
    recall = recall_score(y_test, y_pred_custom)
    auc = roc_auc_score(y_test, y_proba)

    print("\n---------- Relatório de Perfomance ---------------")
    print(f"Threshold: {threshold:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"AUC: {auc:.2f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred_custom))

    return y_proba

def feature_importaces(model, X_train):
    """
        Exibe as variáveis mais importantes para o modelo XGBoost.
        Args:
        model: Modelo treinado (XGBClassifier)
        X_train: DataFrame usado no treino (para pegar os nomes das colunas)
    """

    importancia = pd.DataFrame({
        'Feature': X_train.columns,
        'Importancia': model.feature_importances_
    }).sort_values(by = 'Importancia', ascending = False)
    print(f"\n[DEBUG] Top 10 Variáveis mais importantes:")
    print(importancia.head())

    return importancia





def calculate_financial_impact(y_test, y_pred, replacement_cost=40000, intervention_cost=500):
    """
    Calcula o ROI do modelo baseado na Matriz de Confusão.

    Params:
    - replacement_cost: Custo médio de substituir um funcionário (R$ 40k)
    - intervention_cost: Custo da hora do gestor/RH para fazer a retenção (R$ 500)
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Cenário SEM Modelo: Você perde todos que sairiam (TP + FN)
    loss_without_model = (tp + fn) * replacement_cost

    # Cenário COM Modelo:
    # 1. Você salva os TP (assumindo que a intervenção funciona em 50% dos casos)
    saved_talent_value = tp * replacement_cost * 0.50

    # 2. Você gasta dinheiro intervindo nos alertas (TP + FP)
    cost_of_intervention = (tp + fp) * intervention_cost

    # 3. Você ainda perde os que não viu (FN)
    remaining_loss = fn * replacement_cost

    roi = saved_talent_value - cost_of_intervention

    print("\n" + "=" * 40)
    print("💰 RELATÓRIO DE IMPACTO FINANCEIRO")
    print("=" * 40)
    print(f"Pessoas em Risco Detectadas (TP): {tp}")
    print(f"Alarmes Falsos (Custo Operacional) (FP): {fp}")
    print("-" * 30)
    print(f"Custo da Perda Sem Modelo:      R$ {loss_without_model:,.2f}")
    print(f"Economia Gerada (ROI Estimado): R$ {roi:,.2f}")
    print("=" * 40)

    # Plot bonitinho da Matriz
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão - Threshold 0.35')
    plt.xlabel('Previsto (Modelo)')
    plt.ylabel('Real (Dados)')
    plt.show()






# ==========================
# EXECUÇÃO PRINCIPAL
# ==========================

if __name__ == '__main__':
    # 1. Carregar
    df = load_data(DATA_PATH)

    # 2. Limpar
    df = clean_data(df)

    # 3. Pré-processar
    df_processed = enconding_data(df)

    diagnose_data(df_processed)

    # 3.1. Separar X e y
    X = df_processed.drop('Attrition', axis=1)
    y = df_processed['Attrition']

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.25, random_state = 42, stratify = y
    )



    # 5. Treinar
    model = train_model(X_train, y_train)

    melhor_corte = find_optimal_threshold(model, X_test, y_test)

    feature_importaces(model, X_train)

    # 6. Avaliar (Usando o threshold de 30%)
    evaluate_model(model, X_test, y_test, threshold = THRESHOLD_ALERT)
    # No seu main, chame assim usando o corte escolhido:
    y_pred_final = (model.predict_proba(X_test)[:, 1] >= 0.35).astype(int)
    calculate_financial_impact(y_test, y_pred_final)

    # 7. Salvar Artefatos (Modelo e nome das colunas)
    # Importante salvar as colunas para garantir a ordem do deploy
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(X_train.columns.tolist(), COLS_SAVE_PATH)
    print(f"[SUCESSO] Modelo salvo em {MODEL_SAVE_PATH}")
