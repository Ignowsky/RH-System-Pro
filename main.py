# Importando as Libs utilizadas no projeto como um todo
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score


# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================
DATA_PATH = 'HR-Employee-Attrition.csv'
MODEL_SAVE_PATH = 'turnover_model.pkl'
COLS_SAVE_PATH = 'model_columns.pkl'
THRESHOLD_ALERT = 0.30

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


# ==========================
# FUNÇÕES DE MODELAGEM
# ==========================

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

    # 3.1. Separar X e y
    X = df_processed.drop('Attrition', axis=1)
    y = df_processed['Attrition']

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.25, random_state = 42, stratify = y
    )

    # 5. Treinar
    model = train_model(X_train, y_train)

    # 6. Avaliar (Usando o threshold de 30%)
    evaluate_model(model, X_test, y_test, threshold = THRESHOLD_ALERT)

    # 7. Salvar Artefatos (Modelo e nome das colunas)
    # Importante salvar as colunas para garantir a ordem do deploy
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(X_train.columns.tolist(), COLS_SAVE_PATH)
    print(f"[SUCESSO] Modelo salvo em {MODEL_SAVE_PATH}")
