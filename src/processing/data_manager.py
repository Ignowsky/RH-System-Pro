import pandas as pd


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