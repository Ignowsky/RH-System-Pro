import sys
import pandas as pd
import joblib

# Ajuste de path para encontrar os módulos irmãos e config
sys.path.append('.')
import config
from src.processing import data_manager


def load_artifacts():
    """Carrega o modelo treinado e a lista de features (colunas)."""
    print(f"[INFO] Carregando artefatos do modelo...")
    try:
        model = joblib.load(config.MODEL_PATH)
        features = joblib.load(config.FEATURES_PATH)
        return model, features
    except FileNotFoundError:
        print(f"❌ [ERRO] Artefatos não encontrados em {config.MODEL_DIR}.")
        print("Dica: Rode o pipeline de treino (src/train/train_pipeline.py) primeiro.")
        sys.exit(1)


def generate_report(df_raw, probs):
    """Gera o dataframe final mantendo as colunas originais de identificação."""

    # Criamos uma cópia dos dados originais para não perder Nome/ID
    report = df_raw.copy()

    # Adicionamos a probabilidade calculada
    report['Probabilidade'] = probs

    # Classificação baseada no Threshold do Config (0.30)
    report['Nivel_Risco'] = report['Probabilidade'].apply(
        lambda p: '🔴 CRÍTICO' if p >= 0.70 else ('🟡 ALERTA' if p >= config.THRESHOLD_ALERT else '🟢 BAIXO')
    )

    return report


def make_prediction(file_path):
    print(f"🔮 [INFERÊNCIA] Iniciando processamento para: {file_path}")

    # 1. Carregar Artefatos
    model, train_features = load_artifacts()

    # 2. Carregar Novos Dados
    try:
        df_raw = data_manager.load_data(file_path)  # Alterado para load_data conforme seu data_manager
    except FileNotFoundError:
        print(f"❌ [ERRO] Arquivo de dados não encontrado: {file_path}")
        return

    # 3. Limpeza (Passo 1 do Data Manager)
    # Nota: clean_data remove EmployeeNumber, então usamos df_raw depois para o relatório
    df_clean = data_manager.clean_data(df_raw)

    # 4. Encoding (Passo 2 do Data Manager)
    # Transforma texto em número (get_dummies) e mapeia binários
    df_processed = data_manager.enconding_data(df_clean)

    # 5. ALINHAMENTO DE COLUNAS (O Pulo do Gato) 🐈
    # O df_processed pode ter colunas diferentes do treino (ex: faltar um departamento).
    # O reindex força o df a ter EXATAMENTE as colunas que o modelo aprendeu.
    # fill_value=0 preenche com 0 onde não houver a coluna.
    X_new = df_processed.reindex(columns=train_features, fill_value=0)

    # 6. Predição
    print(f"[INFO] Calculando probabilidades para {len(X_new)} colaboradores...")
    probs = model.predict_proba(X_new)[:, 1]

    # 7. Gerar Relatório Final
    # Usamos o df_raw original aqui para ter acesso ao EmployeeNumber/Name que foi dropado no clean_data
    df_report = generate_report(df_raw, probs)

    # 8. Salvar
    output_name = f"Relatorio_Risco_Final.csv"
    output_path = config.DATASET_DIR / output_name

    # Salvamos apenas as colunas mais relevantes para o RH + Risco
    cols_to_save = ['EmployeeNumber', 'Age', 'Department', 'OverTime', 'Nivel_Risco', 'Probabilidade']
    # Verifica se as colunas existem antes de filtrar (para evitar erro em bases diferentes)
    cols_final = [c for c in cols_to_save if c in df_report.columns]

    df_report[cols_final].sort_values('Probabilidade', ascending=False).to_csv(output_path, index=False)

    print(f"✅ [SUCESSO] Relatório salvo em: {output_path}")
    print("\n--- Resumo dos Riscos Detectados ---")
    print(df_report['Nivel_Risco'].value_counts())
    print("-" * 30)


if __name__ == '__main__':
    # Aponta para a base nova (sintética) definida no config
    input_data = config.NEW_DATA_FILE

    make_prediction(input_data)