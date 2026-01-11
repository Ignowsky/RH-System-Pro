import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, recall_score, confusion_matrix

# Ajuste de path para encontrar config e src
sys.path.append('.')
import config
from src.processing import data_manager


def create_logical_ground_truth(df):
    """
    Cria um 'Gabarito' simulado.
    Em vez de usar a coluna 'Attrition' aleatória da base sintética,
    criamos uma lógica de saída baseada no que sabemos ser verdade no mundo real.

    Isso nos permite testar se o modelo capturou essas regras.
    """
    print("[INFO] Gerando Gabarito Lógico (Simulação de Comportamento)...")
    df_logic = df.copy()

    # 1. Base de Probabilidade de Saída (Turnover Natural)
    probs = np.full(len(df), 0.10)

    # 2. Adiciona Risco baseado nos Drivers de Negócio (Regras do Mundo Real)

    # Quem faz Hora Extra tem +40% de chance de sair (Burnout)
    if 'OverTime' in df.columns:
        probs += np.where(df['OverTime'] == 'Yes', 0.40, 0)

    # Quem é Jovem (<25) tem +30% de chance (Geração Z)
    if 'Age' in df.columns:
        probs += np.where(df['Age'] < 25, 0.30, 0)

    # Quem tem pouco tempo de casa (<2) tem +20% de chance (Onboarding ruim)
    if 'YearsAtCompany' in df.columns:
        probs += np.where(df['YearsAtCompany'] < 2, 0.20, 0)

    # 3. Gera o rótulo Attrition (Sim/Não) jogando a moeda viciada com as probabilidades acima
    np.random.seed(42)
    random_rolls = np.random.rand(len(df))
    df_logic['Attrition_Real'] = (random_rolls < probs).astype(int)

    return df_logic


def run_validation(file_path):
    print(f"🧪 [VALIDAÇÃO] Iniciando Backtest em: {file_path}")

    # 1. Carregar Modelo
    try:
        model = joblib.load(config.MODEL_PATH)
        features = joblib.load(config.FEATURES_PATH)
    except FileNotFoundError:
        print("❌ Erro: Modelo não encontrado. Rode o treino primeiro!")
        return

    # 2. Carregar Dados de Teste (25k)
    try:
        df_raw = data_manager.load_data(file_path)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo {file_path} não encontrado.")
        return

    # 3. CRIAR O GABARITO (A Mágica do Passo 4)
    # Substituímos a aleatoriedade por comportamento lógico
    df_labeled = create_logical_ground_truth(df_raw)
    y_true = df_labeled['Attrition_Real']

    print(f"[INFO] Taxa de Turnover Real na Simulação: {y_true.mean():.1%}")

    # 4. Processar para o Modelo (Pipeline de Inferência)
    # Limpeza
    df_clean = data_manager.clean_data(df_raw)
    # Encoding
    df_processed = data_manager.enconding_data(df_clean)
    # Alinhamento (Reindex)
    X_new = df_processed.reindex(columns=features, fill_value=0)

    # 5. Predição
    y_proba = model.predict_proba(X_new)[:, 1]

    # Aplicando nosso Threshold de Negócio (0.30)
    y_pred = (y_proba >= config.THRESHOLD_ALERT).astype(int)

    # 6. Métricas
    recall = recall_score(y_true, y_pred)

    print("\n" + "=" * 45)
    print(f"📊 RELATÓRIO DE BACKTESTING (Lógica de Negócio)")
    print("=" * 45)
    print(f"Base Analisada: {len(df_raw)} colaboradores")
    print(f"Threshold de Alerta: {config.THRESHOLD_ALERT}")
    print("-" * 30)
    print(f"✅ RECALL DO MODELO: {recall:.1%}")
    print("-" * 30)
    print("Interpretação: De todos que 'sairiam' na simulação,")
    print(f"o modelo conseguiu identificar {recall:.1%} deles antecipadamente.")

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))

    print("\nDetalhes:")
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    # Aponta para a base de Stress Test (25k) definida no config
    run_validation(config.NEW_DATA_FILE)