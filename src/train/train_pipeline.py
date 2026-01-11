import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Ajustando o path para encontrar os módulos irmãos
sys.path.append(".")
import config
from src.processing import data_manager


def load_training():
    """
    Carrega o dataset de treino.
    """
    print(f"[INFO] Carregando o dataset de treino...")
    df = data_manager.load_data(config.TRAIN_DATA_FILE)
    return df


def cleaning_training(df):
    """
    Limpa o dataset (Remove colunas inúteis).
    """
    print(f"[INFO] Limpando o dataset de treino...")
    # CORREÇÃO: Chama a sua função específica clean_data
    df_clean = data_manager.clean_data(df)
    return df_clean


def encoding_training(df):
    """
    Realiza o encoding e separa X e y.
    """
    print(f"[INFO] Realizando encoding e separando X/y...")

    # CORREÇÃO: Chama a sua função específica enconding_data
    # Ela já retorna o dataframe com dummies e Attrition mapeado para 0/1
    df_processed = data_manager.enconding_data(df)

    # Agora separamos Features e Target
    X = df_processed.drop('Attrition', axis=1)
    y = df_processed['Attrition']

    return X, y


def split_training(X, y):
    """
    Divide os dados em treino e teste.
    """
    print(f"[INFO] Dividindo os dados de treino...")

    # Lembrando: train_test_split retorna 4 variáveis
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    return X_train, y_train


def training_model(X_train, y_train):
    """
    Treina o modelo e salva os artefatos.
    """
    print(f"[INFO] Treinando o dataset...")

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=config.RANDOM_STATE
    )

    model.fit(X_train, y_train)

    # Salvando os artefatos
    features = X_train.columns.tolist()

    # Criando a pasta models se não existir
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(features, config.FEATURES_PATH)

    print(f"✅ [SUCESSO] Modelo treinado e salvo em: {config.MODEL_PATH}")
    print(f"📝 Colunas salvas: {len(features)} features")

    return model


# =====================================
# O MAESTRO (ORQUESTRAÇÃO)
# =====================================
def run_pipeline():
    print(f"[INFO] --- Iniciando Pipeline de Treinamento ---")

    # 1. Load
    df_raw = load_training()

    # 2. Clean (Chama clean_data do data_manager)
    df_clean = cleaning_training(df_raw)

    # 3. Encoding & Separate (Chama enconding_data do data_manager)
    X, y = encoding_training(df_clean)

    # 4. Split
    X_train, y_train = split_training(X, y)

    # 5. Train & Save
    training_model(X_train, y_train)


if __name__ == "__main__":
    run_pipeline()