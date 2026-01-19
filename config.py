import pathlib
import os

# Raiz do projeto
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

# Caminho dos dados
DATASET_DIR = PACKAGE_ROOT / 'data'
TRAIN_DATA_FILE = DATASET_DIR / 'raw' / 'HR-Employee-Attrition-50.csv'
NEW_DATA_FILE = DATASET_DIR / 'new' / 'HR-Employee-Attrition-50.csv'

# Caminhos de modelos
MODEL_DIR = PACKAGE_ROOT / 'models'
MODEL_PATH = MODEL_DIR / 'xgboost_model_v2.pkl'
FEATURES_PATH = MODEL_DIR / 'features.pkl'

# Hiperparâmetros de Negócio
THRESHOLD_ALERT = 0.20
RANDOM_STATE = 42
