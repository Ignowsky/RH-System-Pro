import pathlib
import os

# Raiz do projeto
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

# Caminho dos dados
DATASET_DIR = PACKAGE_ROOT / 'data'
TRAIN_DATA_FILE = DATASET_DIR / 'raw' / 'HR-Employee-Attrition.csv'
NEW_DATA_FILE = DATASET_DIR / 'new' / 'Base_Funcionarios_Sintetica_Large_nova.csv'

# Caminhos de modelos
MODEL_DIR = PACKAGE_ROOT / 'models'
MODEL_PATH = MODEL_DIR / 'decision_tree_v1.pkl'
FEATURES_PATH = MODEL_DIR / 'features.pkl'

# Hiperparâmetros de Negócio
THRESHOLD_ALERT = 0.3
RANDOM_STATE = 42
