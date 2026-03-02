from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Raiz do projeto (mlet_5/)
BASE_DIR = Path(__file__).resolve().parents[2]

# Pastas principais
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
REFINED_DIR = DATA_DIR / "refined"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

SRC_DIR = BASE_DIR / "src"

# Arquivos padrão (paths completos)
RAW_XLSX_PATH = RAW_DIR / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
INTERIM_DATASET_PATH = INTERIM_DIR / "dataset_concatenado.csv"
REFINED_DATASET_PATH = REFINED_DIR / "dados_modelo.csv"

MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"

FEATURE_COLUMNS = [
    "FASE","IDADE","GENERO","ANO_INGRESSO","INSTITUICAO_ENSINO",
    "IAA","IEG","IPS","IDA","IND_PV","IPP","NOTA_MATEM","NOTA_PORT","INDE_ATUAL"]

@dataclass(frozen=True)
class SplitConfig:
    train_years: tuple[int, ...] = (2022, 2023)
    test_year: int = 2024
    target_col: str = "POSSUI_DEFASAGEM"
    drop_cols: tuple[str, ...] = ("ANO", "IAN")

@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 200
    random_state: int = 42
    class_weight: str = "balanced"

def ensure_dirs() -> None:
    """Garante que as pastas essenciais existam."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    REFINED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)