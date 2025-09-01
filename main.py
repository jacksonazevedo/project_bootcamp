import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_SEED = 42
N_SPLITS = 5
CALIB_METHOD = "isotonic"  # ou "sigmoid"

TRAIN_PATH = Path("bootcamp_train.csv")
TEST_PATH = Path("bootcamp_test.csv")

RENAME_LABELS = {
    "FDF (Falha Desgaste Ferramenta)": "FDF",
    "FDC (Falha Dissipacao Calor)": "FDC",
    "FP (Falha Potencia)": "FP",
    "FTE (Falha Tensao Excessiva)": "FTE",
    "FA (Falha Aleatoria)": "FA",
}

CAT_COLS = ["tipo"]
NUM_COLS = [
    "temperatura_ar",
    "temperatura_processo",
    "umidade_relativa",
    "velocidade_rotacional",
    "torque",
    "desgaste_da_ferramenta",
]
NUM_FOR_MODEL = [
    "temperatura_ar",
    "temperatura_processo",
    "velocidade_rotacional",
    "torque",
    "desgaste_da_ferramenta",
    "delta_temp",
    "potencia_mecanica",
]

# 0=sem_falha; 1..5 = tipos de falha
CLASS_NAMES = {
    0: "sem_falha",
    1: "desgaste_ferramenta",
    2: "dissipacao_calor",
    3: "falha_potencia",
    4: "falha_tensao",
    5: "falha_aleatoria",
}


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _to_binary_label(v):
    if pd.isna(v):
        return pd.NA
    if isinstance(v, (int, np.integer)):
        return int(v != 0)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return pd.NA
        return int(int(v) != 0)
    s = str(v).strip().lower()
    if s in {"nao", "não", "n", "no", "false", "falso", "0", "off", "0.0"}:
        return 0
    if s in {"sim", "s", "yes", "y", "true", "verdadeiro", "1", "on", "1.0"}:
        return 1
    try:
        f = float(s.replace(",", "."))
        return int(int(f) != 0)
    except Exception:
        return pd.NA


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["falha_maquina", "FDF", "FDC", "FP", "FTE", "FA"]
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(_to_binary_label).astype("Int64")
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "tipo" in df.columns:
        s = df["tipo"].astype(str).str.strip().str.upper()
        s = s.where(s.isin(["L", "M", "H"]), other=np.nan)
        df["tipo"] = s
    must = ["temperatura_ar", "temperatura_processo", "velocidade_rotacional", "desgaste_da_ferramenta"]
    for c in must:
        if c not in df.columns:
            raise KeyError(f"Coluna ausente: {c}")
    mask = (
        (df["temperatura_ar"] > 0)
        & (df["temperatura_processo"] > 0)
        & (df["velocidade_rotacional"] >= 0)
        & (df["desgaste_da_ferramenta"] >= 0)
    )
    removed = int((~mask).sum())
    if removed:
        logging.info(f"Linhas removidas (regras físicas): {removed}")
    return df[mask].copy()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    if {"temperatura_processo", "temperatura_ar"}.issubset(df.columns):
        df["delta_temp"] = df["temperatura_processo"] - df["temperatura_ar"]
    if {"torque", "velocidade_rotacional"}.issubset(df.columns):
        df["potencia_mecanica"] = df["torque"] * df["velocidade_rotacional"]
    if "umidade_relativa" in df.columns and df["umidade_relativa"].nunique(dropna=True) <= 2:
        df = df.drop(columns=["umidade_relativa"])
    return df


def build_y_multiclass(df: pd.DataFrame) -> pd.Series:
    fm = df["falha_maquina"].fillna(0).astype(int)
    fdf = df["FDF"].fillna(0).astype(int)
    fdc = df["FDC"].fillna(0).astype(int)
    fp = df["FP"].fillna(0).astype(int)
    fte = df["FTE"].fillna(0).astype(int)
    fa = df["FA"].fillna(0).astype(int)

    y = np.where(
        fm == 0,
        0,
        np.where(fdf == 1, 1, np.where(fdc == 1, 2, np.where(fp == 1, 3, np.where(fte == 1, 4, 5)))),
    )
    return pd.Series(y, name="target").astype(int)


def load_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Arquivos de treino/teste não encontrados.")

    tr = pd.read_csv(train_path).rename(columns=RENAME_LABELS)
    te = pd.read_csv(test_path).rename(columns=RENAME_LABELS)

    tr = feature_engineering(clean_features(tr))
    te = feature_engineering(clean_features(te))

    tr = normalize_labels(tr)
    needed = ["falha_maquina", "FDF", "FDC", "FP", "FTE", "FA"]
    before = len(tr)
    tr = tr.dropna(subset=needed).copy()
    removed = before - len(tr)
    if removed:
        logging.info(f"Linhas removidas por rótulos ausentes: {removed}")

    tr.reset_index(drop=True, inplace=True)
    te.reset_index(drop=True, inplace=True)

    y_bin = tr["falha_maquina"].astype(int)
    y_mc = build_y_multiclass(tr)

    feat_cols = [c for c in CAT_COLS if c in tr.columns] + [c for c in NUM_FOR_MODEL if c in tr.columns]
    Xtr = tr[feat_cols].copy().reset_index(drop=True)
    Xte = te[feat_cols].copy().reset_index(drop=True)
    return Xtr, y_bin.reset_index(drop=True), y_mc.reset_index(drop=True), Xte


def make_preprocess(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat = Pipeline(
        [("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    num = Pipeline([("imp", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        [("cat", cat, cat_cols), ("num", num, num_cols)], remainder="drop", verbose_feature_names_out=False
    )

def get_algorithm(name: str, class_weight=None):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            class_weight=class_weight,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if name == "dt":
        return DecisionTreeClassifier(max_depth=None, class_weight=class_weight, random_state=RANDOM_SEED)
    if name == "svm":
        return SVC(kernel="rbf", probability=True, class_weight=class_weight, random_state=RANDOM_SEED)
    raise ValueError("Algoritmo não suportado: rf, dt, svm")


def make_stage_A(pre: ColumnTransformer, algo: str) -> Pipeline:
    # Binário: class_weight específico para {0,1}
    base = get_algorithm(algo, class_weight={0: 1.0, 1: 8.0})
    calibrated = CalibratedClassifierCV(base, method=CALIB_METHOD, cv=3)
    return Pipeline([("pre", pre), ("clf", calibrated)])


def _sampling_targets(y_fail: pd.Series, multipliers: Dict[int, float]) -> Dict[int, int]:
    counts = y_fail.value_counts().to_dict()  # classes 1..5
    target = {}
    for cls, cnt in counts.items():
        m = multipliers.get(int(cls), 1.0)
        target[int(cls)] = int(max(cnt, round(cnt * m)))
    return target


def make_stage_B(pre: ColumnTransformer, algo: str, y_fail: pd.Series) -> ImbPipeline:
    # Multiclasse: usar "balanced" para 1..5 (evita erro do class_weight)
    multipliers = {1: 3.0, 2: 1.0, 3: 1.4, 4: 1.4, 5: 4.0}  # reforça 1 (desgaste) e 5 (aleatória)
    sampling = _sampling_targets(y_fail, multipliers)
    min_count = int(y_fail.value_counts().min())
    k_safe = max(1, min(3, min_count - 1))

    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_safe, sampling_strategy=sampling)
    base = get_algorithm(algo, class_weight="balanced")  # <- aqui é balanced para 5 classes
    calibrated = CalibratedClassifierCV(base, method=CALIB_METHOD, cv=3)
    return ImbPipeline([("pre", pre), ("smote", smote), ("clf", calibrated)])


def evaluate_binary_with_threshold(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, beta: float = 2.0) -> Tuple[float, dict, List[List[int]]]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]

    p, r, t = precision_recall_curve(y, proba)
    denom = np.clip(beta**2 * p + r, 1e-12, None)
    f = (1 + beta**2) * (p * r) / denom
    i = int(np.nanargmax(f))
    thr = 0.5 if i >= len(t) else float(t[i])

    y_pred = (proba >= thr).astype(int)
    rep = classification_report(y, y_pred, target_names=["sem_falha", "falha"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=[0, 1]).tolist()
    return thr, rep, cm


def evaluate_multiclass(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Tuple[dict, List[List[int]]]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    y_pred = cross_val_predict(pipe, X, y, cv=skf, method="predict", n_jobs=-1)
    rep = classification_report(
        y,
        y_pred,
        target_names=[CLASS_NAMES[i] for i in [1, 2, 3, 4, 5]],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y, y_pred, labels=[1, 2, 3, 4, 5]).tolist()
    return rep, cm


def infer_and_save(
    algo: str,
    pipeA: Pipeline,
    pipeB: ImbPipeline,
    Xte: pd.DataFrame,
    thr: float,
    out_prefix: str,
) -> str:
    p_fail = pipeA.predict_proba(Xte)[:, 1]
    probB = pipeB.predict_proba(Xte)

    # Guardrails numéricos
    p_fail = np.clip(p_fail, 0.0, 1.0)
    probB = np.clip(probB, 0.0, 1.0)
    row_sums = probB.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probB = probB / row_sums

    # Prob final: P(sem_falha)=1-P(falha); P(tipo)=P(falha)*P(tipo|falha)
    final_probs = np.zeros((len(Xte), 6))
    final_probs[:, 0] = 1.0 - p_fail
    for j, cls in enumerate(pipeB.classes_):
        final_probs[:, cls] = p_fail * probB[:, j]

    # Sanidade
    assert np.all(final_probs >= -1e-9) and np.all(final_probs <= 1 + 1e-9)
    assert np.allclose(final_probs.sum(axis=1), 1.0, atol=1e-6)

    # Decisão final com threshold do Estágio A
    pred_idx = np.where(p_fail >= thr, (final_probs[:, 1:]).argmax(axis=1) + 1, 0)
    pred_df = pd.DataFrame(final_probs, columns=[f"prob_{CLASS_NAMES[i]}" for i in range(6)])
    pred_df.insert(0, "pred_class_idx", pred_idx)
    pred_df.insert(1, "pred_class", [CLASS_NAMES[i] for i in pred_idx])
    pred_df.insert(2, "p_falha_binaria", p_fail)
    pred_df.insert(3, "threshold_usado", thr)

    out_csv = f"{out_prefix}_{algo}.csv"
    pred_df.to_csv(out_csv, index=False, float_format="%.8f")
    return out_csv

def main():
    setup_logging()
    Xtr, y_bin, y_mc, Xte = load_data(TRAIN_PATH, TEST_PATH)

    cat_cols = [c for c in CAT_COLS if c in Xtr.columns]
    num_cols = [c for c in NUM_FOR_MODEL if c in Xtr.columns and c not in cat_cols]
    pre = make_preprocess(cat_cols, num_cols)

    algos = ["rf", "dt", "svm"]
    results = {}

    for algo in algos:
        logging.info(f"=== {algo.upper()} ===")

        # Estágio A
        pipeA = make_stage_A(pre, algo)
        thr, repA, cmA = evaluate_binary_with_threshold(pipeA, Xtr, y_bin, beta=2.0)
        logging.info(f"Threshold ótimo (F2): {thr:.4f}")
        pipeA.fit(Xtr, y_bin)

        # Estágio B (apenas linhas com falha)
        idx_fail = y_mc.index[y_mc != 0]
        X_fail = Xtr.loc[idx_fail].reset_index(drop=True)
        y_fail = y_mc.loc[idx_fail].reset_index(drop=True)

        pipeB = make_stage_B(pre, algo, y_fail)
        repB, cmB = evaluate_multiclass(pipeB, X_fail, y_fail)
        pipeB.fit(X_fail, y_fail)

        # Inferência e persistência
        out_csv = infer_and_save(
            algo=algo,
            pipeA=pipeA,
            pipeB=pipeB,
            Xte=Xte,
            thr=thr,
            out_prefix="pred_test_calibrado",
        )
        joblib.dump(
            {"pipeA": pipeA, "pipeB": pipeB, "threshold": thr},
            f"modelo_hierarquico_{algo}_calibrado.pkl",
        )

        results[algo] = {
            "stageA": {"threshold": thr, "report": repA, "cm": cmA},
            "stageB": {"report": repB, "cm": cmB},
            "pred_csv": out_csv,
        }

    with open("metrics_algos_hierarquico_calibrado.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info("Concluído.")


if __name__ == "__main__":
    main()
