import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("Импорты выполнены успешно")

DATASETS = {
    "all_v3":        "dataset_all(3).csv",
    "all_v4":        "dataset_all(4).csv",
    "gaze_only_v3":  "dataset_gaze_only(3).csv",
    "gaze_only_v4":  "dataset_gaze_only(4).csv",
    "head_only_v3":  "dataset_head_only(3).csv",
    "head_only_v4":  "dataset_head_only(4).csv",
}

print("Датасеты для обучения:")
for k, v in DATASETS.items():
    print(f"  {k:20s} -> {v}")


DIRECTION_MAP = {
    "center": 0, "up": 1, "down": 2, "left": 3, "right": 4,
    "left up": 5, "right up": 6, "left down": 7, "right down": 8,
    "blink": 9, "not detected": 10,
}
TRIGGER_MAP = {"gaze": 0, "head_pose": 1, "gaze_and_head": 2}
DROP_COLS = ["source_file", "event_timestamp", "current_status"]


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    for c in [col for col in df.columns if col.endswith("_direction")]:
        df[c] = df[c].map(DIRECTION_MAP)
    if "cheating_trigger" in df.columns:
        df["cheating_trigger"] = df["cheating_trigger"].map(TRIGGER_MAP)

    y = df["label"].astype(int)
    X = df.drop(columns=["label"]).apply(pd.to_numeric, errors="coerce")

    null_ratio = X.isnull().mean()
    X = X[null_ratio[null_ratio < 0.8].index]

    gaze_num_cols = [c for c in X.columns
                     if c.startswith("gaze_") and not c.endswith("_direction")
                     and not c.endswith("_timestamp")]
    gaze_dir_cols = [c for c in X.columns
                     if c.startswith("gaze_") and c.endswith("_direction")]
    head_num_cols = [c for c in X.columns
                     if c.startswith("head_") and not c.endswith("_direction")
                     and not c.endswith("_timestamp") and not c.endswith("_is_suspicious")]
    head_dir_cols = [c for c in X.columns
                     if c.startswith("head_") and c.endswith("_direction")]
    head_susp_cols = [c for c in X.columns if c.endswith("_is_suspicious")]

    feats = {}

    base_cols = [
        "suspicious_actions", "cheating_trigger",
        "gaze_trigger_count", "head_trigger_count",
        "calib_horizontal_ratio", "calib_vertical_ratio",
        "calib_head_neutral_yaw", "calib_head_neutral_pitch",
    ]
    for c in base_cols:
        if c in X.columns:
            feats[c] = X[c]

    if gaze_num_cols:
        gn = X[gaze_num_cols]
        feats["gaze_num_mean"] = gn.mean(axis=1)
        feats["gaze_num_std"]  = gn.std(axis=1)
        feats["gaze_num_max"]  = gn.max(axis=1)
        feats["gaze_num_min"]  = gn.min(axis=1)

        td_cols = [c for c in gaze_num_cols if "total_deviation" in c]
        if td_cols:
            td = X[td_cols]
            feats["gaze_total_dev_mean"] = td.mean(axis=1)
            feats["gaze_total_dev_std"]  = td.std(axis=1)
            feats["gaze_total_dev_max"]  = td.max(axis=1)

        ac_cols = [c for c in gaze_num_cols if "angle_from_calibrated_center" in c]
        if ac_cols:
            ac = X[ac_cols]
            feats["gaze_calib_angle_mean"] = ac.mean(axis=1)
            feats["gaze_calib_angle_std"]  = ac.std(axis=1)

    if gaze_dir_cols:
        gd = X[gaze_dir_cols]
        n_frames = gd.notna().sum(axis=1).clip(lower=1)
        feats["gaze_dir_non_center_ratio"] = (gd != 0).sum(axis=1) / n_frames
        feats["gaze_dir_up_ratio"]         = (gd == 1).sum(axis=1) / n_frames
        feats["gaze_dir_down_ratio"]       = (gd == 2).sum(axis=1) / n_frames
        feats["gaze_dir_blink_ratio"]      = (gd == 9).sum(axis=1) / n_frames
        feats["gaze_n_frames"]             = n_frames

    if head_num_cols:
        hn = X[head_num_cols]
        feats["head_num_mean"] = hn.mean(axis=1)
        feats["head_num_std"]  = hn.std(axis=1)

        yaw_cols = [c for c in head_num_cols if "yaw_deviation" in c]
        pitch_cols = [c for c in head_num_cols if "pitch_deviation" in c]
        if yaw_cols:
            yaw = X[yaw_cols].abs()
            feats["head_yaw_abs_mean"] = yaw.mean(axis=1)
            feats["head_yaw_abs_max"]  = yaw.max(axis=1)
        if pitch_cols:
            pitch = X[pitch_cols].abs()
            feats["head_pitch_abs_mean"] = pitch.mean(axis=1)
            feats["head_pitch_abs_max"]  = pitch.max(axis=1)

    if head_dir_cols:
        hd = X[head_dir_cols]
        hn_frames = hd.notna().sum(axis=1).clip(lower=1)
        feats["head_dir_non_center_ratio"] = (hd != 0).sum(axis=1) / hn_frames
        feats["head_dir_up_ratio"]         = (hd == 1).sum(axis=1) / hn_frames
        feats["head_dir_down_ratio"]       = (hd == 2).sum(axis=1) / hn_frames
        feats["head_n_frames"]             = hn_frames

    if head_susp_cols:
        hs = X[head_susp_cols]
        feats["head_is_suspicious_ratio"] = hs.mean(axis=1)
        feats["head_is_suspicious_any"]   = (hs == 1).any(axis=1).astype(int)

    X_feat = pd.DataFrame(feats)
    return X_feat, y


print("Функция prepare_features определена")


loaded = {}

for ds_key, ds_path in DATASETS.items():
    try:
        df = pd.read_csv(ds_path, low_memory=False)
        X_feat, y = prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, y, test_size=0.2, random_state=42, stratify=y
        )
        prep = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        X_tr = prep.fit_transform(X_train)
        X_te = prep.transform(X_test)

        loaded[ds_key] = {
            "X_feat":   X_feat,
            "y":        y,
            "X_tr":     X_tr,
            "X_te":     X_te,
            "y_train":  y_train,
            "y_test":   y_test,
            "prep":     prep,
        }
        print(f"[OK] {ds_key:20s} | rows={len(df):>5} | "
              f"features={X_feat.shape[1]:>3} | "
              f"label={y.value_counts().to_dict()}")
    except FileNotFoundError:
        print(f"[SKIP] {ds_key:20s} — файл не найден: {ds_path}")

print(f"\nЗагружено датасетов: {len(loaded)}")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ALL_RESULTS = []


def run_one(ds_key, estimator, param_grid):
    d = loaded[ds_key]
    t0 = time.time()

    gs = GridSearchCV(estimator, param_grid, cv=cv,
                      scoring="f1", n_jobs=-1, refit=True)
    gs.fit(d["X_tr"], d["y_train"])
    elapsed = int(time.time() - t0)

    m   = gs.best_estimator_
    yp  = m.predict(d["X_te"])
    ypr = m.predict_proba(d["X_te"])[:, 1]

    return {
        "Dataset":      ds_key,
        "CV_F1":        round(gs.best_score_, 4),
        "Test_F1":      round(f1_score(d["y_test"],  yp,  zero_division=0), 4),
        "Recall":       round(recall_score(d["y_test"],    yp,  zero_division=0), 4),
        "Precision":    round(precision_score(d["y_test"], yp,  zero_division=0), 4),
        "ROC_AUC":      round(roc_auc_score(d["y_test"],   ypr), 4),
        "Balanced_Acc": round(balanced_accuracy_score(d["y_test"], yp), 4),
        "Best_Params":  gs.best_params_,
        "Time_s":       elapsed,
        "_model":       m,
        "_yp":          yp,
        "_y_test":      d["y_test"],
        "_X_feat_cols": list(d["X_feat"].columns),
    }


def run_algorithm(algo_name, estimator, param_grid):
    print("=" * 70)
    print(f"  АЛГОРИТМ: {algo_name}")
    print("=" * 70)

    algo_rows = []
    for ds_key in loaded:
        import copy
        est_copy = copy.deepcopy(estimator)
        print(f"  [{ds_key}] ...", end=" ", flush=True)
        row = run_one(ds_key, est_copy, param_grid)
        row["Model"] = algo_name
        algo_rows.append(row)
        print(f"готово [{row['Time_s']}s]  F1={row['Test_F1']:.4f}")

    display_cols = ["Dataset", "CV_F1", "Test_F1", "Recall",
                    "Precision", "ROC_AUC", "Balanced_Acc"]
    tbl = (pd.DataFrame(algo_rows)[display_cols]
             .sort_values("Test_F1", ascending=False)
             .reset_index(drop=True))
    tbl.index += 1

    print(f"\n  Результаты для {algo_name} (↓ по F1):")
    print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))

    best_row = algo_rows[0]
    best_row = sorted(algo_rows, key=lambda r: r["Test_F1"], reverse=True)[0]
    yp = best_row["_yp"]
    yt = best_row["_y_test"]
    print(f"\n  Лучший датасет: {best_row['Dataset']}  (F1={best_row['Test_F1']:.4f})")
    print(f"  Лучшие параметры: {best_row['Best_Params']}")
    print("\n  Classification Report (лучший датасет):")
    print(classification_report(yt, yp,
          target_names=["Норма (0)", "Списывание (1)"], digits=4))
    cm = confusion_matrix(yt, yp)
    print("  Confusion Matrix:")
    print(f"                  Предсказано 0  Предсказано 1")
    print(f"  Реальный кл. 0  {cm[0,0]:>13}  {cm[0,1]:>13}")
    print(f"  Реальный кл. 1  {cm[1,0]:>13}  {cm[1,1]:>13}")

    bm = best_row["_model"]
    if hasattr(bm, "feature_importances_"):
      fi = bm.feature_importances_
      cols = list(best_row["_X_feat_cols"])

      if len(fi) != len(cols):
          cols = cols[:len(fi)]

      imp = pd.Series(fi, index=cols).sort_values(ascending=False).head(5)
      print("\n  Топ-5 важных фичей:")
      for fn, fv in imp.items():
          print(f"    {fn:<45} {fv:.4f}")

    ALL_RESULTS.extend(algo_rows)
    print()


print("Служебные функции определены")


run_algorithm(
    "LogisticRegression",
    LogisticRegression(max_iter=2000, random_state=42),
    {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "saga"]},
)


run_algorithm(
    "DecisionTree",
    DecisionTreeClassifier(random_state=42),
    {"max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10], "criterion": ["gini", "entropy"]},
)


run_algorithm(
    "RandomForest",
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {"n_estimators": [100, 300], "max_depth": [10, 20, None], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
)


run_algorithm(
    "ExtraTrees",
    ExtraTreesClassifier(random_state=42, n_jobs=-1),
    {"n_estimators": [100, 300], "max_depth": [10, 20, None], "min_samples_split": [2, 5]},
)


run_algorithm(
    "GradientBoosting",
    GradientBoostingClassifier(random_state=42),
    {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5], "subsample": [0.8, 1.0]},
)


run_algorithm(
    "AdaBoost",
    AdaBoostClassifier(random_state=42),
    {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 2.0]},
)


run_algorithm(
    "KNN",
    KNeighborsClassifier(n_jobs=-1),
    {"n_neighbors": [3, 5, 9, 15, 21], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
)


run_algorithm(
    "SVM",
    SVC(probability=True, random_state=42),
    {"C": [0.1, 1, 10, 100], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
)


run_algorithm(
    "XGBoost",
    XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss", verbosity=0),
    {"n_estimators": [100, 200, 300], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5, 7], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0], "min_child_weight": [1, 3]},
)


run_algorithm(
    "LightGBM",
    LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1),
    {"n_estimators": [100, 200, 300], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [-1, 5, 10], "num_leaves": [31, 63, 127], "subsample": [0.8, 1.0], "min_child_samples": [5, 20]},
)


summary_cols = ["Dataset", "Model", "CV_F1", "Test_F1",
                "Recall", "Precision", "ROC_AUC", "Balanced_Acc"]

df_all = pd.DataFrame(ALL_RESULTS)[summary_cols].sort_values(
    "Test_F1", ascending=False
).reset_index(drop=True)
df_all.index += 1

print("=" * 80)
print("ИТОГОВАЯ ТАБЛИЦА: все датасеты × все алгоритмы (по Test F1)")
print("=" * 80)
print(df_all.to_string(float_format=lambda x: f"{x:.4f}"))


best_per_algo = (
    df_all
    .groupby("Model", sort=False)
    .apply(lambda g: g.sort_values("Test_F1", ascending=False).iloc[0])
    .reset_index(drop=True)
    [["Model", "Dataset", "Test_F1", "ROC_AUC", "Balanced_Acc"]]
    .sort_values("Test_F1", ascending=False)
    .reset_index(drop=True)
)
best_per_algo.index += 1

print("=" * 65)
print("ЛУЧШИЙ ДАТАСЕТ ДЛЯ КАЖДОГО АЛГОРИТМА (по Test F1)")
print("=" * 65)
print(best_per_algo.to_string(float_format=lambda x: f"{x:.4f}"))


df_all.to_csv("results_all.csv", index=False)
best_per_algo.to_csv("results_best_per_algo.csv", index=False)
print("Сохранено:")
print("  results_all.csv          — полная сводная таблица")
print("  results_best_per_algo.csv — лучший датасет на алгоритм")