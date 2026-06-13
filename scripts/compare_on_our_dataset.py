import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


DATASETS = {
    "ortin":     "dataset_ortin.csv",
    "lamba":     "dataset_lamba.csv",
    "rahmawati": "dataset_rahmawati.csv",
    "singh_das": "dataset_singh_das.csv",
    "dang":      "dataset_dang.csv",
}

RANDOM_STATE  = 42
TEST_SIZE     = 0.2

# LSTM параметры
ORTIN_SEQUENCE_LEN = 49
ORTIN_DIRECTION_VOCAB = [
    "center", "up", "down", "left", "right",
    "up_left", "up_right", "down_left", "down_right", ""
]

DANG_WINDOW_SIZE = 10
LSTM_EPOCHS      = 30
LSTM_BATCH_SIZE  = 32



def _check_file(path):
    if not os.path.exists(path):
        print("  [ERROR] Файл не найден: {}".format(path))
        return False
    return True


def _print_metrics(name, y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    print("  Precision : {:.4f}".format(p))
    print("  Recall    : {:.4f}".format(r))
    print("  F1        : {:.4f}".format(f))
    print("  Подробно:")
    report = classification_report(y_true, y_pred, zero_division=0)
    for line in report.splitlines():
        print("    " + line)
    return {"model": name, "precision": p, "recall": r, "f1": f}


def _train_test_split_df(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size,
                            random_state=random_state, stratify=df["label"])

# ORTIN
def train_ortin(path):
    print("\n" + "="*60)
    print("Ortin  |  LSTM → Dense → Sigmoid")
    print("="*60)

    if not _check_file(path):
        return None

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path, dtype=str)
    if "label" not in df.columns:
        print("  [ERROR] Нет колонки label"); return None

    labels = df["label"].astype(int).values

    # Собираем direction-колонки в матрицу (N, ORTIN_SEQUENCE_LEN)
    dir_cols = ["gaze_{}_direction".format(i) for i in range(ORTIN_SEQUENCE_LEN)]
    dir_cols = [c for c in dir_cols if c in df.columns]
    seq_len   = len(dir_cols)

    if seq_len == 0:
        print("  [ERROR] Нет direction-колонок"); return None

    # Заполняем пропуски пустой строкой и кодируем
    seq_data = df[dir_cols].fillna("").values   # (N, seq_len)

    le = LabelEncoder()
    le.fit(ORTIN_DIRECTION_VOCAB)
    # Безопасное transform: неизвестные значения → индекс ""
    unk_idx = list(le.classes_).index("") if "" in le.classes_ else 0

    def safe_transform(arr_2d):
        out = np.full(arr_2d.shape, unk_idx, dtype=np.int32)
        for j in range(arr_2d.shape[1]):
            col = arr_2d[:, j]
            mask = np.isin(col, le.classes_)
            out[mask, j] = le.transform(col[mask])
        return out

    X = safe_transform(seq_data)   # (N, seq_len) int
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    vocab_size = len(le.classes_)

    # ── Архитектура ──────────────────────────────────────────────────────────
    inp  = keras.Input(shape=(seq_len,))
    emb  = keras.layers.Embedding(input_dim=vocab_size + 1,
                                   output_dim=16,
                                   mask_zero=False)(inp)
    lstm = keras.layers.LSTM(64)(emb)
    out  = keras.layers.Dense(1, activation="sigmoid")(lstm)
    model = keras.Model(inp, out)

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # Веса классов для дисбаланса
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    cw  = {0: 1.0, 1: (neg / pos) if pos > 0 else 1.0}
    print("  class_weight: {}".format(cw))

    model.fit(X_train, y_train,
              epochs=LSTM_EPOCHS,
              batch_size=LSTM_BATCH_SIZE,
              validation_split=0.1,
              class_weight=cw,
              verbose=1)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    return _print_metrics("Ortin", y_test, y_pred)

def train_lamba(path):
    print("\n" + "="*60)
    print("Lamba  |  ExtraTreesClassifier")
    print("="*60)

    if not _check_file(path):
        return None

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path, dtype=str)

    num_cols = [c for c in df.columns
                if c not in ("source_file", "event_timestamp", "label")
                and not c.endswith("_direction")
                and not c.endswith("_timestamp")]

    dir_cols = [c for c in df.columns if c.endswith("_direction")]

    X_num = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    if dir_cols:
        dir_data = df[dir_cols].fillna("").values
        all_dirs = np.unique(dir_data)
        le = LabelEncoder()
        le.fit(all_dirs)
        dir_encoded = np.zeros((len(df), len(dir_cols)), dtype=np.float32)
        for j, col in enumerate(dir_cols):
            col_vals = dir_data[:, j]
            mask = np.isin(col_vals, le.classes_)
            encoded = np.zeros(len(col_vals), dtype=np.float32)
            encoded[mask] = le.transform(col_vals[mask]).astype(np.float32)
            dir_encoded[:, j] = encoded
        X = np.hstack([X_num, dir_encoded])
    else:
        X = X_num

    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    cw  = {0: 1.0, 1: (neg / pos) if pos > 0 else 1.0}

    clf = ExtraTreesClassifier(
        n_estimators=200,
        max_features="sqrt",
        class_weight=cw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("  n_estimators=200, max_features=sqrt")
    print("  Признаков: {}  |  train: {}  |  test: {}".format(
        X.shape[1], len(y_train), len(y_test)))

    return _print_metrics("Lamba", y_test, y_pred)

def train_rahmawati(path):
    print("\n" + "="*60)
    print("Rahmawati  |  SVM (RBF kernel) + StandardScaler")
    print("="*60)

    if not _check_file(path):
        return None

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path, dtype=str)

    feat_cols = [c for c in df.columns
                 if c not in ("source_file", "event_timestamp", "label")]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    cw  = {0: 1.0, 1: (neg / pos) if pos > 0 else 1.0}

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf",
                       C=1.0,
                       gamma="scale",
                       class_weight=cw,
                       probability=True,
                       random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("  SVM kernel=rbf, C=1.0, gamma=scale")
    print("  Признаков: {}  |  train: {}  |  test: {}".format(
        X.shape[1], len(y_train), len(y_test)))

    return _print_metrics("Rahmawati", y_test, y_pred)

def train_singh_das(path):
    print("\n" + "="*60)
    print("Singh & Das  |  RandomForestClassifier (вместо w1-w4)")
    print("="*60)

    if not _check_file(path):
        return None

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path, dtype=str)

    feature_cols = [
        "t_gaze_outside_frames",
        "f_gaze_exits",
        "NR_gaze_reversals",
        "NH_head_suspicious",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print("  [ERROR] Отсутствуют колонки: {}".format(missing))
        return None

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    cw  = {0: 1.0, 1: (neg / pos) if pos > 0 else 1.0}

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight=cw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("  Признаки: {}".format(feature_cols))
    print("  n_estimators=200  |  train: {}  |  test: {}".format(
        len(y_train), len(y_test)))

    print("  Feature importances (≈ w1-w4):")
    for name, imp in zip(feature_cols, clf.feature_importances_):
        print("    {:30s}: {:.4f}".format(name, imp))

    return _print_metrics("Singh & Das", y_test, y_pred)

def train_dang(path):
    print("\n" + "="*60)
    print("Dang  |  LSTM → Dropout → Dense → Sigmoid")
    print("="*60)

    if not _check_file(path):
        return None

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path, dtype=str)

    feat_cols = [c for c in df.columns
                 if c not in ("source_file", "event_timestamp",
                               "window_start", "window_end",
                               "window_size", "label")]
    X_raw = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = df["label"].astype(int).values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_test_s  = scaler.transform(X_test_raw)


    X_train_lstm = X_train_s.reshape(-1, 1, X_train_s.shape[1])
    X_test_lstm  = X_test_s.reshape(-1, 1, X_test_s.shape[1])

    n_features = X_train_s.shape[1]

    inp     = keras.Input(shape=(1, n_features))
    lstm    = keras.layers.LSTM(64, return_sequences=False)(inp)
    dropout = keras.layers.Dropout(0.3)(lstm)
    out     = keras.layers.Dense(1, activation="sigmoid")(dropout)
    model   = keras.Model(inp, out)

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    cw  = {0: 1.0, 1: (neg / pos) if pos > 0 else 1.0}
    print("  class_weight: {}".format(cw))

    model.fit(X_train_lstm, y_train,
              epochs=LSTM_EPOCHS,
              batch_size=LSTM_BATCH_SIZE,
              validation_split=0.1,
              class_weight=cw,
              verbose=1)

    y_prob = model.predict(X_test_lstm, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    return _print_metrics("Dang", y_test, y_pred)

results = []

r = train_ortin(DATASETS["ortin"])
if r: results.append(r)

r = train_lamba(DATASETS["lamba"])
if r: results.append(r)

r = train_rahmawati(DATASETS["rahmawati"])
if r: results.append(r)

r = train_singh_das(DATASETS["singh_das"])
if r: results.append(r)

r = train_dang(DATASETS["dang"])
if r: results.append(r)

print("\n")
print("=" * 60)
print("  ИТОГОВЫЕ МЕТРИКИ ПО 5 МОДЕЛЯМ")
print("=" * 60)
print("{:<22} {:>10} {:>10} {:>10}".format("Модель", "Precision", "Recall", "F1"))
print("-" * 60)
for r in results:
    print("{:<22} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        r["model"], r["precision"], r["recall"], r["f1"]))
print("=" * 60)
