import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (GradientBoostingClassifier, ExtraTreesClassifier,
                               StackingClassifier, RandomForestClassifier)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_features(csv_path='dataset_observations.csv'):
    df = pd.read_csv(csv_path, low_memory=False)

    dir_map = {
        'center': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4,
        'left up': 5, 'left down': 6, 'right up': 7, 'right down': 8,
        'blink': 9, 'not detected': -1
    }

    gaze_dir_cols    = [c for c in df.columns if c.startswith('gaze_') and c.endswith('_direction')  and c.split('_')[1].isdigit()]
    gaze_total_dev   = [c for c in df.columns if 'total_deviation' in c              and c.split('_')[1].isdigit()]
    gaze_dev_h       = [c for c in df.columns if 'horizontal_deviation' in c         and c.split('_')[1].isdigit()]
    gaze_dev_v       = [c for c in df.columns if 'vertical_deviation' in c           and c.split('_')[1].isdigit()]
    gaze_angle_real  = [c for c in df.columns if 'angle_from_real_center' in c       and c.split('_')[1].isdigit()]
    gaze_angle_calib = [c for c in df.columns if 'angle_from_calibrated_center' in c and c.split('_')[1].isdigit()]
    head_yaw         = [c for c in df.columns if 'yaw_deviation' in c]
    head_pitch       = [c for c in df.columns if 'pitch_deviation' in c]
    head_dir_cols    = [c for c in df.columns if c.startswith('head_') and c.endswith('_direction')]
    head_susp        = [c for c in df.columns if 'is_suspicious' in c]

    enc = lambda cols: df[cols].applymap(
        lambda x: dir_map.get(str(x).lower() if pd.notna(x) else 'not detected', -1))
    ge = enc(gaze_dir_cols)
    he = enc(head_dir_cols)

    feat = pd.DataFrame()
    feat['gaze_dir_center_ratio'] = (ge == 0).sum(axis=1) / len(gaze_dir_cols)
    feat['gaze_dir_off_ratio']    = (ge > 0).sum(axis=1)  / len(gaze_dir_cols)
    feat['gaze_dir_blink_count']  = (ge == 9).sum(axis=1)
    feat['gaze_dir_left_ratio']   = ((ge == 3) | (ge == 5) | (ge == 6)).sum(axis=1) / len(gaze_dir_cols)
    feat['gaze_dir_down_ratio']   = ((ge == 2) | (ge == 6) | (ge == 8)).sum(axis=1) / len(gaze_dir_cols)
    feat['gaze_dir_unique']       = ge.nunique(axis=1)
    feat['gaze_total_dev_mean']   = df[gaze_total_dev].astype(float).mean(axis=1)
    feat['gaze_total_dev_std']    = df[gaze_total_dev].astype(float).std(axis=1)
    feat['gaze_total_dev_max']    = df[gaze_total_dev].astype(float).max(axis=1)
    feat['gaze_h_dev_mean']       = df[gaze_dev_h].astype(float).mean(axis=1)
    feat['gaze_v_dev_mean']       = df[gaze_dev_v].astype(float).mean(axis=1)
    feat['gaze_angle_real_mean']  = df[gaze_angle_real].astype(float).mean(axis=1)
    feat['gaze_angle_calib_mean'] = df[gaze_angle_calib].astype(float).mean(axis=1)
    feat['head_dir_center_ratio'] = (he == 0).sum(axis=1) / len(head_dir_cols)
    feat['head_dir_off_ratio']    = (he > 0).sum(axis=1)  / len(head_dir_cols)
    feat['head_susp_ratio']       = df[head_susp].astype(float).sum(axis=1) / len(head_susp)
    feat['head_yaw_abs_mean']     = df[head_yaw].astype(float).abs().mean(axis=1)
    feat['head_yaw_max']          = df[head_yaw].astype(float).abs().max(axis=1)
    feat['head_pitch_abs_mean']   = df[head_pitch].astype(float).abs().mean(axis=1)
    feat['head_pitch_max']        = df[head_pitch].astype(float).abs().max(axis=1)
    feat['suspicious_actions']    = df['suspicious_actions'].astype(float)
    feat['gaze_x_head_off']       = feat['gaze_dir_off_ratio'] * feat['head_dir_off_ratio']
    feat['gaze_dev_x_yaw']        = feat['gaze_total_dev_mean'] * feat['head_yaw_abs_mean']
    feat['suspicious_x_off']      = feat['suspicious_actions'] * feat['gaze_dir_off_ratio']
    feat['yaw_pitch_combined']    = feat['head_yaw_abs_mean'] + feat['head_pitch_abs_mean']
    feat['label'] = df['label'].astype(int)
    return feat.fillna(0)


def make_sequences(csv_path='dataset_observations.csv', seq_len=20):
    df = pd.read_csv(csv_path, low_memory=False)
    dir_map = {
        'center': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4,
        'left up': 5, 'left down': 6, 'right up': 7, 'right down': 8,
        'blink': 9, 'not detected': -1
    }
    sequences, labels = [], df['label'].astype(int).values
    for _, row in df.iterrows():
        fvs = []
        for i in range(seq_len):
            fv = []
            gc = f'gaze_{i}_direction'
            fv.append(dir_map.get(str(row.get(gc, 'center')).lower() if pd.notna(row.get(gc)) else 'center', 0) / 9.0)
            for feat_name in ['horizontal_deviation', 'vertical_deviation', 'total_deviation',
                               'angle_from_real_center', 'angle_from_calibrated_center']:
                col = f'gaze_{i}_{feat_name}'
                fv.append(float(row.get(col, 0)) if pd.notna(row.get(col, None)) else 0.0)
            if i < 15:
                hc = f'head_{i}_direction'
                fv.append(dir_map.get(str(row.get(hc, 'center')).lower() if pd.notna(row.get(hc)) else 'center', 0) / 9.0)
                for hf in ['yaw_deviation', 'pitch_deviation']:
                    hval = float(row.get(f'head_{i}_{hf}', 0)) if pd.notna(row.get(f'head_{i}_{hf}', None)) else 0.0
                    fv.append(hval)
                fv.append(float(row.get(f'head_{i}_is_suspicious', 0)) if pd.notna(row.get(f'head_{i}_is_suspicious', None)) else 0.0)
            else:
                fv.extend([0.0, 0.0, 0.0, 0.0])
            fvs.append(fv)
        sequences.append(fvs)
    X = np.array(sequences, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), labels


def evaluate(y_true, y_pred, name):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    return {'model': name, 'precision': prec, 'recall': rec, 'f1': f1}


def m1_lstm(Xs_tr, Xs_te, y_tr, y_te):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        m = Sequential([
            LSTM(64, return_sequences=True, input_shape=Xs_tr.shape[1:]),
            Dropout(0.2), LSTM(32), Dropout(0.2),
            Dense(16, activation='relu'), Dropout(0.2), Dense(1, activation='sigmoid')
        ])
        m.compile(optimizer='adam', loss='binary_crossentropy')
        m.fit(Xs_tr, y_tr, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
        return (m.predict(Xs_te, verbose=0).flatten() >= 0.5).astype(int)
    except ImportError:
        print("  [TF unavailable, using GradientBoosting fallback]")
        Xf = lambda x: x.reshape(len(x), -1)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(Xf(Xs_tr), y_tr)
        return clf.predict(Xf(Xs_te))


def m2_bilstm(Xs_tr, Xs_te, y_tr, y_te):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
        m = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=Xs_tr.shape[1:]),
            Dropout(0.3), Bidirectional(LSTM(32)),
            Dense(32, activation='relu'), Dropout(0.3), Dense(1, activation='sigmoid')
        ])
        m.compile(optimizer='adam', loss='binary_crossentropy')
        m.fit(Xs_tr, y_tr, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
        return (m.predict(Xs_te, verbose=0).flatten() >= 0.5).astype(int)
    except ImportError:
        print("  [TF unavailable, using GradientBoosting fallback]")
        Xf = lambda x: x.reshape(len(x), -1)
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
        clf.fit(Xf(Xs_tr), y_tr)
        return clf.predict(Xf(Xs_te))


def m3_svm_threshold(X_tr, X_te, y_tr, _):
    pipe = Pipeline([('sc', StandardScaler()), ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42))])
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def m4_random_forest(X_tr, X_te, y_tr, _):
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)


def m5_svm_rbf(X_tr, X_te, y_tr, _):
    pipe = Pipeline([('sc', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='auto', random_state=42))])
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def m6_decision_tree(X_tr, X_te, y_tr, _):
    clf = DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=5,
                                  class_weight='balanced', random_state=42)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)


def m7_gbdt_xgboost(X_tr, X_te, y_tr, _):
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                                      subsample=0.8, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)


def m8_hybrid_gbdt_mlp(X_tr, X_te, y_tr, _):
    gbdt = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, random_state=42)
    gbdt.fit(X_tr, y_tr)
    X_tr_aug = np.hstack([X_tr, gbdt.apply(X_tr).reshape(len(X_tr), -1)])
    X_te_aug = np.hstack([X_te, gbdt.apply(X_te).reshape(len(X_te), -1)])
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_aug)
    X_te_s = sc.transform(X_te_aug)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu',
                         solver='adam', alpha=1e-4, max_iter=300, random_state=42)
    mlp.fit(X_tr_s, y_tr)
    return mlp.predict(X_te_s)


def m9_extratrees(X_tr, X_te, y_tr, _):
    clf = ExtraTreesClassifier(n_estimators=300, min_samples_split=4,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)


def m10_stacking(X_tr, X_te, y_tr, _):
    base = [
        ('rf',  RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('et',  ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=43, n_jobs=-1)),
        ('gbm', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42))
    ]
    clf = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(C=5.0, max_iter=300, random_state=42),
        cv=3, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)


def run_all(csv_path='dataset_observations.csv'):
    print(">>> Загрузка данных и построение признаков...")
    feat_df = build_features(csv_path)
    feat_cols = [c for c in feat_df.columns if c != 'label']
    X = feat_df[feat_cols].values
    y = feat_df['label'].values
    print(f"    Табличные признаки: {X.shape}")

    print("\n>>> Построение последовательностей для LSTM...")
    Xs, ys = make_sequences(csv_path, seq_len=20)
    print(f"    Последовательные данные: {Xs.shape}")

    Xt, Xte, yt, yte = train_test_split(X,  y,  test_size=0.2, random_state=42, stratify=y)
    Xs_t, Xs_te, ys_t, ys_te = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=ys)
    print(f"\n    Train: {len(yt)}  Test: {len(yte)}")

    models = [
        ("Model 1:",                  lambda: m1_lstm(Xs_t, Xs_te, ys_t, ys_te),      ys_te),
        ("Model 2:",   lambda: m2_bilstm(Xs_t, Xs_te, ys_t, ys_te),    ys_te),
        ("Model 3:",             lambda: m3_svm_threshold(Xt, Xte, yt, yte),     yte),
        ("Model 4:",    lambda: m4_random_forest(Xt, Xte, yt, yte),     yte),
        ("Model 5:",       lambda: m5_svm_rbf(Xt, Xte, yt, yte),           yte),
        ("Model 6:",  lambda: m6_decision_tree(Xt, Xte, yt, yte),     yte),
        ("Model 7:",        lambda: m7_gbdt_xgboost(Xt, Xte, yt, yte),      yte),
        ("Model 8:",     lambda: m8_hybrid_gbdt_mlp(Xt, Xte, yt, yte),   yte),
        ("Model 9:",       lambda: m9_extratrees(Xt, Xte, yt, yte),         yte),
        ("Model 10:", lambda: m10_stacking(Xt, Xte, yt, yte),      yte),
    ]

    results = []
    for i, (name, fn, y_true) in enumerate(models, 1):
        print(f"\n[{i}/10] {name}...")
        y_pred = fn()
        results.append(evaluate(y_true, y_pred, name))

    print("\n" + "="*75)
    print("  ИТОГОВЫЕ РЕЗУЛЬТАТЫ (сортировка по F1)")
    print("="*75)
    df_r = pd.DataFrame(results).sort_values('f1', ascending=False).reset_index(drop=True)
    print(f"  {'Модель':<50} {'Prec':>6} {'Recall':>7} {'F1':>7}")
    print("-"*75)
    for _, r in df_r.iterrows():
        print(f"  {r['model']:<50} {r['precision']:>6.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")
    print("="*75)

    df_r.to_csv('model_results_all10.csv', index=False)
    print("\n>>> Результаты сохранены в model_results_all10.csv")
    return df_r


if __name__ == '__main__':
    run_all()