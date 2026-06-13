import os, glob, warnings
import numpy as np
import pandas as pd
import cv2
import kagglehub
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
)
warnings.filterwarnings('ignore')
np.random.seed(42)
ALL_RESULTS = []
print('OK')

def diagnose(df, label):
    print('\n--- ' + label + ' ---')
    print('  shape=' + str(df.shape))
    print('  balance=' + str(dict(df['label'].value_counts())))
    print('  NaN=' + str(df.isnull().sum().sum()))
    fc = [c for c in df.columns if c != 'label']
    if fc:
        v = df[fc].fillna(0).values.astype(float)
        print('  feat: min={:.3f} max={:.3f} std={:.3f}'.format(v.min(), v.max(), v.std()))

def check_balance(y, lbl, min_r=0.05):
    u, c = np.unique(y, return_counts=True)
    if len(u) < 2:
        print('  SKIP [' + lbl + ']: только класс ' + str(u) + ' => невозможно обучить')
        return False
    r = c.min() / c.sum()
    print('  minority={:.1%}'.format(r))
    if r < min_r:
        print('  WARNING: дисбаланс {:.1%}'.format(r))
    return True

def enc(df):
    df = df.copy()
    for col in df.columns:
        if col == 'label': continue
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna('x').astype(str))
    return df

def prep(df):
    df = enc(df)
    fc = [c for c in df.columns if c != 'label']
    return df[fc].fillna(0).values.astype(float), df['label'].astype(int).values

def run_cv(clf, name, X, y, ds, n=5):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    pipe = Pipeline([('sc', StandardScaler()), ('clf', clf)])
    sc = cross_validate(pipe, X, y, cv=skf,
        scoring={'p': 'precision', 'r': 'recall', 'f': 'f1'}, n_jobs=-1)
    p = sc['test_p'].mean(); r = sc['test_r'].mean()
    f = sc['test_f'].mean(); fs = sc['test_f'].std()
    print('  {:<26} P={:.4f} R={:.4f} F1={:.4f}+-{:.4f}'.format(name, p, r, f, fs))
    ALL_RESULTS.append({'dataset': ds, 'model': name,
        'precision': round(p, 4), 'recall': round(r, 4),
        'f1': round(f, 4), 'f1_std': round(fs, 4)})

def hdr(lbl, note, n, nf):
    print('=' * 60)
    print(lbl)
    print(note)
    print('Records={} Features={}'.format(n, nf))
    print('=' * 60)

print('Утилиты OK')

def mmss2sec(t):
    t = t.strip()
    if not t.isdigit(): return 0
    if len(t) <= 2: return int(t)
    mm, ss = int(t[:-2]), int(t[-2:])
    return mm * 60 + ss if ss < 60 else int(t)

def parse_oep_gt(gt_path):
    ivs = []
    if not gt_path or not os.path.exists(gt_path): return ivs
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 2: continue
            try:
                s, e = mmss2sec(p[0]), mmss2sec(p[1])
                if e > s: ivs.append((s, e))
            except: pass
    return ivs

def frame_lbl(fi, fps, ivs):
    t = fi / fps
    return int(any(s <= t <= e for s, e in ivs))

assert mmss2sec('0135') == 95
assert mmss2sec('1024') == 624
assert mmss2sec('0204') == 124
assert mmss2sec('1245') == 765
print('mmss2sec: OK')
print('  0135 =>', mmss2sec('0135'), 'сек  (1м35с)')
print('  1245 =>', mmss2sec('1245'), 'сек  (12м45с)')


CHEAT_KW = {'cheat', 'cheating', 'fraud', 'suspicious', 'menengok',
            'contekan', 'looking_around', 'malpractice'}
NORM_KW  = {'non_cheat', 'non-cheat', 'no_cheat', 'not_cheat', 'normal',
            'noncheating', 'non-cheating', 'good', 'honest'}

def name2lbl(name):
    n = name.lower().replace('-', '_').replace(' ', '_')
    if any(k.replace(' ', '_') in n for k in NORM_KW):  return 0
    if any(k.replace(' ', '_') in n for k in CHEAT_KW): return 1
    return None

def load_cmap(d):
    for fn in ['classes.txt', 'obj.names', '_darknet.labels']:
        for r, _, fs in os.walk(d):
            if fn in fs:
                with open(os.path.join(r, fn)) as f:
                    names = [l.strip() for l in f if l.strip()]
                m = {i: name2lbl(n) for i, n in enumerate(names)}
                print('  ' + fn + ': ' + str(list(enumerate(names))) + ' => ' + str(m))
                return m
    try:
        import yaml
        for yf in glob.glob(os.path.join(d, '**', 'data.yaml'), recursive=True):
            with open(yf) as f: data = yaml.safe_load(f)
            names = data.get('names', [])
            if isinstance(names, list) and names:
                m = {i: name2lbl(n) for i, n in enumerate(names)}
                print('  data.yaml: ' + str(list(enumerate(names))) + ' => ' + str(m))
                return m
    except: pass
    print('  Нет classes.txt, class_id=0 => cheating')
    return {}

def parse_yolo(d):
    cmap = load_cmap(d)
    SKIP = {'classes.txt', 'obj.names', '_darknet.labels',
            'README.dataset.txt', 'README.roboflow.txt'}
    txts = [os.path.join(r, fn)
            for r, _, fs in os.walk(d) for fn in fs
            if fn.endswith('.txt') and fn not in SKIP]
    recs, skip_n = [], 0
    for fp in txts:
        boxes, lbls = [], []
        try:
            with open(fp) as f:
                for line in f:
                    ps = line.strip().split()
                    if len(ps) < 5: continue
                    cid = int(ps[0]); cx, cy, w, h = map(float, ps[1:5])
                    boxes.append({'cx': cx, 'cy': cy, 'w': w, 'h': h,
                                  'area': w * h, 'asp': w / (h + 1e-9)})
                    lbl = cmap.get(cid, 1 if cid == 0 else 0) if cmap else (1 if cid == 0 else 0)
                    lbls.append(lbl)
        except: continue
        if not boxes: continue
        known = [l for l in lbls if l is not None]
        if not known: skip_n += 1; continue
        il = int(any(l == 1 for l in known))
        b = pd.DataFrame(boxes)
        recs.append({'n_boxes': len(boxes),
            'n_cheat': sum(1 for l in known if l == 1),
            'cx_mean': b.cx.mean(), 'cx_std': b.cx.std(ddof=0),
            'cx_min': b.cx.min(), 'cx_max': b.cx.max(),
            'cy_mean': b.cy.mean(), 'cy_std': b.cy.std(ddof=0),
            'w_mean': b.w.mean(), 'w_std': b.w.std(ddof=0),
            'h_mean': b.h.mean(), 'h_std': b.h.std(ddof=0),
            'area_mean': b.area.mean(), 'area_std': b.area.std(ddof=0),
            'asp_mean': b.asp.mean(), 'label': il})
    if skip_n: print('  Пропущено (неизвестные классы): ' + str(skip_n))
    return pd.DataFrame(recs)

print('YOLO-парсер OK')

def hog_feat(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    g = cv2.resize(g, (64, 64))
    h = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    f = h.compute(g).flatten()
    e = cv2.Canny(g, 50, 150)
    ex = np.array([g.mean(), g.std(), float(g.min()), float(g.max()),
                   np.percentile(g, 25), np.percentile(g, 75),
                   e.mean(), e.std()])
    return np.concatenate([f, ex])

def load_oep(root, fps_s=1.0, max_subj=None, max_rec=8000):
    dirs = sorted(set(
        os.path.realpath(d)
        for d in glob.glob(os.path.join(root, '**', 'subject*'), recursive=True)
        if os.path.isdir(d)
    ))
    if max_subj: dirs = dirs[:max_subj]
    print('  Субъектов: ' + str(len(dirs)))
    recs = []
    for sd in dirs:
        if max_rec and len(recs) >= max_rec: break
        avi = glob.glob(os.path.join(sd, '*1.avi'))
        if not avi: continue
        ivs = parse_oep_gt(os.path.join(sd, 'gt.txt'))
        cap = cv2.VideoCapture(avi[0])
        if not cap.isOpened(): continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(fps / fps_s))
        fi, ext, ch = 0, 0, 0
        while True:
            if max_rec and len(recs) >= max_rec: break
            ret, frame = cap.read()
            if not ret: break
            if fi % step == 0:
                try:
                    feat = hog_feat(frame)
                    lbl = frame_lbl(fi, fps, ivs)
                    rec = {'f' + str(i): v for i, v in enumerate(feat)}
                    rec['label'] = lbl
                    recs.append(rec)
                    ext += 1; ch += lbl
                except: pass
            fi += 1
        cap.release()
        ratio = ch / ext if ext else 0
        print('  ' + os.path.basename(sd) + ': ' + str(ext) + ' fr cheat=' + str(ch) + '({:.0%})'.format(ratio))
    return pd.DataFrame(recs)

print('HOG-парсер OK')

L1 = 'DS1'
N1 = '13 классов (eye+head)'
D1 = 'mendeley_ds1'
if os.path.exists(D1) and os.listdir(D1):
    df1 = parse_yolo(D1); diagnose(df1, L1)
    X1, y1 = prep(df1)
    hdr(L1, N1, len(df1), X1.shape[1])
    O1 = check_balance(y1, L1)
else:
    O1 = False
    print('DS1 пропущен — нет папки ' + D1 + '/')

if O1:
    run_cv(GradientBoostingClassifier(
        learning_rate=0.05, max_depth=5, n_estimators=300,
        subsample=0.8, random_state=42),
        'GradientBoosting', X1, y1, L1)

if O1:
    run_cv(RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'RandomForest', X1, y1, L1)

if O1:
    run_cv(ExtraTreesClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'ExtraTrees', X1, y1, L1)


L2 = 'DS2'
N2 = '24 субъекта'
print('Скачивание OEP...')
p2 = kagglehub.dataset_download('raajanwankhade/oep-dataset')
print('Path: ' + p2)
print('Извлечение HOG (1 кадр/сек, max=8000)...')
df2 = load_oep(p2, fps_s=1.0, max_rec=8000)
diagnose(df2, L2); X2, y2 = prep(df2)
hdr(L2, N2, len(df2), X2.shape[1])
O2 = check_balance(y2, L2)

if O2:
    run_cv(GradientBoostingClassifier(
        learning_rate=0.05, max_depth=5, n_estimators=300,
        subsample=0.8, random_state=42),
        'GradientBoosting', X2, y2, L2)

if O2:
    run_cv(RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'RandomForest', X2, y2, L2)

if O2:
    run_cv(ExtraTreesClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'ExtraTrees', X2, y2, L2)


L3 = 'DS3'
N3 = 'cheating-detection'
print('Скачивание...')
p3 = kagglehub.dataset_download('aneelapervez/classroom-exam-cheating-detection')
df3 = parse_yolo(p3)
diagnose(df3, L3)
X3, y3 = prep(df3)
hdr(L3, N3, len(df3), X3.shape[1])
O3 = check_balance(y3, L3)

if O3:
    run_cv(GradientBoostingClassifier(learning_rate=0.05,max_depth=5,n_estimators=300,subsample=0.8,random_state=42), 'GradientBoosting', X3, y3, L3)

if O3:
    run_cv(RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1), 'RandomForest', X3, y3, L3)

if O3:
    run_cv(ExtraTreesClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1), 'ExtraTrees', X3, y3, L3)


L4 = 'DS4'
N4 = 'yolo'
print('Скачивание...')
p4 = kagglehub.dataset_download('aneelapervez/yolo-11-cheating-dataset')
df4 = parse_yolo(p4)
diagnose(df4, L4)
X4, y4 = prep(df4)
hdr(L4, N4, len(df4), X4.shape[1])
O4 = check_balance(y4, L4)

if O4:
    run_cv(GradientBoostingClassifier(learning_rate=0.05,max_depth=5,n_estimators=300,subsample=0.8,random_state=42), 'GradientBoosting', X4, y4, L4)

if O4:
    run_cv(RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1), 'RandomForest', X4, y4, L4)

if O4:
    run_cv(ExtraTreesClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1), 'ExtraTrees', X4, y4, L4)


L5 = 'DS5'
N5 = 'yolo'
print('Скачивание...')
p5 = kagglehub.dataset_download('aneelapervez/yolo-exam-chaet-detection')
df5 = parse_yolo(p5)
diagnose(df5, L5)
X5, y5 = prep(df5)
hdr(L5, N5, len(df5), X5.shape[1])
O5 = check_balance(y5, L5)

if O5:
    run_cv(GradientBoostingClassifier(learning_rate=0.05,max_depth=5,n_estimators=300,subsample=0.8,random_state=42), 'GradientBoosting', X5, y5, L5)

if O5:
    run_cv(RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1), 'RandomForest', X5, y5, L5)

if O5:
    run_cv(ExtraTreesClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1), 'ExtraTrees', X5, y5, L5)


L6 = 'DS6'
N6 = 'cheat-exam'
print('Скачивание...')
p6 = kagglehub.dataset_download('abderahmanelzahar/cheat-exam')
df6 = parse_yolo(p6)
diagnose(df6, L6)
X6, y6 = prep(df6)
hdr(L6, N6, len(df6), X6.shape[1])
O6 = check_balance(y6, L6)

if O6:
    run_cv(GradientBoostingClassifier(learning_rate=0.05,max_depth=5,n_estimators=300,subsample=0.8,random_state=42), 'GradientBoosting', X6, y6, L6)

if O6:
    run_cv(RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1), 'RandomForest', X6, y6, L6)

if O6:
    run_cv(ExtraTreesClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1), 'ExtraTrees', X6, y6, L6)


if ALL_RESULTS:
    df_r = pd.DataFrame(ALL_RESULTS)
    print('=' * 70)
    print('СВОДНАЯ ТАБЛИЦА  (5-fold Stratified CV)')
    print('=' * 70)
    print(df_r.to_string(index=False))
    df_r.to_csv('benchmark_results.csv', index=False)
    print('Сохранено: benchmark_results.csv')
else:
    print('Нет результатов')

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 120
if ALL_RESULTS:
    df_p = pd.DataFrame(ALL_RESULTS)
    dsets = df_p['dataset'].unique()
    fig, axes = plt.subplots(1, len(dsets), figsize=(5 * len(dsets), 5), sharey=True)
    if len(dsets) == 1: axes = [axes]
    cols = ['#4C72B0', '#DD8452', '#55A868']
    for ax, ds in zip(axes, dsets):
        sub = df_p[df_p['dataset'] == ds]
        x = np.arange(len(sub)); w = 0.25
        for i, (m, c) in enumerate(zip(['precision', 'recall', 'f1'], cols)):
            ax.bar(x + i * w, sub[m], w, label=m.capitalize(), color=c, alpha=0.85)
        ax.errorbar(x + 2 * w, sub['f1'], yerr=sub['f1_std'],
                    fmt='none', color='k', capsize=4, lw=1.5)
        ax.set_xticks(x + w)
        ax.set_xticklabels(sub['model'], rotation=20, ha='right', fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(ds.split('—')[0].strip(), fontsize=8)
        ax.legend(fontsize=7)
        ax.axhline(0.9, color='r', ls='--', lw=0.8, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Cheating Detection Benchmark (5-fold CV)', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig('benchmark_chart.png', bbox_inches='tight')
    plt.show()
    print('Сохранено: benchmark_chart.png')
else:
    print('Нет данных для графика')