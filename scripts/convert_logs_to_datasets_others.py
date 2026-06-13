import json
import csv
import os
from datetime import datetime


FILES_CONFIG = [
    ("behavior_log1.json", 1),
    ("behavior_log2.json", 2),
    ("behavior_log3.json", 3),
    ("behavior_log4.json", 4),
]

LABEL_WINDOW_SEC = 10
MAX_GAZE_FRAMES  = 49
MAX_HEAD_FRAMES  = 15
LSTM_WINDOW_SIZE = 10

GAZE_FRAME_FIELDS_ALL = [
    "direction",
    "timestamp",
    "horizontal_deviation",
    "vertical_deviation",
    "total_deviation",
    "horizontal_ratio",
    "vertical_ratio",
    "angle_from_real_center",
    "angle_from_calibrated_center",
]

GAZE_FRAME_FIELDS_NUMERIC = [
    "horizontal_deviation",
    "vertical_deviation",
    "total_deviation",
    "horizontal_ratio",
    "vertical_ratio",
    "angle_from_real_center",
    "angle_from_calibrated_center",
]

HEAD_FRAME_FIELDS_ALL = [
    "direction",
    "timestamp",
    "yaw_deviation",
    "pitch_deviation",
    "is_suspicious",
]

HEAD_FRAME_FIELDS_NUMERIC = [
    "yaw_deviation",
    "pitch_deviation",
    "is_suspicious",
]

BASE_FIELDS_FULL = [
    "source_file",
    "event_timestamp",
    "suspicious_actions",
    "current_status",
    "cheating_trigger",
    "gaze_trigger_count",
    "head_trigger_count",
]

BASE_FIELDS_MINIMAL = [
    "source_file",
    "event_timestamp",
]

CALIB_GAZE_FIELDS = ["calib_horizontal_ratio", "calib_vertical_ratio"]
CALIB_HEAD_FIELDS = ["calib_head_neutral_yaw", "calib_head_neutral_pitch"]
CALIB_ALL_FIELDS  = CALIB_GAZE_FIELDS + CALIB_HEAD_FIELDS


def _v(val):
    return "" if val is None else val


def parse_ts(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return None


def load_events(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    stripped = raw.strip()
    if stripped.startswith("["):
        return json.loads(stripped)
    decoder = json.JSONDecoder()
    events, pos = [], 0
    while pos < len(raw):
        raw_slice = raw[pos:].lstrip()
        if not raw_slice:
            break
        skip = len(raw[pos:]) - len(raw_slice)
        try:
            obj, consumed = decoder.raw_decode(raw_slice)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        events.append(item)
            elif isinstance(obj, dict):
                events.append(obj)
            pos += skip + consumed
        except json.JSONDecodeError:
            pos += skip + 1
    return events


def _gaze_cols(fields, n=MAX_GAZE_FRAMES):
    return ["gaze_{}_{}".format(i, f) for i in range(n) for f in fields]


def _head_cols(fields, n=MAX_HEAD_FRAMES):
    return ["head_{}_{}".format(i, f) for i in range(n) for f in fields]


def _build_row(base_row, gaze_cells, head_cells, fieldnames):
    row = {k: "" for k in fieldnames}
    for k, v in base_row.items():
        if k in row:
            row[k] = v
    for k, v in gaze_cells.items():
        if k in row:
            row[k] = v
    for k, v in head_cells.items():
        if k in row:
            row[k] = v
    return row

def process_file(filepath, version):
    fname  = os.path.basename(filepath)
    events = load_events(filepath)

    manual_cheating_ts = []
    for event in events:
        data = event.get("data", {})
        if not isinstance(data, dict):
            continue
        is_manual = (
            data.get("event_type") == "manual_cheating_mark"
            or "Пользователь отметил" in data.get("message", "")
        )
        if is_manual:
            t = parse_ts(event.get("timestamp", ""))
            if t is not None:
                manual_cheating_ts.append(t)

    current_calib = {
        "calib_horizontal_ratio":   "",
        "calib_vertical_ratio":     "",
        "calib_head_neutral_yaw":   "",
        "calib_head_neutral_pitch": "",
    }

    rows = []

    for event in events:
        ts_str = event.get("timestamp", "")
        data   = event.get("data", {})
        if not isinstance(data, dict):
            continue


        if "participant_number" in data and "message" in data:
            msg = data["message"]
            if "Калибровка" in msg or "калибров" in msg.lower():
                current_calib["calib_horizontal_ratio"]   = _v(data.get("horizontal_ratio"))
                current_calib["calib_vertical_ratio"]     = _v(data.get("vertical_ratio"))
                current_calib["calib_head_neutral_yaw"]   = _v(data.get("head_neutral_yaw"))
                current_calib["calib_head_neutral_pitch"] = _v(data.get("head_neutral_pitch"))
            continue

        if data.get("event_type") == "manual_cheating_mark":
            continue
        if "Пользователь отметил" in data.get("message", ""):
            continue
        if "suspicious_actions" not in data and "gaze_history" not in data:
            continue

        ev_ts = parse_ts(ts_str)


        label = 0
        if ev_ts is not None:
            for mts in manual_cheating_ts:
                if 0 <= (ev_ts - mts) <= LABEL_WINDOW_SEC:
                    label = 1
                    break


        base_row = {
            "source_file":        fname,
            "event_timestamp":    ts_str,
            "suspicious_actions": _v(data.get("suspicious_actions")),
            "current_status":     _v(data.get("current_status")),
            "cheating_trigger":   _v(data.get("cheating_trigger")),
            "gaze_trigger_count": _v(data.get("gaze_trigger_count")),
            "head_trigger_count": _v(data.get("head_trigger_count")),
            "label":              label,
            **current_calib,
        }

        # Gaze history
        gh         = data.get("gaze_history", [])
        gaze_all   = {}
        gaze_num   = {}
        gaze_raw   = []

        for i, item in enumerate(gh[:MAX_GAZE_FRAMES]):
            if version == 1:
                gaze_all["gaze_{}_direction".format(i)] = item if isinstance(item, str) else ""
                gaze_raw.append({"direction": item} if isinstance(item, str) else {})
            else:
                if isinstance(item, dict):
                    for field in GAZE_FRAME_FIELDS_ALL:
                        gaze_all["gaze_{}_{}".format(i, field)] = _v(item.get(field))
                    for field in GAZE_FRAME_FIELDS_NUMERIC:
                        gaze_num["gaze_{}_{}".format(i, field)] = _v(item.get(field))
                    gaze_raw.append(item)

        hh         = data.get("head_history", [])
        head_all   = {}
        head_num   = {}
        head_raw   = []

        for i, item in enumerate(hh[:MAX_HEAD_FRAMES]):
            if isinstance(item, dict):
                for field in HEAD_FRAME_FIELDS_ALL:
                    val = item.get(field)
                    if field == "is_suspicious" and val is not None:
                        val = int(val)
                    head_all["head_{}_{}".format(i, field)] = _v(val)
                for field in HEAD_FRAME_FIELDS_NUMERIC:
                    val = item.get(field)
                    if field == "is_suspicious" and val is not None:
                        val = int(val)
                    head_num["head_{}_{}".format(i, field)] = _v(val)
                head_raw.append(item)

        rows.append({
            "base":     base_row,
            "gaze_all": gaze_all,
            "gaze_num": gaze_num,
            "gaze_raw": gaze_raw,
            "head_all": head_all,
            "head_num": head_num,
            "head_raw": head_raw,
            "ev_ts":    ev_ts,
        })

    return rows



all_records = []

for fname, ver in FILES_CONFIG:
    if not os.path.exists(fname):
        print("[WARN] Файл не найден: {}, пропускаем.".format(fname))
        continue
    records = process_file(fname, ver)
    all_records.extend(records)
    print("[OK] {} (v{}): {:,} наблюдений".format(fname, ver, len(records)))

print("Итого записей: {:,}\n".format(len(all_records)))


# Ortin
ORTIN_BASE = ["source_file", "event_timestamp"] + CALIB_GAZE_FIELDS
ORTIN_GAZE = _gaze_cols(["direction"], MAX_GAZE_FRAMES)
ORTIN_FIELDS = ORTIN_BASE + ORTIN_GAZE + ["label"]

OUTPUT_ORTIN = "dataset_ortin.csv"

with open(OUTPUT_ORTIN, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=ORTIN_FIELDS)
    writer.writeheader()
    for rec in all_records:
        row = {k: "" for k in ORTIN_FIELDS}
        for k in ORTIN_BASE:
            row[k] = rec["base"].get(k, "")
        row["label"] = rec["base"]["label"]
        for i, item in enumerate(rec["gaze_raw"][:MAX_GAZE_FRAMES]):
            d = item.get("direction", "") if isinstance(item, dict) else item
            key = "gaze_{}_direction".format(i)
            if key in row:
                row[key] = _v(d)
        writer.writerow(row)

print("[1] {} — {:,} строк, {} столбцов".format(
    OUTPUT_ORTIN, len(all_records), len(ORTIN_FIELDS)))


# Lamba

LAMBA_BASE   = BASE_FIELDS_MINIMAL + CALIB_ALL_FIELDS
LAMBA_GAZE   = _gaze_cols(GAZE_FRAME_FIELDS_ALL, MAX_GAZE_FRAMES)
LAMBA_HEAD   = _head_cols(HEAD_FRAME_FIELDS_ALL, MAX_HEAD_FRAMES)
LAMBA_FIELDS = LAMBA_BASE + LAMBA_GAZE + LAMBA_HEAD + ["label"]

OUTPUT_LAMBA = "dataset_lamba.csv"

with open(OUTPUT_LAMBA, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=LAMBA_FIELDS)
    writer.writeheader()
    for rec in all_records:
        row = {k: "" for k in LAMBA_FIELDS}
        for k in LAMBA_BASE:
            row[k] = rec["base"].get(k, "")
        row["label"] = rec["base"]["label"]
        row.update({k: v for k, v in rec["gaze_all"].items() if k in row})
        row.update({k: v for k, v in rec["head_all"].items() if k in row})
        writer.writerow(row)

print("[2] {} — {:,} строк, {} столбцов".format(
    OUTPUT_LAMBA, len(all_records), len(LAMBA_FIELDS)))


# Rahmawati

RAHM_BASE   = BASE_FIELDS_MINIMAL + CALIB_ALL_FIELDS
RAHM_GAZE   = _gaze_cols(GAZE_FRAME_FIELDS_NUMERIC, MAX_GAZE_FRAMES)
RAHM_HEAD   = _head_cols(HEAD_FRAME_FIELDS_NUMERIC, MAX_HEAD_FRAMES)
RAHM_FIELDS = RAHM_BASE + RAHM_GAZE + RAHM_HEAD + ["label"]

OUTPUT_RAHM = "dataset_rahmawati.csv"

with open(OUTPUT_RAHM, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=RAHM_FIELDS)
    writer.writeheader()
    for rec in all_records:
        row = {k: "" for k in RAHM_FIELDS}
        for k in RAHM_BASE:
            row[k] = rec["base"].get(k, "")
        row["label"] = rec["base"]["label"]
        row.update({k: v for k, v in rec["gaze_num"].items() if k in row})
        row.update({k: v for k, v in rec["head_num"].items() if k in row})
        writer.writerow(row)

print("[3] {} — {:,} строк, {} столбцов".format(
    OUTPUT_RAHM, len(all_records), len(RAHM_FIELDS)))


# Singh & Das

SINGH_FIELDS = (BASE_FIELDS_MINIMAL + CALIB_ALL_FIELDS
                + ["t_gaze_outside_frames", "f_gaze_exits",
                   "NR_gaze_reversals", "NH_head_suspicious"]
                + ["label"])

GAZE_DEVIATION_THRESHOLD = 0.15
OUTSIDE_DIRECTIONS = {"up", "down", "left", "right",
                      "up_left", "up_right", "down_left", "down_right"}

OUTPUT_SINGH = "dataset_singh_das.csv"

def compute_singh_features(gaze_raw, head_raw):
    t = 0
    for item in gaze_raw:
        if not isinstance(item, dict):
            continue
        td = item.get("total_deviation")
        d  = str(item.get("direction", "")).lower()
        if td is not None:
            try:
                if float(td) > GAZE_DEVIATION_THRESHOLD:
                    t += 1
                    continue
            except (ValueError, TypeError):
                pass
        if d in OUTSIDE_DIRECTIONS:
            t += 1

    f = 0
    prev_outside = False
    for item in gaze_raw:
        d = str(item.get("direction", "")).lower() if isinstance(item, dict) else str(item).lower()
        curr_outside = d in OUTSIDE_DIRECTIONS
        if curr_outside and not prev_outside:
            f += 1
        prev_outside = curr_outside

    NR = 0
    prev_dir = None
    for item in gaze_raw:
        d = str(item.get("direction", "")).lower() if isinstance(item, dict) else str(item).lower()
        if prev_dir is not None and d != prev_dir and d and prev_dir:
            NR += 1
        prev_dir = d

    NH = 0
    for item in head_raw:
        if isinstance(item, dict):
            val = item.get("is_suspicious")
            if val is not None:
                try:
                    NH += int(val)
                except (ValueError, TypeError):
                    pass

    return t, f, NR, NH

with open(OUTPUT_SINGH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=SINGH_FIELDS)
    writer.writeheader()
    for rec in all_records:
        row = {k: "" for k in SINGH_FIELDS}
        for k in BASE_FIELDS_MINIMAL + CALIB_ALL_FIELDS:
            row[k] = rec["base"].get(k, "")
        row["label"] = rec["base"]["label"]
        t, f_val, NR, NH = compute_singh_features(rec["gaze_raw"], rec["head_raw"])
        row["t_gaze_outside_frames"] = t
        row["f_gaze_exits"]          = f_val
        row["NR_gaze_reversals"]     = NR
        row["NH_head_suspicious"]    = NH
        writer.writerow(row)

print("[4] {} — {:,} строк, {} столбцов".format(
    OUTPUT_SINGH, len(all_records), len(SINGH_FIELDS)))


# Dang

DANG_NUMERIC_GAZE = GAZE_FRAME_FIELDS_NUMERIC
DANG_NUMERIC_HEAD = HEAD_FRAME_FIELDS_NUMERIC

DANG_AGG_FUNCS  = ["mean", "std", "min", "max"]
DANG_GAZE_AGG   = ["gaze_{}_{}".format(feat, agg)
                   for feat in DANG_NUMERIC_GAZE for agg in DANG_AGG_FUNCS]
DANG_HEAD_AGG   = ["head_{}_{}".format(feat, agg)
                   for feat in DANG_NUMERIC_HEAD for agg in DANG_AGG_FUNCS]
DANG_BASE       = BASE_FIELDS_MINIMAL + CALIB_ALL_FIELDS
DANG_FIELDS     = (DANG_BASE
                   + ["window_start", "window_end", "window_size"]
                   + DANG_GAZE_AGG + DANG_HEAD_AGG
                   + ["label"])

OUTPUT_DANG = "dataset_dang.csv"

import statistics

def _safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def _agg(values):
    nums = [x for x in values if x is not None]
    if not nums:
        return {"mean": "", "std": "", "min": "", "max": ""}
    mean = sum(nums) / len(nums)
    std  = statistics.pstdev(nums)
    return {"mean": round(mean, 6), "std": round(std, 6),
            "min": min(nums), "max": max(nums)}

with open(OUTPUT_DANG, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=DANG_FIELDS)
    writer.writeheader()

    n = len(all_records)
    window_count = 0

    for start in range(0, n - LSTM_WINDOW_SIZE + 1, LSTM_WINDOW_SIZE):
        window = all_records[start: start + LSTM_WINDOW_SIZE]
        if len(window) < LSTM_WINDOW_SIZE:
            break

        row = {k: "" for k in DANG_FIELDS}

        first = window[0]
        for k in DANG_BASE:
            row[k] = first["base"].get(k, "")
        row["window_start"] = first["base"].get("event_timestamp", "")
        row["window_end"]   = window[-1]["base"].get("event_timestamp", "")
        row["window_size"]  = LSTM_WINDOW_SIZE
        row["label"]        = 1 if any(r["base"]["label"] == 1 for r in window) else 0

        for feat in DANG_NUMERIC_GAZE:
            vals = []
            for rec in window:
                for i in range(MAX_GAZE_FRAMES):
                    v = rec["gaze_num"].get("gaze_{}_{}".format(i, feat))
                    vals.append(_safe_float(v))
            agg = _agg(vals)
            for a in DANG_AGG_FUNCS:
                row["gaze_{}_{}".format(feat, a)] = agg[a]

        for feat in DANG_NUMERIC_HEAD:
            vals = []
            for rec in window:
                for i in range(MAX_HEAD_FRAMES):
                    v = rec["head_num"].get("head_{}_{}".format(i, feat))
                    vals.append(_safe_float(v))
            agg = _agg(vals)
            for a in DANG_AGG_FUNCS:
                row["head_{}_{}".format(feat, a)] = agg[a]

        writer.writerow(row)
        window_count += 1

print("[5] {} — {:,} окон, {} столбцов".format(
    OUTPUT_DANG, window_count, len(DANG_FIELDS)))

print("\n=== Готово! 5 датасетов сформированы. ===")