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

OUTPUT_ALL       = "dataset_all.csv"
OUTPUT_GAZE_ONLY = "dataset_gaze_only.csv"
OUTPUT_HEAD_ONLY = "dataset_head_only.csv"

LABEL_WINDOW_SEC = 10

MAX_GAZE_FRAMES = 49
MAX_HEAD_FRAMES = 15

MODEL_FIELDS_TO_EXCLUDE = {"model_verdict", "model_probability"}

GAZE_FRAME_FIELDS = [
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

HEAD_FRAME_FIELDS = [
    "direction",
    "timestamp",
    "yaw_deviation",
    "pitch_deviation",
    "is_suspicious",
]

BASE_FIELDS = [
    "source_file",
    "event_timestamp",
    "suspicious_actions",
    "current_status",
    "cheating_trigger",
    "gaze_trigger_count",
    "head_trigger_count",
]

CALIB_GAZE_FIELDS = ["calib_horizontal_ratio", "calib_vertical_ratio"]
CALIB_HEAD_FIELDS = ["calib_head_neutral_yaw", "calib_head_neutral_pitch"]


def _gaze_cols():
    cols = []
    for i in range(MAX_GAZE_FRAMES):
        for f in GAZE_FRAME_FIELDS:
            cols.append("gaze_{}_{}".format(i, f))
    return cols


def _head_cols():
    cols = []
    for i in range(MAX_HEAD_FRAMES):
        for f in HEAD_FRAME_FIELDS:
            cols.append("head_{}_{}".format(i, f))
    return cols


FIELDNAMES_ALL  = (BASE_FIELDS + CALIB_GAZE_FIELDS + CALIB_HEAD_FIELDS
                   + _gaze_cols() + _head_cols() + ["label"])
FIELDNAMES_GAZE = BASE_FIELDS + CALIB_GAZE_FIELDS + _gaze_cols() + ["label"]
FIELDNAMES_HEAD = BASE_FIELDS + CALIB_HEAD_FIELDS + _head_cols() + ["label"]


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
    events = []
    pos = 0
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
            "calib_horizontal_ratio":   current_calib["calib_horizontal_ratio"],
            "calib_vertical_ratio":     current_calib["calib_vertical_ratio"],
            "calib_head_neutral_yaw":   current_calib["calib_head_neutral_yaw"],
            "calib_head_neutral_pitch": current_calib["calib_head_neutral_pitch"],
        }

        gaze_cells = {}
        gh = data.get("gaze_history", [])
        for i, item in enumerate(gh[:MAX_GAZE_FRAMES]):
            if version == 1:
                gaze_cells["gaze_{}_direction".format(i)] = (
                    item if isinstance(item, str) else ""
                )
            else:
                if isinstance(item, dict):
                    for field in GAZE_FRAME_FIELDS:
                        gaze_cells["gaze_{}_{}".format(i, field)] = _v(item.get(field))

        head_cells = {}
        hh = data.get("head_history", [])
        for i, item in enumerate(hh[:MAX_HEAD_FRAMES]):
            if isinstance(item, dict):
                for field in HEAD_FRAME_FIELDS:
                    val = item.get(field)
                    if field == "is_suspicious" and val is not None:
                        val = int(val)
                    head_cells["head_{}_{}".format(i, field)] = _v(val)

        rows.append({"base": base_row, "gaze": gaze_cells, "head": head_cells})

    return rows


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


all_records = []

for fname, ver in FILES_CONFIG:
    if not os.path.exists(fname):
        print("[WARN] Файл не найден: {}, пропускаем.".format(fname))
        continue
    records = process_file(fname, ver)
    all_records.extend(records)
    print("[OK] {} (v{}): {:,} наблюдений".format(fname, ver, len(records)))

with open(OUTPUT_ALL, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES_ALL)
    writer.writeheader()
    for rec in all_records:
        writer.writerow(_build_row(rec["base"], rec["gaze"], rec["head"], FIELDNAMES_ALL))
print("\n[1] {} — {:,} строк, {} столбцов".format(
    OUTPUT_ALL, len(all_records), len(FIELDNAMES_ALL)))

with open(OUTPUT_GAZE_ONLY, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES_GAZE)
    writer.writeheader()
    for rec in all_records:
        writer.writerow(_build_row(rec["base"], rec["gaze"], {}, FIELDNAMES_GAZE))
print("[2] {} — {:,} строк, {} столбцов".format(
    OUTPUT_GAZE_ONLY, len(all_records), len(FIELDNAMES_GAZE)))

with open(OUTPUT_HEAD_ONLY, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES_HEAD)
    writer.writeheader()
    for rec in all_records:
        writer.writerow(_build_row(rec["base"], {}, rec["head"], FIELDNAMES_HEAD))
print("[3] {} — {:,} строк, {} столбцов".format(
    OUTPUT_HEAD_ONLY, len(all_records), len(FIELDNAMES_HEAD)))

print("\nГотово!")