import os
import cv2
import sqlite3
import time
import base64
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_DIR = os.environ.get("DATABASE_DIR", "database")
DB_PATH = os.path.join(DATABASE_DIR, "ppe_safety.db")
LOCAL_MODEL = os.path.join(os.path.dirname(__file__), "models", "ppe_model.pt")
MODEL_PATH  = os.environ.get("PPE_MODEL", LOCAL_MODEL)
EXPORT_PATH = os.path.join(DATABASE_DIR, "violations_export.csv")

# PPE class definitions from keremberke/yolov8n-PPE-detection
# Each label maps to its semantic meaning and BGR draw color
PPE_META = {
    "Hardhat":       {"kind": "present",   "ppe": "hardhat",      "color": (0, 210, 0)},
    "Mask":          {"kind": "present",   "ppe": "mask",         "color": (0, 210, 0)},
    "Safety Vest":   {"kind": "present",   "ppe": "safety_vest",  "color": (0, 210, 0)},
    "NO-Hardhat":    {"kind": "violation", "ppe": "hardhat",      "color": (0, 0, 220)},
    "NO-Mask":       {"kind": "violation", "ppe": "mask",         "color": (0, 0, 220)},
    "NO-Safety Vest":{"kind": "violation", "ppe": "safety_vest",  "color": (0, 0, 220)},
    "Person":        {"kind": "person",    "ppe": None,           "color": (255, 120, 30)},
    "Safety Cone":   {"kind": "object",    "ppe": None,           "color": (0, 165, 255)},
    "machinery":     {"kind": "object",    "ppe": None,           "color": (0, 165, 255)},
    "vehicle":       {"kind": "object",    "ppe": None,           "color": (0, 165, 255)},
}

# ---------------------------------------------------------------------------
# Safety helmet color verifier
# Hard hats are always bright safety colors: yellow, orange, white, red,
# safety blue, or lime green. Caps are dark (navy, black, grey, brown).
# We crop the YOLO box region and check what fraction of pixels fall within
# recognised safety-helmet HSV ranges. If too low → reject the detection.
# ---------------------------------------------------------------------------
HELMET_SAFE_RATIO = 0.22   # ≥ 22 % of box pixels must be safety-helmet color
VEST_SAFE_RATIO   = 0.09   # ≥  9 % — vest boxes are large (include body/bg pixels)

def _crop_hsv(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return None, None, None, None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2], crop.shape[0] * crop.shape[1]

def _is_safety_helmet(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    h, s, v, total_px = _crop_hsv(frame, x1, y1, x2, y2)
    if h is None:
        return True

    # Safety helmets: white, yellow, orange, red, safety-blue, lime-green
    white  = (s < 55)  & (v > 155)
    yellow = (h >= 18) & (h <= 37) & (s > 70) & (v > 90)
    orange = (h >= 7)  & (h <= 20) & (s > 80) & (v > 90)
    red    = ((h <= 10) | (h >= 158)) & (s > 80) & (v > 70)
    blue   = (h >= 95) & (h <= 125) & (s > 80) & (v > 70)
    green  = (h >= 35) & (h <= 82)  & (s > 90) & (v > 70)

    ratio = (white | yellow | orange | red | blue | green).sum() / total_px
    return ratio >= HELMET_SAFE_RATIO

def _is_safety_vest(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    h, s, v, total_px = _crop_hsv(frame, x1, y1, x2, y2)
    if h is None:
        return True

    # Hi-vis vests: lime-yellow, yellow, orange, or bright orange-red.
    # Threshold is low (9%) because the bounding box covers the whole torso —
    # most pixels are body/background, not just the vest fabric.
    # Also count silver/white reflective strips (low saturation, high brightness).
    lime       = (h >= 28) & (h <= 85)  & (s > 70)  & (v > 90)    # lime / yellow-green
    yellow     = (h >= 15) & (h <= 32)  & (s > 60)  & (v > 100)   # bright yellow
    orange     = (h >= 5)  & (h <= 22)  & (s > 80)  & (v > 90)    # orange hi-vis
    reflective = (s < 60)  & (v > 170)                              # silver / white strips

    ratio = (lime | yellow | orange | reflective).sum() / total_px
    return ratio >= VEST_SAFE_RATIO

# ---------------------------------------------------------------------------
# Mutable runtime state (protected by Python's GIL for simple dict swap)
# ---------------------------------------------------------------------------
required_ppe = {
    "hardhat":     True,
    "safety_vest": True,
    "mask":        False,
}

app_settings = {
    "yolo_conf": 0.45,
    "location":  "Main Entrance",
    "theme":     "dark",
}

current_status = {
    "person_detected": False,
    "is_compliant":    None,
    "present_ppe":     [],
    "missing_ppe":     [],
    "person_count":    0,
    "other_objects":   [],
    "last_update":     None,
}

_last_notification = None
_last_db_write     = 0.0   # throttle DB writes to every 5 s

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_db():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            date         TEXT NOT NULL,
            time         TEXT NOT NULL,
            missing_ppe  TEXT NOT NULL,
            present_ppe  TEXT,
            location     TEXT,
            person_count INTEGER DEFAULT 1
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            date         TEXT NOT NULL,
            time         TEXT NOT NULL,
            compliant    INTEGER NOT NULL,
            person_count INTEGER DEFAULT 1,
            location     TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------------------------
# YOLO model — loaded in background so the app starts instantly
# ---------------------------------------------------------------------------
yolo_model       = None
model_loading    = True   # True while thread is running
model_load_error = None   # set to error string on failure

def _load_model():
    global yolo_model, model_loading, model_load_error
    from ultralytics import YOLO
    try:
        print(f"[startup] Loading PPE model from: {MODEL_PATH}")
        yolo_model = YOLO(MODEL_PATH)
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        yolo_model(dummy, verbose=False)
        print(f"[startup] Model ready ✓  classes={list(yolo_model.names.values())}")
    except Exception as exc:
        model_load_error = str(exc)
        print(f"[startup] Model load failed: {exc}")
    model_loading = False

threading.Thread(target=_load_model, daemon=True).start()

# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/monitor")
def monitor():
    return render_template("monitor.html")

@app.route("/violations")
def violations():
    return render_template("violations.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")

@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": yolo_model is not None,
        "model_loading": model_loading,
        "model_error":  model_load_error,
    })

# ---------------------------------------------------------------------------
# API — process_frame
# ---------------------------------------------------------------------------
@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    global current_status, _last_notification, _last_db_write

    try:
        data = request.json or {}
        raw = data.get("image", "")
        if not raw:
            return jsonify({"success": False, "message": "No image data"}), 400

        if "," in raw:
            raw = raw.split(",", 1)[1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(raw), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "message": "Cannot decode image"}), 400

        # --- Model not ready yet ----------------------------------------
        if yolo_model is None:
            return jsonify({
                "success":    True,
                "loading":    model_loading,
                "error":      model_load_error,
                "detections": [],
                "frame_w":    frame.shape[1],
                "frame_h":    frame.shape[0],
                "status":     current_status,
            })

        # --- Run detection ----------------------------------------------
        results = yolo_model(frame, verbose=False, conf=float(app_settings["yolo_conf"]))

        person_count  = 0
        present_set   = set()
        violation_set = set()
        other_objs    = []
        detections    = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                label = yolo_model.names[cls]

                # Reject detections that fail the safety-color check
                if label == "Hardhat" and not _is_safety_helmet(frame, x1, y1, x2, y2):
                    continue   # caps, handkerchiefs, dark hats → ignored
                if label == "Safety Vest" and not _is_safety_vest(frame, x1, y1, x2, y2):
                    continue   # dark jackets, shirts, hoodies → ignored

                meta  = PPE_META.get(label, {"kind": "object", "ppe": None, "color": (120, 120, 120)})
                kind  = meta["kind"]
                ppe   = meta.get("ppe")

                # BGR → CSS hex
                b, g, rv = meta["color"]
                hex_col = f"#{rv:02x}{g:02x}{b:02x}"

                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": label, "conf": round(conf, 2),
                    "color": hex_col, "kind": kind,
                })

                if kind == "person":
                    person_count += 1
                elif kind == "present" and ppe:
                    present_set.add(ppe)
                elif kind == "violation" and ppe and required_ppe.get(ppe):
                    violation_set.add(ppe)
                elif kind == "object":
                    other_objs.append(label.capitalize())

        # --- Compliance -------------------------------------------------
        person_found = person_count > 0
        missing_list = sorted(p.replace("_", " ").title() for p in violation_set)
        present_list = sorted(p.replace("_", " ").title() for p in present_set)
        compliant    = person_found and len(violation_set) == 0

        # --- Throttled DB write (every 5 s when person present) ----------
        now_ts  = datetime.now()
        now_str = now_ts.strftime("%Y-%m-%d %H:%M:%S")
        d_str   = now_ts.strftime("%Y-%m-%d")
        t_str   = now_ts.strftime("%H:%M:%S")

        if person_found and (time.time() - _last_db_write) >= 5:
            try:
                conn = sqlite3.connect(DB_PATH)
                c    = conn.cursor()
                c.execute(
                    "INSERT INTO scans(timestamp,date,time,compliant,person_count,location) VALUES(?,?,?,?,?,?)",
                    (now_str, d_str, t_str, 1 if compliant else 0, person_count, app_settings["location"]),
                )
                if not compliant and violation_set:
                    c.execute(
                        "INSERT INTO violations(timestamp,date,time,missing_ppe,present_ppe,location,person_count) VALUES(?,?,?,?,?,?,?)",
                        (now_str, d_str, t_str,
                         ", ".join(missing_list), ", ".join(present_list),
                         app_settings["location"], person_count),
                    )
                    _last_notification = f"PPE Violation — Missing: {', '.join(missing_list)}"
                conn.commit()
                conn.close()
            except Exception as db_exc:
                print(f"[db] write error: {db_exc}")
            _last_db_write = time.time()

        # --- Update live status -----------------------------------------
        current_status = {
            "person_detected": person_found,
            "is_compliant":    compliant if person_found else None,
            "present_ppe":     present_list,
            "missing_ppe":     missing_list,
            "person_count":    person_count,
            "other_objects":   list(set(other_objs)),
            "last_update":     now_str,
        }

        return jsonify({
            "success":    True,
            "loading":    False,
            "detections": detections,
            "frame_w":    frame.shape[1],
            "frame_h":    frame.shape[0],
            "status":     current_status,
        })

    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


# ---------------------------------------------------------------------------
# API — live status (polled every 3 s by the monitor page)
# ---------------------------------------------------------------------------
@app.route("/api/live_status")
def api_live_status():
    global _last_notification
    notif = _last_notification
    _last_notification = None
    return jsonify({**current_status, "notification": notif})


# ---------------------------------------------------------------------------
# API — dashboard stats
# ---------------------------------------------------------------------------
@app.route("/api/stats")
def api_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()

    c.execute("SELECT COUNT(*) FROM scans WHERE date=?",               (today,))
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM scans WHERE date=? AND compliant=1", (today,))
    ok    = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM violations WHERE date=?",          (today,))
    viol  = c.fetchone()[0]

    c.execute("SELECT COALESCE(SUM(person_count),0) FROM scans WHERE date=?", (today,))
    ppl   = c.fetchone()[0]

    conn.close()
    rate  = round(ok * 100.0 / total, 1) if total else 100.0
    return jsonify({
        "total_scans":       total,
        "compliance_rate":   rate,
        "violations_today":  viol,
        "people_detected":   ppl,
    })


# ---------------------------------------------------------------------------
# API — chart data
# ---------------------------------------------------------------------------
@app.route("/api/charts")
def api_charts():
    today = datetime.now().strftime("%Y-%m-%d")
    conn  = sqlite3.connect(DB_PATH)

    daily = pd.read_sql_query("""
        SELECT date, ROUND(SUM(compliant)*100.0/COUNT(*),1) AS rate
        FROM scans GROUP BY date ORDER BY date DESC LIMIT 7
    """, conn).iloc[::-1]

    vtype = pd.read_sql_query("""
        SELECT missing_ppe, COUNT(*) AS cnt
        FROM violations WHERE date=?
        GROUP BY missing_ppe
    """, conn, params=(today,))

    hourly = pd.read_sql_query("""
        SELECT SUBSTR(time,1,2) AS hr, COUNT(*) AS cnt
        FROM violations WHERE date=? GROUP BY hr ORDER BY hr
    """, conn, params=(today,))

    conn.close()
    return jsonify({
        "daily":   {"labels": daily["date"].tolist() or [today],
                    "data":   daily["rate"].tolist() or [100.0]},
        "vtype":   {"labels": vtype["missing_ppe"].tolist() if not vtype.empty else ["None"],
                    "data":   vtype["cnt"].tolist()         if not vtype.empty else [0]},
        "hourly":  {"labels": [f"{h}:00" for h in hourly["hr"].tolist()] if not hourly.empty else [],
                    "data":   hourly["cnt"].tolist() if not hourly.empty else []},
    })


# ---------------------------------------------------------------------------
# API — violations list
# ---------------------------------------------------------------------------
@app.route("/api/violations/list")
def api_violations_list():
    date = request.args.get("date", "")
    ptype = request.args.get("ppe_type", "")
    conn  = sqlite3.connect(DB_PATH)
    sql   = "SELECT id,timestamp,date,time,missing_ppe,present_ppe,location,person_count FROM violations WHERE 1=1"
    params = []
    if date:
        sql += " AND date=?";   params.append(date)
    if ptype:
        sql += " AND missing_ppe LIKE ?"; params.append(f"%{ptype}%")
    sql += " ORDER BY timestamp DESC LIMIT 200"
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return jsonify(df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# API — export CSV
# ---------------------------------------------------------------------------
@app.route("/api/export/csv")
def api_export_csv():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT id AS ID, timestamp AS Timestamp, date AS Date, time AS Time,
               missing_ppe AS [Missing PPE], present_ppe AS [Present PPE],
               location AS Location, person_count AS People
        FROM violations ORDER BY timestamp DESC
    """, conn)
    conn.close()
    os.makedirs(DATABASE_DIR, exist_ok=True)
    df.to_csv(EXPORT_PATH, index=False)
    fname = f"PPE_Violations_{datetime.now().strftime('%Y%m%d')}.csv"
    return send_file(EXPORT_PATH, as_attachment=True, download_name=fname)


# ---------------------------------------------------------------------------
# API — settings
# ---------------------------------------------------------------------------
@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global app_settings, required_ppe
    if request.method == "POST":
        d = request.json or {}
        app_settings["yolo_conf"] = float(d.get("yolo_conf",  app_settings["yolo_conf"]))
        app_settings["location"]  = d.get("location",          app_settings["location"])
        app_settings["theme"]     = d.get("theme",             app_settings["theme"])
        required_ppe["hardhat"]     = bool(d.get("require_hardhat",     required_ppe["hardhat"]))
        required_ppe["safety_vest"] = bool(d.get("require_safety_vest", required_ppe["safety_vest"]))
        required_ppe["mask"]        = bool(d.get("require_mask",        required_ppe["mask"]))
        return jsonify({"success": True, "message": "Settings saved."})
    return jsonify({
        **app_settings,
        "require_hardhat":     required_ppe["hardhat"],
        "require_safety_vest": required_ppe["safety_vest"],
        "require_mask":        required_ppe["mask"],
    })


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5002))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
