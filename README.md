<div align="center">

# 🦺 PPE Safety AI

**Real-time Personal Protective Equipment compliance detection powered by YOLOv8**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-PPE_Model-FF6B35?style=flat-square)](https://ultralytics.com/)
[![Railway](https://img.shields.io/badge/Deploy-Railway-0B0D0E?style=flat-square&logo=railway&logoColor=white)](https://railway.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Point a camera at a worker — the AI instantly detects whether they're wearing required safety gear and logs every violation.

</div>

---

## What It Does

Workers enter a camera's field of view. The system:

1. Detects people and PPE items using a dedicated **YOLOv8 model**
2. Checks which required equipment is **present** and which is **missing**
3. Displays a real-time **COMPLIANT** or **VIOLATION** banner on the live feed
4. **Logs every violation** to a database with timestamp and location
5. Renders annotated bounding boxes — green for worn PPE, red for missing PPE

---

## Detection Classes

| Class | Status | Meaning |
|---|---|---|
| `Hardhat` | ✅ Safe | Hard hat is being worn |
| `Safety Vest` | ✅ Safe | Hi-vis vest is being worn |
| `Mask` | ✅ Safe | Face mask is being worn |
| `NO-Hardhat` | 🚨 Violation | Hard hat is **missing** |
| `NO-Safety Vest` | 🚨 Violation | Safety vest is **missing** |
| `NO-Mask` | 🚨 Violation | Mask is **missing** |
| `Person` | 👷 Info | Worker detected |
| `Safety Cone` / `machinery` / `vehicle` | 🟠 Info | Other site objects |

> The model detects both the **presence** and **absence** of PPE — not just objects, but compliance state.

---

## Detection Flow

```
Browser Webcam
      │
      ▼  (frame sent every ~1 second)
/api/process_frame
      │
      ▼
YOLOv8n PPE Model
      │
      ├── Green boxes → PPE items detected
      ├── Red boxes   → PPE items missing
      │
      ▼
Compliance Check
      │
      ├── COMPLIANT  → Banner turns green
      └── VIOLATION  → Banner turns red + logged to SQLite
```

---

## Features

- **Live Monitor** — browser webcam feed with real-time annotated detection overlay
- **Compliance Banner** — instant green/red status across the top of the video feed
- **PPE Checklist** — per-item status panel showing detected / missing for each gear type
- **Violation Logging** — every incident saved with timestamp, location, and people count
- **Dashboard** — today's compliance rate, 7-day trend chart, violations by PPE type
- **Violations Log** — filterable table by date and PPE type, CSV export
- **Settings** — configure which PPE items are required, detection confidence, site location
- **Lightweight** — no TensorFlow, no face recognition stack — deploys in ~3 minutes

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask 3.1 |
| PPE Detection | YOLOv8n — `keremberke/yolov8n-PPE-detection` |
| Image Processing | OpenCV (headless) |
| Database | SQLite |
| Charts | Chart.js |
| Frontend | HTML + CSS + Vanilla JS |
| Production Server | Gunicorn |
| Deployment | Railway (Docker) |

---

## Project Structure

```
PPE-Safety-AI/
│
├── app.py                   # Flask backend — all routes and detection logic
│
├── templates/
│   ├── dashboard.html       # Compliance stats and trend charts
│   ├── monitor.html         # Live webcam detection terminal
│   ├── violations.html      # Violation log with filters and hourly chart
│   └── settings.html        # PPE requirements and detection settings
│
├── static/
│   ├── css/style.css        # Dark theme design system
│   └── js/app.js            # Shared utilities (charts, toasts, table loader)
│
├── database/                # Auto-created on first run
│   └── ppe_safety.db        # SQLite — violations + scans tables
│
├── Dockerfile               # Python 3.11-slim container
├── Procfile                 # Gunicorn command
├── requirements.txt         # Python dependencies
└── railway.json             # Railway health check config
```

---

## Running Locally

**Requirements:** Python 3.11, a webcam

```bash
git clone https://github.com/ani14006/PPE-Safety-AI.git
cd PPE-Safety-AI

pip install -r requirements.txt

python app.py
```

Open `http://localhost:5002`

> On first run, the YOLOv8 PPE model (~6 MB) downloads automatically from HuggingFace and is cached locally.

---

## Deploying to Railway

Pre-configured for one-click Railway deployment via Docker.

1. Fork or push this repo to your GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select `PPE-Safety-AI` — Railway auto-detects the `Dockerfile`
4. Go to **Settings → Networking → Generate Domain**

**Build time: ~3–5 minutes**

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PPE_MODEL` | `keremberke/yolov8n-PPE-detection` | HuggingFace model ID or local `.pt` path |
| `DATABASE_DIR` | `database` | SQLite storage path. Set to a Railway Volume for persistence across deploys |
| `FLASK_DEBUG` | `0` | Set to `1` for local development only |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — `{"status":"ok","model_loaded":true}` |
| `POST` | `/api/process_frame` | Accepts base64 frame, returns annotated image + compliance status |
| `GET` | `/api/live_status` | Current detection state and any pending violation notification |
| `GET` | `/api/stats` | Today's scans, compliance rate, violations, people detected |
| `GET` | `/api/charts` | Chart data — 7-day compliance, violations by type, hourly breakdown |
| `GET` | `/api/violations/list` | Paginated violation log with `date` and `ppe_type` query filters |
| `GET` | `/api/export/csv` | Download all violations as a CSV file |
| `GET/POST` | `/api/settings` | Read or update detection configuration |

---

## Important Notes

- **Violations are logged every 5 seconds** while a non-compliant person remains in frame — not every detection frame — to avoid flooding the database
- **SQLite is ephemeral on Railway** — data resets on each redeploy. Mount a Railway Volume and set `DATABASE_DIR` to make it persistent
- **HTTPS is required** for browser webcam access. Railway provides HTTPS automatically on generated domains
- **Settings apply immediately** — no server restart needed to change required PPE items or confidence threshold

---

## License

MIT License — see [LICENSE](LICENSE) for details.
