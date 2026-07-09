# PPE Safety AI

An AI-powered Personal Protective Equipment (PPE) compliance monitoring system. Point a camera at workers — the system detects in real-time whether they are wearing the required safety gear and logs every violation.

---

## What It Detects

| Detection | Meaning |
|---|---|
| ✅ Hardhat | Hard hat is being worn |
| ✅ Safety Vest | Hi-vis vest is being worn |
| ✅ Mask | Face mask is being worn |
| ❌ NO-Hardhat | Hard hat is **missing** → violation |
| ❌ NO-Safety Vest | Safety vest is **missing** → violation |
| ❌ NO-Mask | Mask is **missing** → violation |
| 👷 Person | Person detected |
| 🟠 Safety Cone / Machinery / Vehicle | Other site objects |

---

## How It Works

```
Browser webcam → frame sent to /api/process_frame every ~1 second
       ↓
YOLOv8n PPE model runs detection on the frame
       ↓
Compliance check: Is required PPE present or missing?
       ↓
Annotated frame returned + violation logged to SQLite
       ↓
Live monitor displays status: COMPLIANT or VIOLATION
```

The YOLO model (`keremberke/yolov8n-PPE-detection`) is trained specifically to detect both the presence AND absence of PPE items — it will draw green boxes around detected PPE and red boxes around missing PPE.

---

## Pages

### Dashboard `/`
Today's compliance rate, total scans, violation count, people detected. 7-day compliance trend chart and violations-by-type donut chart.

### Live Monitor `/monitor`
Start your browser webcam. Frames are sent to the server for detection every second. The right panel shows real-time PPE status for each item (detected / missing). A compliance banner turns green for COMPLIANT and red for VIOLATION.

### Violations Log `/violations`
Full searchable log of all violations with date and PPE-type filters. Hourly bar chart for today. Export to CSV.

### Settings `/settings`
Configure which PPE items are required (hard hat, vest, mask), the YOLO confidence threshold, and the site location name.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Flask |
| PPE Detection | YOLOv8n (Ultralytics) |
| Model | keremberke/yolov8n-PPE-detection (HuggingFace) |
| Image Processing | OpenCV (headless) |
| Database | SQLite |
| Frontend | HTML + CSS + Vanilla JS + Chart.js |
| Production Server | Gunicorn |
| Deployment | Railway (Docker) |

**No TensorFlow, no DeepFace — much lighter than a face recognition system.**

---

## Project Structure

```
PPE-Safety-AI/
├── app.py                  # Flask application — all routes and detection logic
│
├── templates/
│   ├── dashboard.html      # Stats and charts overview
│   ├── monitor.html        # Live webcam detection
│   ├── violations.html     # Violation log with filters
│   └── settings.html       # Configure PPE requirements
│
├── static/
│   ├── css/style.css       # Dark theme UI
│   └── js/app.js           # Shared JS utilities
│
├── database/               # Auto-created on first run
│   └── ppe_safety.db       # SQLite database
│
├── Dockerfile              # Python 3.11-slim for Railway
├── Procfile                # Gunicorn start command
├── requirements.txt        # Python dependencies
└── railway.json            # Health check config
```

---

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/PPE-Safety-AI.git
cd PPE-Safety-AI

pip install -r requirements.txt

python app.py
```

Open `http://localhost:5002`

> On first run the YOLO PPE model (~6 MB) is downloaded automatically from HuggingFace and cached locally.

---

## Deploying to Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select this repository — Railway auto-detects the `Dockerfile`
4. Go to **Settings → Networking → Generate Domain** for your public URL

**Build time:** ~3–5 minutes (much faster than SmartHR AI — no TensorFlow)

### Environment Variables (optional)

| Variable | Default | Description |
|---|---|---|
| `PPE_MODEL` | `keremberke/yolov8n-PPE-detection` | HuggingFace model ID or local `.pt` path |
| `DATABASE_DIR` | `database` | Path for SQLite DB. Use a Railway Volume for persistence |
| `FLASK_DEBUG` | `0` | Set to `1` for local development |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — `{"status": "ok", "model_loaded": true}` |
| POST | `/api/process_frame` | Process a webcam frame, returns annotated image + status |
| GET | `/api/live_status` | Current detection status + any pending notification |
| GET | `/api/stats` | Dashboard stats (scans, compliance rate, violations, people) |
| GET | `/api/charts` | Chart data (daily compliance, violation types, hourly) |
| GET | `/api/violations/list` | Violation log with optional `date` and `ppe_type` filters |
| GET | `/api/export/csv` | Download violations as CSV |
| GET/POST | `/api/settings` | Get or update detection settings |

---

## Notes

- The SQLite database is **ephemeral on Railway** by default — it resets on each deploy. Mount a Railway Volume and set `DATABASE_DIR` to make it persistent.
- The browser needs **camera permission**. On Railway (HTTPS), this works automatically.
- Violations are logged **every 5 seconds** while a non-compliant person is in frame — not every frame — to avoid flooding the database.
