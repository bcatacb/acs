# ACS (True Accompaniment)

Web-first app for acapella upload, analysis, and style-based accompaniment generation.

## Quick Start

### 1. Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:
```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=acs
CORS_ORIGINS=*
JWT_SECRET=change_this_secret
SUNO_API_KEY=optional_legacy_key
```

Run API:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend
```bash
cd frontend
npm install
```

Create `frontend/.env`:
```env
REACT_APP_BACKEND_URL=http://localhost:8000
WDS_SOCKET_PORT=3000
ENABLE_HEALTH_CHECK=false
```

Run web app:
```bash
npm start
```

## Asset Index (SQLite)

Index local loops/samples/MIDI:
```bash
cd backend
python scripts/ingest_assets.py ../asset_drop
```

## Notes
- Keep `.env` files local only (do not commit secrets).
- API health endpoint: `GET /api/health`
