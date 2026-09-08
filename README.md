# FedSentinel

FedSentinel is an impact-aware security layer for simulated federated learning. The integrated application contains the P1 federated-learning core, P2 attack scenarios, P3 Sentinel intelligence pipeline, P4 FastAPI orchestration/persistence, and a backend-connected React analytics dashboard.

## Run the backend

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

The API and OpenAPI documentation are available at `http://localhost:8000` and `http://localhost:8000/docs`.

## Run the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api/v1` to the backend during development. For another deployment URL, copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_BASE_URL`.

## Dashboard data policy

The frontend contains no production mock data. It reads simulation, round, client, threat, impact, metric, recovery, and audit data from the FastAPI routes. Missing backend values are shown as an em dash. The UI refreshes every three seconds when auto-refresh is enabled.

Exports are produced locally in the browser:

- complete run JSON;
- joined threat/impact analytics CSV;
- recovery metrics CSV.

## Verification

```powershell
cd frontend
npm run build
npm test
```
