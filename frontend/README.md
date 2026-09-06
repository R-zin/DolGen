# DolGen frontend

React 19 + Vite viewer for the DolGen floorplan → 3D dollhouse backend.
Renders the returned `.glb` with **three-gpu-pathtracer** (progressive path
tracing) and lets you pick how furniture models are produced (web asset
library, Gemini "nano banana pro" image model, or procedural).

## Develop

```bash
npm install
# point the dev proxy at a backend (default http://localhost:8000):
VITE_BACKEND_URL=http://localhost:8000 npm run dev
# or a deployed Modal URL:
VITE_BACKEND_URL=https://<your-modal-url>.modal.run npm run dev
```

Open http://localhost:5173. The dev server proxies `/generate-3d` and `/api`
to the backend, so no CORS is needed in development.

## Scripts

| Command | What |
|---------|------|
| `npm run dev` | Vite dev server with backend proxy |
| `npm run build` | Production build → `dist/` |
| `npm run preview` | Preview the production build |
| `npm run lint` | oxlint |
| `npm run test` | vitest + @testing-library/react |

## Structure

- `src/App.jsx` — panel UI (upload, furniture-source picker, options, log) and
  the `POST /generate-3d` call.
- `src/PathTracerView.jsx` — the three-gpu-pathtracer viewport (RoomEnvironment
  IBL, per-frame `renderSample()`, ceiling toggle, camera fitting).

## Deploy

`npm run build` emits a static site in `dist/`. Host it anywhere and set the
backend's `ALLOWED_ORIGINS` env var to your frontend origin so `POST
/generate-3d` accepts cross-origin requests and exposes the `X-DolGen-*`
headers.
