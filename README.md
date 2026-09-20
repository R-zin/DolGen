# DolGen — 2D Floorplan → Path-Traced 3D Dollhouse (structure only)

Upload a 2D floorplan image; a Modal backend **preprocesses it with OpenCV**
(EXIF fix, upscale, denoise, white balance, and a binary wall mask that keeps
the thick structural partitions and discards furniture strokes, hatching, and
symbols), then uses **Kimi K3** (Moonshot AI's vision model, self-hosted on
Modal behind an OpenAI-compatible endpoint) to extract the *structure of the
house* — walls / doors / windows only, no furniture — as normalized 2D boxes.
It assembles the shell with **Trimesh** (2.5 m walls, boolean-cut openings,
white PBR walls, light-wood floor) and streams back a `.glb`. A **React
frontend** renders it with **three-gpu-pathtracer** for progressive,
photorealistic path tracing.

## One-time setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt        # installs the modal CLI

# Secrets (keys stay on your machine — never committed here).
# Point KIMI_BASE_URL at YOUR deployed Kimi K3 endpoint. The token pair can be
# Kimi-specific (KIMI_TOKEN_ID/KIMI_TOKEN_SECRET) or omitted, in which case the
# app uses your Modal workspace tokens (MODAL_TOKEN_ID/MODAL_TOKEN_SECRET):
modal secret create kimi-verify \
  KIMI_BASE_URL=https://<your-kimi-endpoint>/v1 \
  KIMI_TOKEN_ID=<token-id> \
  KIMI_TOKEN_SECRET=<token-secret>
```

## Run the backend

```bash
modal serve app.py     # hot-reload playground URL
modal deploy app.py    # persistent web endpoint
```

The same FastAPI app serves both the API (`POST /generate-3d`) and a legacy
zero-build viewer at `GET /`. You can also run it locally for development
(only the Kimi extraction step needs the secret):

```bash
uvicorn app:create_app --factory --reload --port 8000
```

## Run the React frontend

```bash
cd frontend
npm install
# point the dev proxy at your backend (defaults to http://localhost:8000):
VITE_BACKEND_URL=http://localhost:8000 npm run dev
# or against a deployed Modal URL:
VITE_BACKEND_URL=https://<your-modal-url>.modal.run npm run dev
```

Open http://localhost:5173, drop in a floorplan PNG/JPG, and click
**Generate**.

For a production build: `npm run build` → static files in `frontend/dist/`
(serve them anywhere; set `ALLOWED_ORIGINS` on the backend to your domain).

## Architecture

| Step | Where | What |
|------|-------|------|
| Preprocessing | `preprocess_floorplan()` / `build_wall_mask()` | OpenCV pipeline on the raw upload: EXIF orientation → upscale to ~2048 px → gray-world white balance → `fastNlMeansDenoisingColored` → CLAHE contrast → Otsu threshold on a blurred gray image → 3 px morphological opening → small-connected-component removal (`_WALL_MASK_MIN_AREA_FRAC`, 5e-4 of image area). The result is a binary mask: thick dark partitions (walls) black on white, furniture/text/fills gone. Any failure falls back to the raw upload bytes |
| Extraction | `extract_elements()` | Kimi K3 (`KIMI_MODEL`, default `kimi-k3`) over an OpenAI-compatible endpoint on Modal, `temperature=0.0`, strict JSON schema (with graceful fallback to plain JSON mode) → `[{element_type, box_2d}]` with `element_type ∈ {wall, door, window}` and boxes normalized to 0-1000. The prompt is structure-only: furniture, text, and dimension lines are explicitly out of scope |
| Assembly | `build_shell()` / `assemble_glb_bytes()` | 1000 units → 20 m; walls 2.5 m via `trimesh.creation.box`; doors/windows boolean-subtracted (manifold engine); white (or room-tinted) PBR walls, light-wood floor, optional ceiling slab |
| Serve | `POST /generate-3d` | `StreamingResponse` with `model/gltf-binary`; per-run pipeline notes in `X-DolGen-Log`. Form fields: `file`, `ceiling`, `room_colors` (JSON map). CORS is open to `ALLOWED_ORIGINS` (default `http://localhost:5173`) |
| Render | `frontend/` (React) | Vite + React 19. `PathTracerView.jsx` wraps `WebGLPathTracer`: `RoomEnvironment` IBL for lighting, `renderSample()` each frame, ACES tone mapping, orbit controls. UI: upload, ceiling toggle, auto-rotate, quality slider, `.glb` download |

## Notes

- **Structure only**: furniture extraction and placement were removed so the
  vision model spends all of its attention on walls, doors, and windows — the
  parts that determine whether the dollhouse is right.
- **Why the wall mask**: on raw colored scans, the vision model tends to trace
  furniture outlines as walls or miss thin partitions. Feeding it a normalized
  binary mask of the structural ink makes wall boxes measurably tighter and
  more complete, and door/window gaps read as actual breaks in the wall stroke.
- **Kimi endpoint**: the app talks to Kimi K3 through the `openai` Python
  client pointed at `KIMI_BASE_URL` (from the `kimi-verify` secret). The API
  key is the `KIMI_TOKEN_ID:KIMI_TOKEN_SECRET` pair when those are set,
  otherwise it falls back to the Modal workspace tokens
  `MODAL_TOKEN_ID:MODAL_TOKEN_SECRET` — which is what a Modal-hosted endpoint
  in this workspace typically expects, so the secret usually only needs
  `KIMI_BASE_URL` (and optionally `KIMI_MODEL` / `KIMI_TIMEOUT_S`). Because
  there is no public Modal-hosted URL, you must deploy your own Kimi K3
  endpoint (or any OpenAI-compatible vision endpoint serving that model).
- The 20 MB upload cap guards Modal memory.
- `GET /` works locally too for pure frontend iteration (only POST needs secrets).

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR:

- **Lint** — `ruff check` + `ruff format --check` over `app.py`, `main.py`, `tests/` (config in `pyproject.toml`).
- **Test** — byte-compiles the sources, then runs the offline pipeline tests in `tests/`
  on Python 3.10 / 3.11 / 3.12. The tests exercise the real geometry pipeline (shell build,
  ceiling, room colors, GLB export), the OpenCV preprocessing helpers, and the Kimi
  response parsing/schema helpers, and need **no** Kimi credentials or network.

Run the same checks locally:

```bash
uv pip install -r requirements-dev.txt
ruff check app.py main.py tests
pytest -q
```

Frontend checks:

```bash
cd frontend
npm run lint    # oxlint
npm run test    # vitest + testing-library
npm run build
```
