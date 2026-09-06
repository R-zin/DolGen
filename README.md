# DolGen — 2D Floorplan → Path-Traced 3D Dollhouse

Upload a 2D floorplan image; a Modal backend uses **Gemini 2.5 Pro** to extract
walls / doors / windows / furniture as normalized 2D boxes, builds a `.glb`
model for each furniture piece (web asset library, **Gemini image model**
"nano banana pro", or procedural), assembles a textured dollhouse with
**Trimesh** (2.5 m walls, boolean-cut openings, white PBR walls, light-wood
floor), and streams back a `.glb`. A **React frontend** renders it with
**three-gpu-pathtracer** for progressive, photorealistic path tracing.

## One-time setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt        # installs the modal CLI

# Secrets (keys stay on your machine — never committed here):
modal secret create gemini-secret GEMINI_API_KEY=<your-gemini-key>
modal secret create asset-api-secret ASSET_API_TOKEN=<your-asset-library-token>
# optional, point at your own GLB asset library instead of Sketchfab:
#   modal secret create asset-api-secret ASSET_API_TOKEN=... ASSET_BASE_URL=https://your.api/v3

# optional, cache matched furniture GLBs across runs:
modal volume create dolgen-assets
```

## Run the backend

```bash
modal serve app.py     # hot-reload playground URL
modal deploy app.py    # persistent web endpoint
```

The same FastAPI app serves both the API (`POST /generate-3d`) and a legacy
zero-build viewer at `GET /`. You can also run it locally for development
(only the Gemini/asset steps need secrets):

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

Open http://localhost:5173, drop in a floorplan PNG/JPG, pick a **furniture
source**, and click **Generate**.

For a production build: `npm run build` → static files in `frontend/dist/`
(serve them anywhere; set `ALLOWED_ORIGINS` on the backend to your domain).

### Furniture sources

| Mode | Behavior |
|------|----------|
| `auto` (default) | Try the web asset library per piece; **Gemini image model generates** a `.glb` for any piece with no match; procedural mesh as the final fallback |
| `assets` | Web asset library only → procedural fallback |
| `ai` | **Gemini image model generates** every piece → procedural fallback |
| `procedural` | Built-in trimesh primitives only (no Gemini image calls, no network) |

## Architecture

| Step | Where | What |
|------|-------|------|
| Extraction | `extract_elements()` | `gemini-2.5-pro`, `temperature=0.0`, strict JSON schema → `[{element_type, furniture_class, asset_search_query, box_2d}]` with boxes normalized to 0-1000. `asset_search_query` is a rich type+material+color+style phrase so the fetched model matches the drawing; Gemini may also return a `room_colors` map for wall tinting |
| Assets | `download_assets()` | Sketchfab v3 flow, **match-oriented**: `GET /search?q=…&downloadable=true&count=K` → walk the top-K candidates → `POST /models/{uid}/download` → `GET` the glb url → accept the first candidate whose GLB actually downloads **and parses**. Cached on disk by query (Modal volume `dolgen-assets` if mounted) |
| AI models | `generate_ai_furniture_assets()` | For each furniture piece: crop its footprint out of the floorplan (`_crop_pad_floorplan`), have the Gemini image model (`GEMINI_IMAGE_MODEL`, default `gemini-3-pro-image-preview` — "nano banana pro") render a clean top-down view (`_gen_furniture_view`), texture-map it onto a procedural mesh (`_top_texture`), and export a self-contained `.glb` (`build_ai_furniture_glb`) that flows through the same placement path as a downloaded asset |
| Fallback | `build_procedural_furniture()` | Any furniture piece with no model is synthesized from trimesh primitives (bed / sofa / table / chair / desk / wardrobe / bathtub / toilet / sink / stove / fridge) so rooms are never empty |
| Assembly | `build_shell()` / `assemble_glb_bytes()` | 1000 units → 20 m; walls 2.5 m via `trimesh.creation.box`; doors/windows boolean-subtracted (manifold engine); furniture uniformly scaled to its footprint and grounded at z=0; white (or room-tinted) PBR walls, light-wood floor, optional ceiling slab |
| Serve | `POST /generate-3d` | `StreamingResponse` with `model/gltf-binary`; per-run pipeline notes in `X-DolGen-Log`, the furniture mode in `X-DolGen-Furniture`. Form fields: `file`, `ceiling`, `room_colors` (JSON map), `furniture_source` (`auto`/`assets`/`ai`/`procedural`). CORS is open to `ALLOWED_ORIGINS` (default `http://localhost:5173`) |
| Render | `frontend/` (React) | Vite + React 19. `PathTracerView.jsx` wraps `WebGLPathTracer` (0.0.24): `RoomEnvironment` IBL for lighting, `renderSample()` each frame, ACES tone mapping, orbit controls. UI: upload, furniture-source picker, ceiling toggle, auto-rotate, quality slider, `.glb` download |

## Notes

- **Sketchfab reality check**: the public Sketchfab API rarely allows downloading
  arbitrary `.glb`s (downloads need per-model grants). The client walks the top-`ASSET_TOP_K`
  (default 3) candidates and accepts the first that both downloads and parses. When nothing is
  downloadable, `auto`/`ai` modes generate the piece with the Gemini image model, and everything
  still falls back to a procedural placeholder so the room is never empty.
- **AI furniture is texture-mapped, not free-form geometry**: the Gemini image model renders a
  faithful top-down view of each piece; that image is projected onto a procedural mesh of the
  right class. The result reads correctly from the dollhouse's top-down camera and always has
  sane proportions, at a fraction of the cost/latency of true text-to-3D.
- **Furniture matching**: Gemini's `asset_search_query` is a specific type+material+color+style
  phrase; the matcher tries several downloadable candidates per piece and caches each successful
  GLB on disk (mount a Modal volume at `ASSET_CACHE_DIR`, default `/cache/assets`) to make repeat
  runs fast and consistent.
- The image's 20 MB upload cap and the 64 MB per-asset cap guard Modal memory.
- `GET /` works locally too for pure frontend iteration (only POST needs secrets).

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR:

- **Lint** — `ruff check` + `ruff format --check` over `app.py`, `main.py`, `tests/` (config in `pyproject.toml`).
- **Test** — byte-compiles the sources, then runs the offline pipeline tests in `tests/`
  on Python 3.10 / 3.11 / 3.12. The tests exercise the real geometry pipeline (shell build,
  procedural-furniture fallback, AI-furniture GLB/texture helpers, ceiling, room colors, GLB
  export) and need **no** Gemini key, asset token, or network.

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

