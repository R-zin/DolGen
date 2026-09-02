# DolGen — 2D Floorplan → Path-Traced 3D Dollhouse

Upload a 2D floorplan image; a Modal backend uses **Gemini 2.5 Pro** to extract
walls / doors / windows / furniture as normalized 2D boxes, downloads matching
3D furniture assets from a Sketchfab-compatible API, assembles a textured
dollhouse with **Trimesh** (2.5 m walls, boolean-cut openings, white PBR walls,
light-wood floor), and streams back a `.glb`. The frontend renders it with a
custom Three.js viewer using **three-gpu-pathtracer** for progressive,
photorealistic path tracing.

## One-time setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -r requirements.txt        # installs the modal CLI

# Secrets (keys stay on your machine — never committed here):
modal secret create gemini-secret GEMINI_API_KEY=<your-gemini-key>
modal secret create asset-api-secret ASSET_API_TOKEN=<your-asset-library-token>
# optional, point at your own GLB asset library instead of Sketchfab:
#   modal secret create asset-api-secret ASSET_API_TOKEN=... ASSET_BASE_URL=https://your.api/v3

# optional, cache matched furniture GLBs across runs:
modal volume create dolgen-assets
```

## Run

```bash
modal serve app.py     # hot-reload playground URL
modal deploy app.py    # persistent web endpoint
```

Open the printed URL, drop in a floorplan PNG/JPG, and click **Generate**.

## Architecture

| Step | Where | What |
|------|-------|------|
| Extraction | `extract_elements()` | `gemini-2.5-pro`, `temperature=0.0`, strict JSON schema → `[{element_type, furniture_class, asset_search_query, box_2d}]` with boxes normalized to 0-1000. `asset_search_query` is a rich type+material+color+style phrase so the fetched model matches the drawing; Gemini may also return a `room_colors` map for wall tinting |
| Assets | `download_assets()` | Sketchfab v3 flow, now **match-oriented**: `GET /search?q=…&downloadable=true&count=K` → walk the top-K candidates → `POST /models/{uid}/download` → `GET` the glb url → accept the first candidate whose GLB actually downloads **and parses**. Successful GLBs are cached on disk by query (Modal volume `dolgen-assets` if mounted) |
| Fallback | `build_procedural_furniture()` | Any furniture piece with **no** matching downloadable asset is synthesized from trimesh primitives (bed / sofa / table / chair / desk / wardrobe / bathtub / toilet / sink / stove / fridge) so rooms are never empty |
| Assembly | `build_shell()` / `assemble_glb_bytes()` | 1000 units → 20 m; walls 2.5 m via `trimesh.creation.box`; doors/windows boolean-subtracted (manifold engine); furniture uniformly scaled to its footprint and grounded at z=0; white (or room-tinted) PBR walls, light-wood floor, optional ceiling slab |
| Serve | `POST /generate-3d` | `StreamingResponse` with `model/gltf-binary`; per-run pipeline notes in `X-DolGen-Log` header. Form fields: `file`, `ceiling` (`true/false`), `room_colors` (JSON map) |
| Render | `GET /` | ES-module import map: `three`, `GLTFLoader`, `OrbitControls`, `three-mesh-bvh`, `WebGLPathTracer` from unpkg; `ACESFilmicToneMapping`, `pathTracer.setScene(gltf.scene, camera)`, `renderSample()` each frame. UI: ceiling toggle, auto-rotate, quality (target-samples) slider, and a `.glb` download button |

## Notes

- **Sketchfab reality check**: the public Sketchfab API rarely allows downloading
  arbitrary `.glb`s (downloads need per-model grants). The client walks the top-`ASSET_TOP_K`
  (default 3) candidates and accepts the first that both downloads and parses, which helps —
  but when nothing is downloadable the app now **synthesizes a procedural placeholder** for that
  piece instead of leaving the room empty, and still returns the full wall/floor scene. Set
  `ASSET_BASE_URL` to any GLB-capable asset library to get real furniture.
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
  procedural-furniture fallback, ceiling, room colors, GLB export) and need **no** Gemini key,
  asset token, or network.

Run the same checks locally:

```bash
pip install -r requirements-dev.txt
ruff check app.py main.py tests
pytest -q
```
