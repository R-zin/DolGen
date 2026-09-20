"""Offline tests for DolGen's geometry/assembly pipeline.

These never touch the Kimi endpoint — they build a small synthetic floorplan
element list and run the real builders end to end, asserting the scene graph,
the GLB export, the ceiling/color logic, the OpenCV preprocessing helpers, and
the Kimi response parsing/schema helpers.
"""

from __future__ import annotations

import pytest

import app

# A simple one-room floorplan: four walls, a door, and a window.
ELEMENTS = [
    # exterior walls (thin segments on a 0-1000 grid); door/window boxes sit
    # mid-span on a single wall so the boolean cut never splits a wall in two.
    {"element_type": "wall", "box_2d": [100, 100, 130, 900]},   # top
    {"element_type": "wall", "box_2d": [870, 100, 900, 900]},   # bottom
    {"element_type": "wall", "box_2d": [100, 100, 900, 130]},   # left
    {"element_type": "wall", "box_2d": [100, 870, 900, 900]},   # right
    # a door mid-span on the bottom wall, a window mid-span on the left wall
    {"element_type": "door", "box_2d": [878, 300, 892, 420]},
    {"element_type": "window", "box_2d": [400, 108, 560, 122]},
]


def test_normalize_elements_drops_degenerate():
    els = [
        app.FloorElement(element_type="wall", box_2d=[0, 0, 2, 2]),       # degenerate -> dropped
        app.FloorElement(element_type="wall", box_2d=[10, 10, 200, 40]),  # fine
    ]
    out = app.normalize_elements(els)
    assert len(out) == 1
    assert out[0]["box_2d"] == [10, 10, 200, 40]


def test_normalize_elements_clamps_and_sorts():
    els = [app.FloorElement(element_type="wall", box_2d=[950, -20, 1200, 500])]
    out = app.normalize_elements(els)
    ymin, xmin, ymax, xmax = out[0]["box_2d"]
    assert 0 <= ymin <= ymax <= 1000
    assert 0 <= xmin <= xmax <= 1000


def test_scale_mesh_single_transform():
    # A box from (0,0)-(1000,1000) must map to a 20 m x 20 m square.
    m = app.scale_mesh([0, 0, 1000, 1000], 0.0, 2.5)
    assert m.extents[0] == pytest.approx(20.0, rel=1e-3)
    assert m.extents[1] == pytest.approx(20.0, rel=1e-3)
    assert m.extents[2] == pytest.approx(2.5, rel=1e-3)


def test_opening_hparams_window_vs_door():
    assert app.opening_hparams([0, 0, 0, 0], "door") == (0.0, 2.1)
    sill, head = app.opening_hparams([0, 0, 0, 0], "window")
    assert sill == pytest.approx(0.9)
    assert head == pytest.approx(2.1)


def test_build_shell_has_floor_and_walls():
    scene, logs = app.build_shell(ELEMENTS)
    assert "floor" in scene.geometry
    wall_nodes = [n for n in scene.geometry if n.startswith("wall_")]
    assert len(wall_nodes) == 4
    assert any("4 walls" in line for line in logs)


def test_assemble_exports_valid_glb():
    glb, logs = app.assemble_glb_bytes(ELEMENTS)
    assert isinstance(glb, (bytes, bytearray)) and len(glb) > 0
    assert glb[:4] == b"glTF"
    assert any(line.startswith("Exported dollhouse GLB") for line in logs)


def test_assemble_with_ceiling_adds_named_node():
    glb, logs = app.assemble_glb_bytes(ELEMENTS, include_ceiling=True)
    assert any("ceiling" in line.lower() for line in logs)
    scene = app.trimesh.load(app.io.BytesIO(glb), file_type="glb", force="scene", process=False)
    names = set(scene.geometry.keys())
    assert any("ceiling" in n for n in names)


def test_room_wall_tint_applies():
    glb, logs = app.assemble_glb_bytes(ELEMENTS, room_colors={"bedroom": "#3a6ea5"})
    assert any("room wall tint" in line for line in logs)
    assert glb[:4] == b"glTF"


def test_hex_to_rgba_valid_and_invalid():
    assert app._hex_to_rgba("#3a6ea5") is not None
    assert app._hex_to_rgba("abc") is not None
    assert app._hex_to_rgba("not-a-color") is None
    assert app._hex_to_rgba(123) is None


# --- Preprocessing (OpenCV wall mask) ---------------------------------------

def _make_png(w: int = 64, h: int = 48, rgb=(255, 255, 255)) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (w, h), rgb)
    buf = app.io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _synthetic_floorplan_png(w: int = 400, h: int = 300) -> bytes:
    """White plan with thick black walls, a small square (furniture), and noise."""
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # outer walls, ~8 px thick (well above the 3 px opening pass)
    d.rectangle([40, 40, w - 41, h - 41], outline=(0, 0, 0), width=8)
    # one interior wall
    d.line([w // 2, 40, w // 2, h - 41], fill=(0, 0, 0), width=8)
    # small furniture-like square inside a room (should be removed by the mask)
    d.rectangle([70, 70, 100, 100], outline=(0, 0, 0), width=2)
    arr = np.asarray(img).astype(np.int16)
    arr += np.random.default_rng(7).integers(-12, 13, arr.shape, dtype=np.int16)
    img = Image.fromarray(arr.clip(0, 255).astype("uint8"))
    buf = app.io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_wall_mask_keeps_walls_drops_small_components():
    png = _synthetic_floorplan_png()
    bgr = app._bytes_to_bgr(png)
    assert bgr is not None
    gray = app._normalize_floorplan(bgr)
    mask = app.build_wall_mask(gray)
    assert mask.shape == gray.shape

    dark = mask < 128
    # the long walls survive as dark lines; the small furniture square does not
    assert dark[mask.shape[0] // 2, mask.shape[1] // 2]  # interior wall midline
    assert not dark[85 * mask.shape[0] // 300, 85 * mask.shape[1] // 400]  # furniture square
    # mask is strictly binary
    assert set(app.np.unique(mask).tolist()) <= {0, 255}


def test_preprocess_floorplan_returns_png_bytes():
    png = _synthetic_floorplan_png()
    out, mime = app.preprocess_floorplan(png)
    assert mime == "image/png"
    assert out[:8] == b"\x89PNG\r\n\x1a\n"


def test_preprocess_floorplan_garbage_falls_back():
    out, mime = app.preprocess_floorplan(b"not an image at all")
    assert out == b"not an image at all"
    assert mime == ""


# --- Kimi response parsing / schema (offline) -------------------------------

def test_parse_elements_text_raw_array():
    text = '[{"element_type": "wall", "box_2d": [10, 10, 200, 40]}]'
    out = app._parse_elements_text(text)
    assert len(out) == 1
    assert out[0].element_type == "wall"
    assert out[0].box_2d == [10, 10, 200, 40]


def test_parse_elements_text_wrapped_object():
    text = '{"elements": [{"element_type": "door", "box_2d": [1, 2, 3, 4]}]}'
    out = app._parse_elements_text(text)
    assert len(out) == 1
    assert out[0].element_type == "door"


def test_parse_elements_text_fenced_json():
    text = '```json\n[{"element_type": "window", "box_2d": [5, 6, 7, 8]}]\n```'
    out = app._parse_elements_text(text)
    assert len(out) == 1
    assert out[0].element_type == "window"


def test_parse_elements_text_drops_invalid_entries():
    text = (
        '[{"element_type": "wall", "box_2d": [1, 2, 3, 4]},'
        ' {"element_type": "sofa", "box_2d": [1, 2, 3, 4]},'
        ' {"element_type": "wall", "box_2d": [1, 2, 3]}]'
    )
    out = app._parse_elements_text(text)
    # only the valid wall survives; bad type and bad box are dropped
    assert len(out) == 1
    assert out[0].element_type == "wall"


def test_parse_elements_text_garbage_returns_empty():
    assert app._parse_elements_text("") == []
    assert app._parse_elements_text(None) == []
    assert app._parse_elements_text("not json") == []
    assert app._parse_elements_text('{"a": 1}') == []


def test_elements_json_schema_is_array_of_floor_element():
    spec = app._elements_json_schema()
    assert spec["schema"]["type"] == "array"
    assert spec["strict"] is True
    items = spec["schema"]["items"]
    assert set(items["properties"]) == {"element_type", "box_2d"}
    assert set(items["required"]) == {"element_type", "box_2d"}


def test_kimi_client_token_fallback(monkeypatch):
    # Kimi-specific pair wins when both are set
    monkeypatch.setenv("KIMI_TOKEN_ID", "kid")
    monkeypatch.setenv("KIMI_TOKEN_SECRET", "ksec")
    monkeypatch.setenv("MODAL_TOKEN_ID", "mid")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "msec")
    monkeypatch.setattr(app, "KIMI_BASE_URL", "https://example.test/v1")
    client = app._kimi_client()
    assert client.api_key == "kid:ksec"

    # falls back to the Modal workspace tokens
    monkeypatch.delenv("KIMI_TOKEN_ID")
    monkeypatch.delenv("KIMI_TOKEN_SECRET")
    client = app._kimi_client()
    assert client.api_key == "mid:msec"


def test_kimi_client_missing_config_raises(monkeypatch):
    for var in ("KIMI_TOKEN_ID", "KIMI_TOKEN_SECRET", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(app, "KIMI_BASE_URL", "")
    with pytest.raises(app.HTTPException) as exc:
        app._kimi_client()
    assert exc.value.status_code == 500
