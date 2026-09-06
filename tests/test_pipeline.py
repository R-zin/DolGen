"""Offline tests for DolGen's geometry/assembly pipeline.

These never touch Gemini or the asset API — they build a small synthetic
floorplan element list and run the real builders end to end, asserting the
scene graph, the GLB export, and the furniture fallback/ceiling/color logic.
"""

from __future__ import annotations

import pytest

import app

# A simple one-room floorplan: four walls, a door, a window, and furniture.
ELEMENTS = [
    # exterior walls (thin segments on a 0-1000 grid); door/window boxes sit
    # mid-span on a single wall so the boolean cut never splits a wall in two.
    {"element_type": "wall", "furniture_class": None, "asset_search_query": None, "box_2d": [100, 100, 130, 900]},   # top
    {"element_type": "wall", "furniture_class": None, "asset_search_query": None, "box_2d": [870, 100, 900, 900]},   # bottom
    {"element_type": "wall", "furniture_class": None, "asset_search_query": None, "box_2d": [100, 100, 900, 130]},   # left
    {"element_type": "wall", "furniture_class": None, "asset_search_query": None, "box_2d": [100, 870, 900, 900]},   # right
    # a door mid-span on the bottom wall, a window mid-span on the left wall
    {"element_type": "door", "furniture_class": None, "asset_search_query": None, "box_2d": [878, 300, 892, 420]},
    {"element_type": "window", "furniture_class": None, "asset_search_query": None, "box_2d": [400, 108, 560, 122]},
    # furniture (no asset download in tests -> all become procedural)
    {"element_type": "furniture", "furniture_class": "bed", "asset_search_query": "oak queen bed frame", "box_2d": [200, 200, 420, 400]},
    {"element_type": "furniture", "furniture_class": "sofa", "asset_search_query": "grey fabric sofa", "box_2d": [600, 600, 800, 800]},
    {"element_type": "furniture", "furniture_class": "mystery_object", "asset_search_query": "weird thing", "box_2d": [500, 150, 600, 250]},
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


@pytest.mark.parametrize("cls", [
    "bed", "sofa", "dining_table", "chair", "desk", "wardrobe",
    "bathtub", "toilet", "sink", "stove", "fridge", "totally_unknown",
])
def test_procedural_furniture_builds_valid_mesh(cls):
    mesh = app.build_procedural_furniture(cls, [0, 0, 100, 100])
    assert mesh.extents.max() > 0.05
    placed = app.normalize_furniture_to_box(mesh, [200, 200, 400, 400])
    # grounded on the floor
    assert placed.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    # fits inside its footprint
    assert placed.extents[0] <= 200 * app.SCALE + 1e-6
    assert placed.extents[1] <= 200 * app.SCALE + 1e-6


def test_assemble_without_assets_uses_procedural_fallback():
    glb, logs = app.assemble_glb_bytes(ELEMENTS, furniture_assets=[])
    assert isinstance(glb, (bytes, bytearray)) and len(glb) > 0
    assert glb[:4] == b"glTF"
    # 3 furniture pieces, 0 real assets -> 3 procedural placeholders
    assert any("procedural placeholder(s)" in line for line in logs)
    assert any(line.startswith("Exported dollhouse GLB") for line in logs)


def test_assemble_with_ceiling_adds_named_node():
    glb, logs = app.assemble_glb_bytes(ELEMENTS, [], include_ceiling=True)
    assert any("ceiling" in line.lower() for line in logs)
    scene = app.trimesh.load(app.io.BytesIO(glb), file_type="glb", force="scene", process=False)
    names = set(scene.geometry.keys())
    assert any("ceiling" in n for n in names)


def test_room_wall_tint_applies():
    glb, logs = app.assemble_glb_bytes(ELEMENTS, [], room_colors={"bedroom": "#3a6ea5"})
    assert any("room wall tint" in line for line in logs)
    assert glb[:4] == b"glTF"


def test_hex_to_rgba_valid_and_invalid():
    assert app._hex_to_rgba("#3a6ea5") is not None
    assert app._hex_to_rgba("abc") is not None
    assert app._hex_to_rgba("not-a-color") is None
    assert app._hex_to_rgba(123) is None


def test_glb_validator_rejects_garbage():
    assert app._looks_like_glb(b"glTF" + b"\x00" * 20)
    assert not app._looks_like_glb(b"not a glb")
    assert not app._glb_parses(b"glTF" + b"\x00" * 20)


def test_cache_key_is_stable():
    a = app._cache_key("Oak Queen Bed Frame")
    b = app._cache_key("  oak queen bed frame ")
    assert a == b and a.endswith(".glb")


# --- AI-generated furniture (Gemini image model "nano banana pro") ----------

# 1x1 white PNG
_PNG_1PX = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x00\x03\x00\x01\x00\x05\xfe\xd4\x8f\x1d\x00\x00\x00\x00IEND"
    b"\xaeB`\x82"
)


def _make_png(w: int = 64, h: int = 48, rgb=(255, 255, 255)) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (w, h), rgb)
    buf = app.io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_crop_pad_floorplan_returns_square_png():
    plan = _make_png(1000, 1000)
    out = app._crop_pad_floorplan(plan, [200, 200, 420, 400])
    assert out is not None
    from PIL import Image

    im = Image.open(app.io.BytesIO(out))
    assert im.size[0] == im.size[1]  # square canvas
    assert im.size[0] > 0


def test_crop_pad_floorplan_degenerate_returns_none():
    plan = _make_png(100, 100)
    assert app._crop_pad_floorplan(plan, [0, 0, 0, 0]) is None
    assert app._crop_pad_floorplan(b"not an image", [0, 0, 100, 100]) is None


def test_build_ai_furniture_glb_produces_valid_glb():
    tex = _make_png(32, 32, (180, 120, 80))
    data = app.build_ai_furniture_glb("sofa", [0, 0, 100, 100], tex)
    assert data is not None
    assert app._looks_like_glb(data)
    assert app._glb_parses(data)
    # it loads as a mesh with a texture applied
    mesh = app.trimesh.load(app.io.BytesIO(data), file_type="glb", force="mesh", process=False)
    assert mesh.visual is not None


def test_build_ai_furniture_glb_bad_texture_returns_none():
    assert app.build_ai_furniture_glb("bed", [0, 0, 100, 100], b"garbage") is None


def test_image_response_may_retry_codes():
    class FakeErr(Exception):
        def __init__(self, code):
            self.code = code
            super().__init__(str(code))

    if app.GenAIClientError is None:
        import pytest as _p

        _p.skip("google-genai ClientError not available")
    # Build real ClientError instances
    from google.genai.errors import ClientError

    e429 = ClientError(code=429, response_json={"error": {"message": "quota"}})
    e400 = ClientError(code=400, response_json={"error": {"message": "bad"}})
    assert app._image_response_may_retry(e429) is True
    assert app._image_response_may_retry(e400) is False
    assert app._image_response_may_retry(FakeErr(503)) is False  # not a ClientError
