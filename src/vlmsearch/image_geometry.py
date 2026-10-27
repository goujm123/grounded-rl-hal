"""Explicit image transport and point-coordinate contracts for teacher traces.

The teacher sees a resized image and emits a normalized 0..1000 point grid.
Student traces refer to zero-based pixels of the untouched original image.
No coordinate frame is guessed from the numerical values in a legacy trace.
"""

from collections import OrderedDict
from copy import deepcopy
import base64
import hashlib
from io import BytesIO
import math
from pathlib import Path
import re
import weakref

from PIL import Image


NORMALIZED_COORDINATE_INSTRUCTION = (
    "Coordinates use a normalized 0-1000 grid on both axes: (0, 0) is the top-left "
    "and (1000, 1000) is the bottom-right. Always write coordinates as (x, y), "
    "never as raw image pixels."
)
PIXEL_COORDINATE_INSTRUCTION = (
    "Coordinates refer to zero-based pixels in the original image: (0, 0) is the "
    "top-left and (width - 1, height - 1) is the bottom-right. Always write "
    "coordinates as (x, y)."
)

PAIR_RE = re.compile(r"\(\s*([^(),\n]+?)\s*,\s*([^(),\n]+?)\s*\)")
PAREN_RE = re.compile(r"\(([^()\n]*)\)")
NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.I)
NUMERIC_START_RE = re.compile(r"^[+\-\N{MINUS SIGN}]?(?:\d|\.|nan\b|inf(?:inity)?\b)", re.I)


class CoordinateError(ValueError):
    """A trace violates its declared coordinate contract."""


def _size(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CoordinateError(f"invalid_{name}")
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in value):
        raise CoordinateError(f"invalid_{name}")
    return tuple(value)


def smart_resize_size(width, height, min_pixels=65536, max_pixels=1048576, factor=32):
    """Match the pinned Qwen image processor's factor-aligned area resize."""
    if min(width, height, min_pixels, max_pixels, factor) <= 0 or min_pixels > max_pixels:
        raise ValueError("Image dimensions and ordered pixel bounds must be positive")
    if max(width, height) / min(width, height) > 200:
        raise ValueError("Image aspect ratio exceeds the Qwen processor limit")
    out_h = max(factor, round(height / factor) * factor)
    out_w = max(factor, round(width / factor) * factor)
    if out_h * out_w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        out_h = max(factor, math.floor(height / beta / factor) * factor)
        out_w = max(factor, math.floor(width / beta / factor) * factor)
    elif out_h * out_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        out_h = math.ceil(height * beta / factor) * factor
        out_w = math.ceil(width * beta / factor) * factor
    if not min_pixels <= out_h * out_w <= max_pixels:
        raise ValueError("Pixel bounds cannot be represented for this aspect ratio and resize factor")
    return out_w, out_h


def validate_image_geometry(geometry, original_size=None):
    """Validate schema and dimensions; optionally verify against the source image."""
    if not isinstance(geometry, dict) or geometry.get("version") != 1:
        raise CoordinateError("invalid_image_geometry_version")
    if geometry.get("coordinate_frame") != "normalized_1000":
        raise CoordinateError("unsupported_coordinate_frame")
    if geometry.get("coordinate_extent") != [1000, 1000]:
        raise CoordinateError("invalid_coordinate_extent")
    source = _size(geometry.get("original_size"), "original_size")
    if original_size is not None and source != _size(original_size, "observed_original_size"):
        raise CoordinateError("original_size_mismatch")
    transport = _size(geometry.get("transport_size"), "transport_size")
    policy = geometry.get("preprocessing", {})
    if policy.get("policy") != "rgb_smart_resize_v1" or policy.get("format") != "PNG":
        raise CoordinateError("unsupported_image_preprocessing")
    for name in ("min_pixels", "max_pixels", "resize_factor"):
        value = policy.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CoordinateError("invalid_image_preprocessing_bounds")
    if policy["resize_factor"] != 32:
        raise CoordinateError("unsupported_image_resize_factor")
    try:
        expected = smart_resize_size(*source, policy["min_pixels"], policy["max_pixels"], policy["resize_factor"])
    except ValueError as exc:
        raise CoordinateError("invalid_image_preprocessing_bounds") from exc
    if expected != transport:
        raise CoordinateError("transport_size_mismatch")
    return deepcopy(geometry)


def coordinate_matches(text):
    """Yield point-like pairs, rejecting malformed numeric coordinate tokens."""
    for parenthetical in PAREN_RE.finditer(text or ""):
        tokens = [part.strip() for part in parenthetical.group(1).split(",")]
        if len(tokens) > 2 and any(NUMERIC_START_RE.match(value) for value in tokens):
            raise CoordinateError("invalid_coordinate_pair")
    for match in PAIR_RE.finditer(text or ""):
        tokens = [value.strip() for value in match.groups()]
        if not any(NUMERIC_START_RE.match(value) for value in tokens):
            continue  # Ordinary parenthetical prose, e.g. (sky, cloud).
        if not all(NUMBER_RE.fullmatch(value) for value in tokens):
            raise CoordinateError("invalid_coordinate_number")
        point = tuple(float(value) for value in tokens)
        if not all(math.isfinite(value) for value in point):
            raise CoordinateError("nonfinite_coordinate")
        yield match, point


def parse_coordinate_points(text):
    return [point for _, point in coordinate_matches(text)]


def convert_coordinate_text(text, geometry=None, original_size=None, legacy_coordinate_frame=None):
    """Return text with validated original-pixel points and a conversion record.

    ``legacy_coordinate_frame`` must explicitly be ``original_pixels`` to admit
    coordinate-bearing old traces that lack geometry. Normalized legacy points
    cannot be inferred from their range. Every numeric pair is checked even if
    callers permit reasoning without coordinates.
    """
    matches = list(coordinate_matches(text))
    if geometry is not None:
        geometry = validate_image_geometry(geometry, original_size)
        width, height = geometry["original_size"]
        frame = "normalized_1000"
    elif matches:
        if legacy_coordinate_frame != "original_pixels":
            raise CoordinateError("missing_coordinate_frame")
        width, height = _size(original_size, "original_size")
        frame = "original_pixels"
    else:
        return text, []
    converted = []
    pieces = []
    cursor = 0
    for match, (x, y) in matches:
        if frame == "normalized_1000":
            if not 0 <= x <= 1000 or not 0 <= y <= 1000:
                raise CoordinateError("coordinate_out_of_bounds")
            px, py = round(x * (width - 1) / 1000), round(y * (height - 1) / 1000)
        else:
            if not 0 <= x <= width - 1 or not 0 <= y <= height - 1:
                raise CoordinateError("coordinate_out_of_bounds")
            px, py = x, y
        converted.append({"source": [x, y], "original_pixels": [px, py]})
        pieces.extend([text[cursor:match.start()], f"({px:g}, {py:g})"])
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), converted


class ImagePayloadCache:
    """Bounded process-local RGB/PNG cache; entries do not retain large PIL images."""
    def __init__(self, min_pixels=65536, max_pixels=1048576, max_entries=16, max_bytes=64 * 1024 * 1024):
        self.min_pixels, self.max_pixels = int(min_pixels), int(max_pixels)
        self.max_entries, self.max_bytes = int(max_entries), int(max_bytes)
        if min(self.max_entries, self.max_bytes) <= 0:
            raise ValueError("Image cache bounds must be positive")
        self._entries = OrderedDict()
        self._bytes = 0
        self.hits = self.misses = 0

    def _key(self, image):
        filename = getattr(image, "filename", None)
        if filename:
            path = Path(filename).resolve()
            stat = path.stat()
            return ("file", str(path), stat.st_size, stat.st_mtime_ns, image.size, image.mode), None
        return ("object", id(image), image.size, image.mode), weakref.ref(image)

    def encode(self, image):
        key, reference = self._key(image)
        entry = self._entries.get(key)
        if entry is not None and (entry[0] is None or entry[0]() is image):
            self.hits += 1
            self._entries.move_to_end(key)
            return entry[1], deepcopy(entry[2])
        if entry is not None:
            self._bytes -= len(entry[1])
            del self._entries[key]
        self.misses += 1
        target = smart_resize_size(*image.size, self.min_pixels, self.max_pixels)
        # Resize before RGB conversion to avoid materializing a second 54MP RGB
        # copy. JPEG draft decoding is intentionally not used: its scale differs
        # across encoders and would make the geometry policy non-reproducible.
        transport = image.resize(target, Image.Resampling.BICUBIC) if image.size != target else image
        if transport.mode in {"RGBA", "LA"} or (transport.mode == "P" and "transparency" in transport.info):
            rgba = transport.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            transport = background.convert("RGB")
        else:
            transport = transport.convert("RGB")
        buffer = BytesIO()
        transport.save(buffer, format="PNG", compress_level=3)
        payload = buffer.getvalue()
        uri = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
        geometry = {"version": 1, "coordinate_frame": "normalized_1000", "coordinate_extent": [1000, 1000],
                    "original_size": list(image.size), "transport_size": list(target),
                    "preprocessing": {"policy": "rgb_smart_resize_v1", "min_pixels": self.min_pixels,
                                      "max_pixels": self.max_pixels, "resize_factor": 32, "format": "PNG",
                                      "resampling": "bicubic", "alpha_background": "white"},
                    "transport_sha256": hashlib.sha256(payload).hexdigest(), "transport_bytes": len(payload)}
        validate_image_geometry(geometry, image.size)
        if len(uri) <= self.max_bytes:
            self._entries[key] = (reference, uri, geometry)
            self._bytes += len(uri)
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                _, removed = self._entries.popitem(last=False)
                self._bytes -= len(removed[1])
        return uri, deepcopy(geometry)

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._entries), "bytes": self._bytes}
