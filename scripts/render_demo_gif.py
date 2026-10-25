"""Render ``assets/demo.svg`` into ``assets/demo.gif`` (the self-playing README demo).

The SVG is authored with SMIL animations. GitHub serves README SVGs inside ``<img>``, which
blocks scripts but keeps SMIL in most browsers — still, a GIF is the only format that animates
everywhere (VS Code preview, PyPI, npm, terminals). This script samples the same timeline and
re-encodes it as a looping GIF, so both formats stay in sync.

Requires the optional rendering dependencies::

    pip install -r requirements-dev.txt   # pillow + resvg-py

Usage::

    python scripts/render_demo_gif.py                 # → assets/demo.gif
    python scripts/render_demo_gif.py --duration 8 --fps 9 --width 940
"""

from __future__ import annotations

import argparse
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

BEGIN_RE = re.compile(r"([\d.]+)s")


def _begin_time(element: ET.Element) -> float | None:
    """Return the ``begin`` time (seconds) of an element's opacity animation, if any."""
    for animate in element.iter(f"{{{SVG_NS}}}animate"):
        if animate.get("attributeName") == "opacity":
            match = BEGIN_RE.search(animate.get("begin", "") or "")
            if match:
                return float(match.group(1))
    return None


def _drop(parent: ET.Element, node: ET.Element) -> None:
    """Remove ``node`` but keep its tail text (ElementTree drops it otherwise)."""
    index = list(parent).index(node)
    if node.tail:
        if index > 0:
            previous = parent[index - 1]
            previous.tail = (previous.tail or "") + node.tail
        else:
            parent.text = (parent.text or "") + node.tail
    parent.remove(node)


def _strip_animations(root: ET.Element) -> None:
    """Remove SMIL nodes so the static frame renders exactly as sampled."""
    parents = {child: parent for parent in root.iter() for child in parent}
    for node in list(root.iter()):
        if node.tag in {f"{{{SVG_NS}}}animate", f"{{{SVG_NS}}}animateMotion", f"{{{SVG_NS}}}mpath"}:
            parent = parents.get(node)
            if parent is not None:
                _drop(parent, node)


def font_files() -> list[str]:
    """Fonts used for rasterisation: DejaVu (matplotlib) for Latin + bundled Bengali face."""
    files: list[str] = []
    try:
        import matplotlib

        files.extend(
            str(path) for path in (Path(matplotlib.get_data_path()) / "fonts" / "ttf").glob("DejaVu*.ttf")
        )
    except ImportError:
        pass
    files.extend(str(path) for path in sorted(FONT_DIR.glob("*.otf")))
    files.extend(str(path) for path in sorted(FONT_DIR.glob("*.ttf")))
    return files


def render_frames(
    svg_text: str, duration: float, fps: int, width: int, background: str
) -> list[bytes]:
    """Sample the animation timeline and return one PNG per frame."""
    import resvg_py

    fonts = font_files()

    frames: list[bytes] = []
    steps = max(2, int(duration * fps))
    for index in range(steps):
        moment = index / fps
        root = ET.fromstring(svg_text)
        for element in root.iter():
            if element.get("opacity") != "0":
                continue
            begin = _begin_time(element)
            if begin is None:
                continue
            element.set("opacity", "1" if moment >= begin else "0")
        # blinking caret
        for element in root.iter():
            classes = (element.get("class") or "").split()
            if "caret" in classes:
                element.set("opacity", "1" if (moment % 1.1) < 0.55 else "0")
        _strip_animations(root)
        frame_svg = ET.tostring(root, encoding="unicode")
        png = resvg_py.svg_to_bytes(
            svg_string=frame_svg,
            background=background,
            width=width,
            font_files=fonts,
            sans_serif_family="DejaVu Sans",
            monospace_family="DejaVu Sans Mono",
        )
        frames.append(bytes(png))
    return frames


def main() -> int:
    """Render and write the GIF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "assets" / "demo.svg"))
    parser.add_argument("--output", default=str(ROOT / "assets" / "demo.gif"))
    parser.add_argument("--duration", type=float, default=8.8, help="seconds to sample")
    parser.add_argument("--fps", type=int, default=9)
    parser.add_argument("--width", type=int, default=940)
    parser.add_argument("--colors", type=int, default=96)
    parser.add_argument("--background", default="#070c17")
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - optional dependency
        print("! pillow is required: pip install -r requirements-dev.txt")
        return 1

    source = Path(args.source)
    svg_text = source.read_text(encoding="utf-8")
    frames = render_frames(svg_text, args.duration, args.fps, args.width, args.background)

    images = [Image.open(io.BytesIO(frame)).convert("RGB") for frame in frames]
    paletted = [
        image.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        for image in images
    ]
    output = Path(args.output)
    paletted[0].save(
        output,
        save_all=True,
        append_images=paletted[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"✓ {output.relative_to(ROOT)} · {len(paletted)} frames · {output.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
