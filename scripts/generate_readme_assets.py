"""Generate the README artwork in ``assets/``.

Everything here is deterministic and offline: hand-authored SVG for the banner, the
pipeline diagram and the animated demo, plus matplotlib charts built from the latest
RAGAS report in ``data/evaluation/``.

Usage::

    python scripts/generate_readme_assets.py          # SVGs + charts
    python scripts/generate_readme_assets.py --no-charts
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

FONT_DIR = ASSETS / "fonts"

SANS = "Inter, 'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', ui-monospace, 'DejaVu Sans Mono', Consolas, monospace"
BENGALI = "'Noto Sans Bengali', 'Hind Siliguri', 'Mukti', 'Nirmala UI', 'Vrinda', 'Sölvesborg', sans-serif"

BASE_DEFS = """
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#070b16"/>
    <stop offset="0.55" stop-color="#0b1223"/>
    <stop offset="1" stop-color="#080d1a"/>
  </linearGradient>
  <radialGradient id="glowTeal" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="#14b8a6" stop-opacity="0.55"/>
    <stop offset="0.55" stop-color="#0d9488" stop-opacity="0.14"/>
    <stop offset="1" stop-color="#0d9488" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="glowViolet" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="#8b5cf6" stop-opacity="0.5"/>
    <stop offset="0.55" stop-color="#7c3aed" stop-opacity="0.12"/>
    <stop offset="1" stop-color="#7c3aed" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="glowBlue" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="#3b82f6" stop-opacity="0.45"/>
    <stop offset="1" stop-color="#2563eb" stop-opacity="0"/>
  </radialGradient>
  <pattern id="grid" width="44" height="44" patternUnits="userSpaceOnUse">
    <path d="M44 0H0V44" fill="none" stroke="#1b2740" stroke-width="1"/>
  </pattern>
  <linearGradient id="titleFill" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#f1f5f9"/>
    <stop offset="0.42" stop-color="#5eead4"/>
    <stop offset="0.74" stop-color="#a78bfa"/>
    <stop offset="1" stop-color="#60a5fa"/>
    <animate attributeName="x1" values="0;0.25;0" dur="16s" repeatCount="indefinite"/>
    <animate attributeName="x2" values="1;1.25;1" dur="16s" repeatCount="indefinite"/>
  </linearGradient>
"""

BASE_STYLE = f"""
  <style>
    text {{ font-family: {SANS}; }}
    .mono {{ font-family: {MONO}; }}
    .bn {{ font-family: {BENGALI}; }}
    @keyframes dash {{ to {{ stroke-dashoffset: -96; }} }}
    @keyframes breathe {{ 0%,100% {{ opacity: .5; }} 50% {{ opacity: 1; }} }}
    .flow {{ stroke-dasharray: 9 9; animation: dash 2.6s linear infinite; }}
    .pulse {{ animation: breathe 6s ease-in-out infinite; }}
    .card {{ fill: #0d1526; stroke: #22314b; stroke-width: 1.2; }}
    .label {{ fill: #e8eefb; font-size: 15px; font-weight: 600; }}
    .sub {{ fill: #7d8cad; font-size: 11.5px; }}
    .stage {{ fill: #5eead4; font-size: 10.5px; font-weight: 700; letter-spacing: 1.4px; }}
  </style>
"""


def esc(text: str) -> str:
    """Escape the XML-significant characters used in the hand-written SVG."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #
def banner() -> str:
    """Animated hero banner: gradient title, stack pills, three-index → answer motif."""
    pills = [
        ("LlamaIndex", "#5eead4"),
        ("bge-m3", "#60a5fa"),
        ("FAISS", "#a78bfa"),
        ("bge-reranker-v2-m3", "#f59e0b"),
        ("RAGAS", "#f472b6"),
        ("Bangla + English", "#34d399"),
    ]
    x = 52.0
    pill_svg = []
    for text, colour in pills:
        w = 12 + len(text) * 7.0 + 12
        pill_svg.append(
            f'<rect x="{x:.1f}" y="204" width="{w:.1f}" height="32" rx="16" fill="#111b2f" '
            f'stroke="{colour}" stroke-opacity="0.35"/>'
            f'<circle cx="{x + 15:.1f}" cy="220" r="3.1" fill="{colour}"/>'
            f'<text x="{x + 25:.1f}" y="224.5" font-size="12" fill="#c9d6ee">{esc(text)}</text>'
        )
        x += w + 10

    chips = [("vector", "#2dd4bf", 92), ("sentence-window", "#a78bfa", 152), ("graph", "#f59e0b", 212)]
    motif = []
    for text, colour, cy in chips:
        motif.append(
            f'<rect x="806" y="{cy - 14}" width="150" height="28" rx="14" fill="#0f1a2d" '
            f'stroke="{colour}" stroke-opacity="0.45"/>'
            f'<circle cx="823" cy="{cy}" r="3.4" fill="{colour}"/>'
            f'<text x="834" y="{cy + 4}" font-size="12" fill="#cbd8f0">{esc(text)}</text>'
            f'<path id="b-{text}" d="M960 {cy} C 1000 {cy}, 1002 152, 1030 152" fill="none" '
            f'stroke="{colour}" stroke-opacity="0.30" stroke-width="1.4"/>'
            f'<circle r="3" fill="{colour}">'
            f'<animateMotion dur="{3.0 + cy / 500:.1f}s" repeatCount="indefinite" begin="{cy / 1100:.2f}s">'
            f'<mpath href="#b-{text}"/></animateMotion></circle>'
        )
    motif.append(
        '<rect x="1030" y="112" width="196" height="80" rx="15" fill="#0d1a2e" stroke="#2dd4bf" stroke-opacity="0.5"/>'
        '<rect x="1030" y="112" width="196" height="80" rx="15" fill="url(#glowBlue)" opacity="0.16"/>'
        '<text x="1048" y="138" font-size="11" class="stage" fill="#5eead4">ANSWER</text>'
        '<text x="1048" y="160" font-size="13.5" fill="#e8eefb" font-weight="600">grounded + cited</text>'
        '<text x="1048" y="178" font-size="10.5" class="mono" fill="#7dd3fc">[handbook.pdf, p.2]</text>'
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 300" width="1280" height="300"
     role="img" aria-label="Advanced Multi-Source RAG — PDF, CSV and URLs to three indexes to one grounded, cited answer">
  <defs>{BASE_DEFS}</defs>
  {BASE_STYLE}
  <rect width="1280" height="300" fill="url(#bg)"/>
  <rect width="1280" height="300" fill="url(#grid)" opacity="0.32"/>
  <circle cx="130" cy="20" r="290" fill="url(#glowTeal)" class="pulse"/>
  <circle cx="1210" cy="300" r="300" fill="url(#glowViolet)" opacity="0.85"/>
  <circle cx="820" cy="10" r="190" fill="url(#glowBlue)" opacity="0.45"/>
  <rect x="0.6" y="0.6" width="1278.8" height="298.8" rx="18" fill="none" stroke="#1e2c46"/>

  <text x="52" y="66" font-size="11.5" letter-spacing="3.6" fill="#2dd4bf" font-weight="700">ENTERPRISE KNOWLEDGE BASE</text>
  <text x="52" y="132" font-size="41" font-weight="800" fill="url(#titleFill)" letter-spacing="-0.8">Advanced Multi-Source RAG</text>
  <text x="52" y="168" font-size="17" fill="#93a4c4">PDF · CSV · URLs → three indexes → one grounded, cited answer</text>
  {''.join(pill_svg)}
  {''.join(motif)}
</svg>
"""


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def pipeline() -> str:
    """Annotated 1→10 pipeline diagram with animated packet flow."""
    width, height = 1240, 726
    arrows: list[str] = []
    dots: list[str] = []

    def card(x: float, y: float, w: float, h: float, accent: str, stage: str, title: str, sub: str) -> str:
        return (
            f'<g><rect class="card" x="{x}" y="{y}" width="{w}" height="{h}" rx="14"/>'
            f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{accent}" opacity="0.9"/>'
            f'<text class="stage" x="{x + 18}" y="{y + 22}" fill="{accent}">{stage}</text>'
            f'<text class="label" x="{x + 18}" y="{y + 44}">{esc(title)}</text>'
            f'<text class="sub" x="{x + 18}" y="{y + 63}">{esc(sub)}</text></g>'
        )

    def arrow(path_id: str, d: str, colour: str = "#38bdf8", dur: float = 3.0, begin: float = 0.0) -> None:
        arrows.append(
            f'<path id="{path_id}" d="{d}" fill="none" stroke="{colour}" stroke-opacity="0.45" '
            f'stroke-width="1.5" class="flow"/>'
        )
        dots.append(
            f'<circle r="3.3" fill="{colour}"><animateMotion dur="{dur}s" begin="{begin}s" '
            f'repeatCount="indefinite"><mpath href="#{path_id}"/></animateMotion></circle>'
            f'<circle r="3.3" fill="{colour}" opacity="0.3"><animateMotion dur="{dur}s" '
            f'begin="{begin + dur / 2:.1f}s" repeatCount="indefinite"><mpath href="#{path_id}"/>'
            f"</animateMotion></circle>"
        )

    row1 = [
        (48, 258, "#5eead4", "01 · UI", "Streamlit chat + uploads", "PDF · CSV · TXT/MD · URLs · questions"),
        (336, 258, "#22d3ee", "02 · INGESTION", "Loaders", "PyMuPDF · pandas · trafilatura"),
        (624, 258, "#38bdf8", "03 · DOCUMENT DB", "Unified SQLite store", "text · source · type · page · time"),
        (912, 258, "#818cf8", "04 · PROCESSING", "Clean · chunk · tag", "600 / 80 tokens · language · keywords"),
    ]
    row1_svg = "".join(card(x, 56, w, 104, accent, stage, title, sub) for x, w, accent, stage, title, sub in row1)
    row1_svg += "".join(
        f'<text x="{(row1[i][0] + row1[i][1] + row1[i + 1][0]) / 2:.0f}" y="113" font-size="17" '
        f'fill="#3f5170" text-anchor="middle">›</text>'
        for i in range(len(row1) - 1)
    )

    indexes = [
        (48, 360, "#2dd4bf", "06a · VECTOR INDEX", "FAISS · bge-m3", "cosine · IndexFlatIP · persisted"),
        (440, 360, "#a78bfa", "06b · SENTENCE WINDOW", "window = 3", "sentence hits + neighbour context"),
        (832, 360, "#f59e0b", "06c · GRAPH INDEX", "entities + relations", "LLM triples → NetworkX → graph.json"),
    ]
    index_svg = "".join(card(x, 226, w, 104, accent, stage, title, sub) for x, w, accent, stage, title, sub in indexes)
    for i, (x, w, *_rest) in enumerate(indexes):
        cx = x + w / 2
        arrow(f"r2-{i}", f"M1041 164 C 1041 208, {cx:.0f} 184, {cx:.0f} 220",
              colour="#8b9cc4", dur=3.2, begin=i * 0.5)

    retriever = (
        '<g><rect class="card" x="48" y="390" width="1144" height="86" rx="14"/>'
        '<rect x="48" y="390" width="4" height="86" rx="2" fill="#fbbf24"/>'
        '<text class="stage" x="70" y="416" fill="#fbbf24">07 · MULTI-RETRIEVER</text>'
        '<text class="label" x="70" y="442">Three retrievers in parallel — ThreadPoolExecutor · per-retriever timeout · failure isolation</text>'
        '<text class="sub" x="70" y="462">vector ∥ sentence-window ∥ graph → Reciprocal Rank Fusion → dedupe by node id + normalised text → provenance merge</text>'
        '<text class="sub" x="1170" y="416" fill="#8b9cc4" text-anchor="end">RRF k = 60</text></g>'
    )
    for i, (x, w, *_rest) in enumerate(indexes):
        arrow(f"r3-{i}", f"M{x + w / 2:.0f} 330 V384", colour="#fbbf24", dur=2.4, begin=i * 0.35)

    row3 = [
        (48, 360, "#fbbf24", "08 · RE-RANK", "Cross-encoder → top 5", "bge-reranker-v2-m3 · relevance gate"),
        (440, 360, "#f472b6", "09 · QUERY CONSTRUCTION", "Prompt + cited context", "numbered [i] [file, p.N] blocks"),
        (832, 360, "#fb7185", "10 · GENERATION", "Claude · OpenAI · Ollama", "strictly grounded · retries · streaming"),
    ]
    row3_svg = "".join(card(x, 512, w, 104, accent, stage, title, sub) for x, w, accent, stage, title, sub in row3)
    arrow("r4-0", "M620 482 V506", colour="#fbbf24", dur=2.4)
    row3_svg += "".join(
        f'<text x="{(row3[i][0] + row3[i][1] + row3[i + 1][0]) / 2:.0f}" y="569" font-size="17" '
        f'fill="#3f5170" text-anchor="middle">›</text>'
        for i in range(len(row3) - 1)
    )

    response = (
        '<g><rect x="48" y="656" width="1144" height="62" rx="14" fill="#0b1a24" stroke="#2dd4bf" stroke-opacity="0.45"/>'
        '<text class="stage" x="70" y="680" fill="#5eead4">RESPONSE</text>'
        '<text class="label" x="70" y="704">Answer · bullet citations [file_name, page] · expandable sources with retriever + score</text>'
        '<text class="sub" x="1170" y="704" fill="#fbbf24" text-anchor="end">no context → honest refusal</text></g>'
    )
    for i, x in enumerate((620, 1000)):
        arrow(f"r5-{i}", f"M{x} 616 V650", colour="#5eead4", dur=2.4, begin=i * 0.4)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}"
     role="img" aria-label="Pipeline diagram: sources, ingestion, unified document DB, processing, three indexes, parallel retrieval, fusion, reranking, prompt construction, generation, cited response">
  <defs>{BASE_DEFS}</defs>
  {BASE_STYLE}
  <rect width="{width}" height="{height}" rx="18" fill="#070c17"/>
  <rect width="{width}" height="{height}" rx="18" fill="url(#grid)" opacity="0.26"/>
  <circle cx="90" cy="60" r="260" fill="url(#glowTeal)" opacity="0.28"/>
  <circle cx="1180" cy="700" r="280" fill="url(#glowViolet)" opacity="0.26"/>
  <rect x="0.6" y="0.6" width="{width - 1.2}" height="{height - 1.2}" rx="18" fill="none" stroke="#1e2c46"/>

  <text x="48" y="32" font-size="11.5" letter-spacing="3.2" fill="#5eead4" font-weight="700">INGEST → INDEX → RETRIEVE → RERANK → GENERATE</text>
  <text x="1192" y="32" font-size="11.5" fill="#7d8cad" text-anchor="end">advanced-multi-source-rag</text>

  {row1_svg}
  {index_svg}
  {retriever}
  {row3_svg}
  {response}
  {''.join(arrows)}
  {''.join(dots)}
</svg>
"""


# --------------------------------------------------------------------------- #
# Animated demo
# --------------------------------------------------------------------------- #
DEMO_LINES: list[tuple[str, str]] = [
    ("cmd", "python scripts/build_index.py --reset"),
    ("ok", "✓ 5 documents · 13 chunks · 125 sentence-window nodes"),
    ("ok", "✓ vector 13 · graph 140 entities / 145 relations · 2.4 s"),
    ("cmd", 'python scripts/ask.py "How many days of annual leave?"'),
    ("out", "Every permanent employee is entitled to 20 days of annual"),
    ("out", "leave per calendar year. [employee_handbook.pdf, p.2]"),
    ("dim", "**Citations**"),
    ("cite", "• employee_handbook.pdf, p.2"),
    ("dim", "**Sources**  ▸ 5 passages · vector + window + graph"),
    ("cmd", 'python scripts/ask.py "What is the capital of France?"'),
    ("warn", "I could not find the answer in the provided documents."),
    ("dim", "relevance gate blocked · no citation invented"),
]

COLOURS = {
    "cmd": "#5eead4",
    "ok": "#86efac",
    "out": "#dbeafe",
    "cite": "#7dd3fc",
    "dim": "#64748b",
    "warn": "#fbbf24",
}
STEP = {"cmd": 26, "ok": 24, "out": 24, "cite": 24, "dim": 22, "warn": 24}


def bengali_face() -> str:
    """Return a ``@font-face`` rule with the bundled Bengali font embedded as a data URI.

    GitHub serves README images through a proxy and will not fetch relative font files, so the
    face has to travel inside the SVG itself for Bangla text to render identically everywhere.
    """
    font = FONT_DIR / "HindSiliguri-Regular.otf"
    if not font.exists():
        return ""
    encoded = base64.b64encode(font.read_bytes()).decode("ascii")
    return (
        "@font-face { font-family: 'Repo Bengali'; font-weight: 400; font-style: normal; "
        f"src: url(data:font/otf;base64,{encoded}) format('opentype'); }}"
    )


def demo() -> str:
    """Animated terminal session next to the chat card (self-playing, loops)."""
    delay = 0.62
    rows: list[str] = []
    y = 148.0
    for i, (kind, text) in enumerate(DEMO_LINES):
        begin = i * delay + 0.5
        prefix = '<tspan fill="#64748b">$ </tspan>' if kind == "cmd" else ""
        cls = ' class="mono"' if kind != "warn" else ' class="mono"'
        rows.append(
            f'<text x="60" y="{y}" font-size="{13 if kind != "dim" else 12.5}"{cls} fill="{COLOURS[kind]}" opacity="0">'
            f'<animate attributeName="opacity" values="0;1" dur="0.01s" begin="{begin:.2f}s" fill="freeze"/>'
            f"{prefix}{esc(text)}</text>"
        )
        y += STEP[kind]
    total = len(DEMO_LINES) * delay + 1.4

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 540" width="1120" height="540" role="img"
     aria-label="Animated demo: build indexes from the CLI, ask a grounded question and get a cited answer, then an out-of-domain question that is refused; the Streamlit chat answers the same question in Bangla">
  <defs>{BASE_DEFS}
    <linearGradient id="termBg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#0b1220"/><stop offset="1" stop-color="#080e1b"/>
    </linearGradient>
  </defs>
  {BASE_STYLE}
  <style>
    {bengali_face()}
    .bn {{ font-family: 'Repo Bengali', 'Hind Siliguri', 'Noto Sans Bengali', 'Nirmala UI', 'Bangla MN', sans-serif; }}
    @keyframes blink {{ 0%,49% {{ opacity: 1; }} 50%,100% {{ opacity: 0; }} }}
    .caret {{ animation: blink 1.1s step-end infinite; }}
  </style>
  <rect width="1120" height="540" rx="18" fill="#070c17"/>
  <rect width="1120" height="540" rx="18" fill="url(#grid)" opacity="0.22"/>
  <circle cx="120" cy="60" r="240" fill="url(#glowTeal)" opacity="0.22"/>
  <circle cx="1020" cy="490" r="240" fill="url(#glowViolet)" opacity="0.20"/>
  <rect x="0.6" y="0.6" width="1118.8" height="538.8" rx="18" fill="none" stroke="#1e2c46"/>

  <g>
    <rect x="32" y="32" width="700" height="476" rx="14" fill="url(#termBg)" stroke="#22314b"/>
    <rect x="32" y="32" width="700" height="38" rx="14" fill="#0f1a2d"/>
    <rect x="32" y="56" width="700" height="14" fill="#0f1a2d"/>
    <circle cx="56" cy="51" r="5" fill="#f87171"/><circle cx="74" cy="51" r="5" fill="#fbbf24"/>
    <circle cx="92" cy="51" r="5" fill="#34d399"/>
    <text x="382" y="56" font-size="11.5" class="mono" fill="#7d8cad" text-anchor="middle">advanced-multi-source-rag — bash</text>
    {''.join(rows)}
    <rect x="60" y="{y - 14:.0f}" width="8" height="15" fill="#5eead4" class="caret"/>
  </g>

  <g transform="translate(756, 32)">
    <rect width="332" height="476" rx="14" fill="#0d1526" stroke="#22314b"/>
    <rect width="332" height="40" rx="14" fill="#111c30"/>
    <rect y="26" width="332" height="14" fill="#111c30"/>
    <circle cx="22" cy="20" r="5" fill="#5eead4" class="pulse"/>
    <text x="38" y="24" font-size="12" fill="#cbd8f0" font-weight="600">RAG Studio</text>
    <text x="310" y="24" font-size="10.5" fill="#64748b" text-anchor="end">local · cited</text>

    <g opacity="0">
      <animate attributeName="opacity" values="0;1" dur="0.4s" begin="{total * 0.30:.1f}s" fill="freeze"/>
      <rect x="86" y="58" width="226" height="54" rx="12" fill="#164e63" stroke="#22d3ee" stroke-opacity="0.35"/>
      <text x="102" y="80" font-size="12.5" class="bn" fill="#e0f2fe" lang="bn">বার্ষিক ছুটি কত দিন?</text>
      <text x="102" y="100" font-size="10" fill="#7dd3fc">Bangla query · same index</text>
    </g>

    <g opacity="0">
      <animate attributeName="opacity" values="0;1" dur="0.6s" begin="{total * 0.50:.1f}s" fill="freeze"/>
      <rect x="16" y="126" width="300" height="126" rx="12" fill="#101c30" stroke="#2dd4bf" stroke-opacity="0.35"/>
      <text x="32" y="150" font-size="12.5" class="bn" fill="#e8eefb" lang="bn">প্রতি কর্মীর জন্য বছরে ২০ দিন</text>
      <text x="32" y="170" font-size="12.5" class="bn" fill="#e8eefb" lang="bn">বার্ষিক ছুটি।</text>
      <text x="32" y="196" font-size="10.5" class="mono" fill="#7dd3fc">[employee_handbook.pdf, p.2]</text>
      <rect x="32" y="208" width="128" height="26" rx="13" fill="#0b2233" stroke="#2dd4bf" stroke-opacity="0.4"/>
      <text x="46" y="225" font-size="10.5" fill="#5eead4">▸ 5 sources · 0.9 s</text>
    </g>

    <g opacity="0">
      <animate attributeName="opacity" values="0;1" dur="0.5s" begin="{total * 0.70:.1f}s" fill="freeze"/>
      <rect x="16" y="266" width="300" height="62" rx="12" fill="#101c30" stroke="#f59e0b" stroke-opacity="0.3"/>
      <text x="32" y="288" font-size="10.5" fill="#fbbf24">out-of-domain question</text>
      <text x="32" y="308" font-size="11" fill="#e8eefb">"I could not find the answer…"</text>
    </g>

    <g opacity="0">
      <animate attributeName="opacity" values="0;1" dur="0.5s" begin="{total * 0.86:.1f}s" fill="freeze"/>
      <text x="16" y="366" font-size="10.5" class="stage" fill="#5eead4">ANSWER CONTRACT</text>
      <text x="16" y="386" font-size="10.5" fill="#93a4c4">context only · cite or refuse</text>
      <text x="16" y="406" font-size="10.5" class="mono" fill="#7d8cad">[file_name, page]</text>
      <rect x="16" y="422" width="300" height="1" fill="#1e2c46"/>
      <text x="16" y="444" font-size="10" fill="#64748b">streamed · 3 retrievers · reranked</text>
      <text x="16" y="462" font-size="10" fill="#64748b">offline-capable · swaps to bge-m3</text>
    </g>
  </g>
</svg>
"""


def social_card() -> str:
    """1280×640 social/OG card derived from the banner (for GitHub repo settings)."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640">
  <defs>{BASE_DEFS}</defs>
  {BASE_STYLE}
  <rect width="1280" height="640" fill="url(#bg)"/>
  <rect width="1280" height="640" fill="url(#grid)" opacity="0.25"/>
  <circle cx="150" cy="80" r="360" fill="url(#glowTeal)" opacity="0.7"/>
  <circle cx="1150" cy="600" r="380" fill="url(#glowViolet)" opacity="0.7"/>
  <rect x="0.6" y="0.6" width="1278.8" height="638.8" rx="0" fill="none" stroke="#1e2c46"/>
  <text x="96" y="196" font-size="13" letter-spacing="5" fill="#2dd4bf" font-weight="700">ENTERPRISE KNOWLEDGE BASE</text>
  <text x="96" y="286" font-size="62" font-weight="800" fill="url(#titleFill)">Advanced Multi-Source RAG</text>
  <text x="96" y="348" font-size="24" fill="#93a4c4">PDF · CSV · URLs → vector + sentence-window + graph → grounded, cited answers</text>
  <text x="96" y="452" font-size="17" class="mono" fill="#5eead4">vector ∥ window ∥ graph → RRF → bge-reranker-v2-m3 → top 5</text>
  <text x="96" y="486" font-size="17" class="mono" fill="#a5b4fc">LlamaIndex · bge-m3 · FAISS · NetworkX · Claude/OpenAI/Ollama · RAGAS</text>
  <text x="96" y="552" font-size="16" fill="#7d8cad">Bangla + English · 125 tests · Docker · offline fallbacks</text>
</svg>
"""


# --------------------------------------------------------------------------- #
# Charts (matplotlib)
# --------------------------------------------------------------------------- #
def charts() -> None:
    """Render the evaluation charts from the latest RAGAS report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    reports = sorted((ROOT / "data" / "evaluation").glob("ragas_report_2026*.json"))
    if not reports:
        print("· no evaluation report found — skipping charts")
        return

    # Prefer the report with the most questions so the chart shows the full test set.
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    report: dict[str, Any] = max(loaded, key=lambda item: item.get("config", {}).get("questions", 0))
    for candidate in ("DejaVu Sans", "Liberation Sans", "Arial"):
        if any(candidate in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = candidate
            break

    bg, panel, fg, muted = "#0b1220", "#0f1a2d", "#e8eefb", "#7d8cad"
    teal, amber = "#2dd4bf", "#fbbf24"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 4.3), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.patch.set_facecolor(bg)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels = ["Faithfulness", "Answer\nrelevancy", "Context\nprecision", "Context\nrecall"]
    advanced = [report["metrics"][m] for m in metrics]
    baseline = [(report.get("baseline_metrics") or {}).get(m, 0) for m in metrics]

    for ax in (ax1, ax2):
        ax.set_facecolor(panel)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#22314b")
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(axis="y", color="#1b2740", linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)

    positions = range(len(metrics))
    width = 0.36
    bars_a = ax1.bar([p - width / 2 for p in positions], advanced, width, label="Advanced (3 retrievers + rerank)",
                     color=teal, edgecolor="none")
    bars_b = ax1.bar([p + width / 2 for p in positions], baseline, width, label="Vector-only baseline",
                     color="#3b4a6b", edgecolor="none")
    ax1.bar_label(bars_a, fmt="%.2f", color=fg, fontsize=8, padding=2)
    ax1.bar_label(bars_b, fmt="%.2f", color="#94a3b8", fontsize=8, padding=2)
    ax1.set_xticks(list(positions))
    ax1.set_xticklabels(labels, color=muted, fontsize=9)
    ax1.set_ylim(0, 1.18)
    ax1.set_title("Answer quality · 10-question bilingual test set", color=fg, fontsize=11, pad=12, loc="left")
    legend = ax1.legend(frameon=False, fontsize=8, loc="upper left", ncols=2, bbox_to_anchor=(0, 1.0))
    for text in legend.get_texts():
        text.set_color(muted)

    latencies = [s.get("latency_ms") for s in report.get("samples", []) if s.get("latency_ms")]
    warm = sorted(value for value in latencies if value < 500)
    def short_label(question: str, index: int) -> str:
        """Short axis label; documents with no Latin glyphs are tagged instead of drawn."""
        if any(ord(char) > 0x2FF for char in question):
            return f"Bangla query (Q{index + 1})"
        return question[:38] + "…" if len(question) > 39 else question

    questions = [
        short_label(s["question"], index)
        for index, s in enumerate(report.get("samples", []))
        if s.get("latency_ms") and s["latency_ms"] < 500
    ]
    if warm:
        ax2.barh(range(len(warm)), warm, color=amber, height=0.62, edgecolor="none")
        for i, value in enumerate(warm):
            ax2.text(value + 0.35, i, f"{value:.1f} ms", va="center", color=fg, fontsize=8)
        ax2.set_yticks(range(len(warm)))
        ax2.set_yticklabels(questions, color=muted, fontsize=7.5)
        ax2.invert_yaxis()
        ax2.set_xlim(0, max(warm) * 1.35)
        ax2.grid(axis="x", color="#1b2740", linewidth=0.8)
        ax2.grid(axis="y", visible=False)
        ax2.set_title("Query latency · warm offline stack", color=fg, fontsize=11, pad=12, loc="left")
        ax2.set_xlabel(
            f"milliseconds · avg {report.get('average_latency_ms', 0):.0f} ms incl. cold start",
            color=muted, fontsize=8, labelpad=8,
        )

    fig.text(
        0.006, -0.055,
        "Offline stack: hashing embedder + lexical reranker + extractive answerer — this run proves the pipeline end to end, "
        "not model quality. Configure bge-m3 and a judge LLM for real RAGAS scores.",
        color="#64748b", fontsize=7.6, ha="left",
    )
    fig.tight_layout(pad=1.6)
    out = ASSETS / "eval-report.png"
    fig.savefig(out, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    """Write every asset to ``assets/``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-charts", action="store_true", help="skip the matplotlib charts")
    args = parser.parse_args()

    ASSETS.mkdir(exist_ok=True)
    written = {
        "banner.svg": banner(),
        "pipeline.svg": pipeline(),
        "demo.svg": demo(),
        "social-card.svg": social_card(),
    }
    for name, svg in written.items():
        path = ASSETS / name
        path.write_text(svg, encoding="utf-8")
        print(f"✓ {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.1f} KB)")

    if not args.no_charts:
        try:
            charts()
        except ImportError:
            print("· matplotlib not installed — skipping charts (pip install matplotlib)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
