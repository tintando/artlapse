"""Diagrams for docs/: a sketched stream screen with the setup page's zones drawn on it.

    uv run python scripts/figures.py          # writes docs/figures/*.svg, and the setup page's example

The setup page's "How it works" shows the target figure, from its own copy inside the package.

The PNGs next to them are the SVGs at twice the size, rendered by Chrome:

    google-chrome --headless=new --hide-scrollbars --force-device-scale-factor=2 --window-size=960,540 \
        --screenshot=docs/figures/target.png file://$PWD/docs/figures/target.svg
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
PAGE_EXAMPLE = ROOT / "src" / "artlapse" / "setup-example.svg"  # served by the setup page
W, H = 960, 540  # a 1920x1080 stream at half size

# the setup page's colours
WS, EX, TG = "#2ecc71", "#ff4d4d", "#ff3df2"
OK = "#ff9f1a"  # things that need no box
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

# the screen's parts, as (x, y, w, h)
MENU = (0, 0, W, 24)
TOOLS = (0, 24, 44, H - 24)
PANEL = (800, 24, 160, H - 24)
WORKSPACE = (44, 24, 756, H - 24)
PAPER = (150, 80, 460, 310)
AVATAR = (588, 250, 212, H - 250)  # the part over the workspace
CHAT = (56, 404, 280, 124)
GOAL = (812, 478, 136, 48)
DONATION = (56, 200, 264, 51)  # over the canvas's left edge
DONATION_ZONE = (47, 191, 282, 69)


def rect(r, **attrs) -> str:
    x, y, w, h = r
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" {attr(attrs)}/>'


def attr(attrs: dict) -> str:
    return " ".join(f'{k.rstrip("_").replace("_", "-")}="{v}"' for k, v in attrs.items())


def lines(x, y, widths, gap=16, h=7, fill="#d7d9dd", **attrs) -> str:
    """Blank text: one rounded bar per line."""
    return "".join(rect((x, y + i * gap, w, h), rx=h / 2, fill=fill, **attrs) for i, w in enumerate(widths))


def screen() -> str:
    """The stream as viewers see it: an art program, the canvas, and overlays."""
    s = [rect((0, 0, W, H), fill="#3d4046")]
    # the art program
    # panels in a lighter grey with an edge, so they stand out from the canvas area even when darkened
    s.append(rect(MENU, fill="#7a7e88"))
    s.append("".join(rect((12 + i * 48, 9, [34, 30, 38, 26][i], 6), rx=3, fill="#b4b8c0") for i in range(4)))
    s.append(rect(TOOLS, fill="#7a7e88", stroke="#9da1aa", stroke_width=2))
    s.append("".join(rect((10, 40 + i * 34, 24, 24), rx=5, fill="#9da1aa") for i in range(9)))
    s.append(rect(PANEL, fill="#7a7e88", stroke="#9da1aa", stroke_width=2))
    s.append("".join(rect((812 + (i % 6) * 22, 36 + (i // 6) * 22, 16, 16), rx=3, fill=c)
                     for i, c in enumerate(["#e8c27a", "#d97b5f", "#8fb3d9", "#6fae7f", "#b48fd1", "#f0f0f0",
                                            "#3a3a3a", "#c94f4f", "#4f7ec9", "#e0d36a", "#7a5a40", "#9a9ca3"])))
    s.append("".join(rect((812, 100 + i * 34, 136, 26), rx=4, fill="#858993") +
                     rect((818, 105 + i * 34, 16, 16), rx=2, fill="#f4f1ea") +
                     rect((842, 110 + i * 34, 60 - i * 6, 6), rx=3, fill="#c4c7ce") for i in range(5)))
    # the canvas, with a sketch on it
    s.append(rect(PAPER, fill="#f4f1ea"))
    x, y, w, h = PAPER
    s.append(f'<g fill="none" stroke="#5a5a5a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
             f'<path d="M{x + 20} {y + 230} C {x + 90} {y + 170}, {x + 150} {y + 175}, {x + 220} {y + 215} '
             f'S {x + 360} {y + 160}, {x + 440} {y + 205}"/>'
             f'<path d="M{x + 20} {y + 270} C {x + 140} {y + 245}, {x + 300} {y + 255}, {x + 440} {y + 262}"/>'
             f'<circle cx="{x + 360}" cy="{y + 80}" r="34"/>'
             f'<path d="M{x + 110} {y + 196} l0 -44 m-22 18 l22 -40 l22 40 m-36 -14 l14 -26 l14 26"/>'
             f'<path d="M{x + 70} {y + 70} q 14 -12 28 0 q 14 -12 28 0 M{x + 170} {y + 50} q 10 -9 20 0 q 10 -9 20 0"/>'
             f'</g>')
    # the streamer's avatar: a plain silhouette, over the canvas's corner and the panel
    s.append('<g fill="#b9bcc2">'
             '<circle cx="704" cy="330" r="62"/>'
             '<path d="M588 540 C 592 448, 640 408, 704 408 C 768 408, 818 448, 824 540 Z"/>'
             '</g>'
             '<g fill="#4a4d54"><circle cx="682" cy="320" r="7"/><circle cx="726" cy="320" r="7"/></g>'
             '<path d="M680 346 Q 704 368, 728 346" fill="none" stroke="#4a4d54" stroke-width="5" stroke-linecap="round"/>')
    # chat: names and messages as blank lines
    s.append(rect(CHAT, rx=8, fill="#000", fill_opacity=0.55))
    cx, cy, _, _ = CHAT
    for i, (name, msg) in enumerate([(52, 150), (40, 190), (64, 120), (46, 170), (58, 96)]):
        s.append(rect((cx + 12, cy + 14 + i * 22, name, 8), rx=4, fill="#9fc5ff"))
        s.append(rect((cx + 20 + name, cy + 14 + i * 22, msg, 8), rx=4, fill="#e6e6e6"))
    # a follower goal, over the panel
    s.append(rect(GOAL, rx=8, fill="#000", fill_opacity=0.6))
    gx, gy, gw, _ = GOAL
    s.append(lines(gx + 10, gy + 10, [70], fill="#e6e6e6"))
    s.append(rect((gx + 10, gy + 28, gw - 20, 10), rx=5, fill="#5c5f66") + rect((gx + 10, gy + 28, 70, 10), rx=5, fill="#f2c94c"))
    return "".join(s)


def donation() -> str:
    """A pop-up thanking a viewer for a tip: on screen for a few seconds."""
    x, y, w, h = DONATION
    return (rect(DONATION, rx=15, fill="none", stroke="#f2c94c", stroke_width=3.5) +
            f'<text x="{x + w / 2}" y="{y + h / 2 + 7.5}" text-anchor="middle" font-family="{FONT}" font-size="21" '
            f'font-weight="700" fill="#f2f2f2" stroke="#111" stroke-width="3" paint-order="stroke" letter-spacing="0.6"><tspan fill="#f2c94c">500 Yen</tspan><tspan dx="12">Thank you!</tspan></text>')


def dim_outside(r) -> str:
    """Darken everything but `r`: what the timelapse never looks at."""
    x, y, w, h = r
    return (f'<path fill="#000" fill-opacity="0.35" fill-rule="evenodd" '
            f'd="M0 0H{W}V{H}H0Z M{x} {y}h{w}v{h}h{-w}Z"/>')


def zone(r, colour, hatch=None, dash=False) -> str:
    x, y, w, h = r
    s = ""
    if hatch:
        s += rect(r, fill=f"url(#{hatch})")
    s += rect((x + 1.5, y + 1.5, w - 3, h - 3), fill="none", stroke=colour, stroke_width=3,
              **({"stroke_dasharray": "10 6"} if dash else {}))
    return s


def label(x, y, text, colour, anchor="start", dark_text=True) -> str:
    """A tag in the zone's colour."""
    w = 9.2 * len(text) + 20
    left = x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
    return (rect((left, y, w, 26), rx=13, fill=colour) +
            f'<text x="{left + w / 2}" y="{y + 18}" text-anchor="middle" font-family="{FONT}" font-size="15" '
            f'font-weight="600" fill="{"#111" if dark_text else "#fff"}">{text}</text>')


def note(x, y, text, anchor="start") -> str:
    """A remark on the picture: white text on a dark tag."""
    w = 7.6 * len(text) + 18
    left = x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
    return (rect((left, y, w, 24), rx=6, fill="#111", fill_opacity=0.85) +
            f'<text x="{left + w / 2}" y="{y + 16.5}" text-anchor="middle" font-family="{FONT}" font-size="13" '
            f'fill="#f2f2f2">{text}</text>')


def hatching(name: str, colour: str) -> str:
    return (f'<pattern id="{name}" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="10" height="10" fill="{colour}" fill-opacity="0.18"/>'
            f'<line x1="0" y1="0" x2="0" y2="10" stroke="{colour}" stroke-width="3" stroke-opacity="0.7"/></pattern>')


def svg(title: str, body: str, h: int = H) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img">'
            f'<title>{title}</title>'
            f'<defs>{hatching("ex", EX)}{hatching("ok", OK)}</defs>'
            f'{body}{rect((1, 1, W - 2, h - 2), fill="none", stroke="#9da1aa", stroke_width=2)}</svg>\n')


FIGURES = {
    "stream": ("A stream screen: the art program, its canvas, and overlays on top",
               screen() +
               note(97, 52, "the canvas") + note(52, 248, "tools") + note(948, 404, "layers, colours", "end") +
               note(704, 214, "the streamer's avatar", "middle") + note(196, 374, "chat", "middle") +
               note(880, 446, "follower goal", "middle"), H),
    "target": ("The zones on the frame the target is drawn on: workspace, exclusions, target",
               screen() + dim_outside(WORKSPACE) + zone(WORKSPACE, WS) + zone(AVATAR, EX, hatch="ex") +
               zone(CHAT, EX, hatch="ex") + zone(PAPER, TG) + label(788, 34, "Workspace", WS, "end") +
               label(158, 52, "Target", TG) + label(798, 220, "Exclude", EX, "end") + label(62, 374, "Exclude", EX) +
               donation() + zone(DONATION_ZONE, OK, hatch="ok") + label(50, 164, "OK", OK), H),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (title, body, h) in FIGURES.items():
        (OUT / f"{name}.svg").write_text(svg(title, body, h))
        print(OUT / f"{name}.svg")
    title, body, h = FIGURES["target"]
    PAGE_EXAMPLE.write_text(svg(title, body, h))
    print(PAGE_EXAMPLE)


if __name__ == "__main__":
    main()
