"""Generate the Sunno app icon: a .ico for the window/taskbar and the full MSIX
asset set, plus the splash artwork shown while the app launches.

Design: a speech bubble containing three caption lines, the last one short — the shape a
real caption block makes as it wraps. Chosen because the silhouette is a single solid mass,
so it survives being drawn at 16 px where thin strokes and gradients turn to mud.

Three treatments:
  * plated  — white bubble on the accent-teal tile, for Start menu tiles and the Store.
  * unplated — teal bubble on transparency, for the taskbar and title bar, where Windows
    supplies its own background and a coloured square would look wrong.
  * splash — monochrome: one ink on a plain field, the bubble as a soft tint of that ink
    with the caption lines at full strength, and a waveform running through it. Roomier
    than an icon, so it can be artwork rather than a scaled-up logo. See ``_splash_master``.

Everything is drawn supersampled and downsampled with LANCZOS; drawing directly at 16 px
produces visibly broken geometry.

All original artwork.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "app" / "Assets"
TEAL = (31, 138, 112, 255)      # #1F8A70, the app accent
WHITE = (255, 255, 255, 255)
SS = 8                          # supersample factor


def _rounded(draw: ImageDraw.ImageDraw, box, radius, fill) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def _bubble(size: int, bubble_fill, line_fill, background=None, pad_ratio=0.14) -> Image.Image:
    """Draw the caption-bubble mark at `size` px, supersampled."""
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if background is not None:
        # Square, not rounded: Windows applies its own tile geometry (and Windows 11 rounds
        # corners itself). A pre-rounded asset shows a visible double-rounding artefact.
        d.rectangle((0, 0, s, s), fill=background)

    pad = s * pad_ratio
    body_top = pad
    body_bottom = s - pad - s * 0.13      # leave room for the tail
    body_left = pad
    body_right = s - pad

    _rounded(
        d,
        (body_left, body_top, body_right, body_bottom),
        radius=int((body_bottom - body_top) * 0.26),
        fill=bubble_fill,
    )

    # Tail: a triangle hanging from the lower-left, the conventional speech-bubble cue.
    tail_x = body_left + (body_right - body_left) * 0.22
    tail_w = (body_right - body_left) * 0.20
    d.polygon(
        [
            (tail_x, body_bottom - s * 0.02),
            (tail_x + tail_w, body_bottom - s * 0.02),
            (tail_x + tail_w * 0.15, s - pad * 0.55),
        ],
        fill=bubble_fill,
    )

    # Three caption lines, the third short — reads as wrapped text rather than a list.
    inner_w = body_right - body_left
    inner_h = body_bottom - body_top
    line_h = inner_h * 0.135
    line_x = body_left + inner_w * 0.17
    widths = (0.66, 0.66, 0.40)
    first_y = body_top + inner_h * 0.235
    gap = inner_h * 0.245

    for i, frac in enumerate(widths):
        y = first_y + i * gap
        _rounded(
            d,
            (line_x, y, line_x + inner_w * frac, y + line_h),
            radius=int(line_h / 2),
            fill=line_fill,
        )

    return img.resize((size, size), Image.LANCZOS)


def plated(size: int) -> Image.Image:
    return _bubble(size, bubble_fill=WHITE, line_fill=TEAL, background=TEAL)


def unplated(size: int) -> Image.Image:
    # Slightly tighter padding: with no tile behind it, the mark should fill more of the box.
    return _bubble(size, bubble_fill=TEAL, line_fill=WHITE, background=None, pad_ratio=0.08)


def wide(width: int, height: int) -> Image.Image:
    """Wide tile: the mark on the accent field, left-aligned like Windows' own wide tiles."""
    img = Image.new("RGBA", (width, height), TEAL)
    mark = _bubble(int(height * 0.72), bubble_fill=WHITE, line_fill=TEAL, background=None,
                   pad_ratio=0.02)
    img.paste(mark, (int(height * 0.22), int(height * 0.14)), mark)
    return img


# ---------------------------------------------------------------- splash artwork

SPLASH_W, SPLASH_H = 620, 300   # the scale-100 size Windows expects
SPLASH_SS = 8                   # the master is drawn this much larger, then downsampled
FIELD = (246, 248, 247, 255)    # #F6F8F7 — a plain, near-white field
GHOST = 0.26                    # the bubble's weight: one ink, two strengths


def _bar(d: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, fill) -> None:
    """A rounded bar. Fully rounded ends, so it reads as a stroke rather than a block."""
    d.rounded_rectangle((x, y, x + w, y + h), radius=min(w, h) / 2, fill=fill)


def _splash_master() -> Image.Image:
    """Draw the splash artwork supersampled, ready to be downsampled to each scale.

    Monochrome by construction: a single ink on a plain field, varied only in strength.
    The bubble is a soft tint and the caption lines sit inside it at full weight, which
    gives the composition a focal point without a second colour. A waveform runs through
    the middle and fades outward — speech arriving from the room and resolving into
    captions, which is the whole product in one image.

    A baked-in field rather than transparency: the package is built as a loose (non-PRI)
    layout, so exactly one splash bitmap is ever used and it has to look deliberate under
    both light and dark Windows. The manifest's BackgroundColor matches FIELD so the
    image and the window around it are one surface.
    """
    s = SPLASH_SS
    w, h = SPLASH_W * s, SPLASH_H * s
    img = Image.new("RGBA", (w, h), FIELD)
    cx, cy = w / 2, h / 2

    # --- waveform, behind the mark ---
    # Drawn on its own layer: ImageDraw writes pixels rather than compositing them, so a
    # translucent fill drawn straight onto the field would punch a hole in it.
    wave = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    wd = ImageDraw.Draw(wave)
    bar_w, pitch, tallest = 7 * s, 20 * s, 88 * s
    # Fixed heights, not random: the asset is checked in, so two runs must agree byte for
    # byte or every rebuild shows up as a diff. The two sides differ so the wave reads as
    # speech rather than as a symmetrical graphic.
    sides = {
        -1: (0.95, 0.62, 0.80, 0.38, 0.55, 0.28, 0.42, 0.20, 0.30),
        +1: (0.88, 0.70, 0.45, 0.72, 0.33, 0.50, 0.24, 0.36, 0.18),
    }
    for side, heights in sides.items():
        last = len(heights) - 1
        for i, frac in enumerate(heights):
            x = cx + side * (5 + i) * pitch     # the first five slots are the mark's
            bar_h = tallest * frac
            alpha = 0.10 + 0.40 * (1 - i / last) ** 0.9
            _bar(wd, x - bar_w / 2, cy - bar_h / 2, bar_w, bar_h,
                 TEAL[:3] + (round(255 * alpha),))
    img.alpha_composite(wave)

    # --- the bubble, as a tint ---
    body_w, body_h = 116 * s, 98 * s
    left, right = cx - body_w / 2, cx + body_w / 2
    top = cy - body_h / 2 - 7 * s              # nudged up: the tail carries weight below
    bottom = top + body_h

    shell = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shell)
    sd.rounded_rectangle((left, top, right, bottom), radius=body_h * 0.26, fill=TEAL)
    mouth_l = left + body_w * 0.29
    mouth_r = mouth_l + body_w * 0.23
    sd.polygon(
        [(mouth_l, bottom - 2 * s), (mouth_r, bottom - 2 * s),
         (mouth_l + (mouth_r - mouth_l) * 0.14, bottom + 23 * s)],
        fill=TEAL,
    )
    # Fade the finished silhouette rather than drawing its two parts translucent: where the
    # tail overlaps the body the alphas would otherwise stack and show a darker seam.
    shell.putalpha(shell.getchannel("A").point(lambda a: round(a * GHOST)))
    img.alpha_composite(shell)

    # --- caption lines, full strength ---
    d = ImageDraw.Draw(img)
    line_h, gap = 11 * s, 13 * s
    widths = (0.66, 0.66, 0.40)
    block = len(widths) * line_h + (len(widths) - 1) * gap
    y = top + (body_h - block) / 2
    for frac in widths:
        _bar(d, left + body_w * 0.17, y, body_w * frac, line_h, TEAL)
        y += line_h + gap

    return img


def write_ico(images: list[Image.Image], path: Path) -> None:
    """Write a multi-frame .ico.

    Pillow's ICO writer ignores ``append_images`` and emits a single frame, and its
    ``sizes=`` form just downsamples one source bitmap. Writing the container directly keeps
    the per-size render — which matters most at 16 px, where the geometry is hand-tuned.

    Frames are PNG-compressed, which Windows has accepted at every size since Vista and is
    well within this app's 10.0.17763 floor.
    """
    import io
    import struct

    encoded: list[bytes] = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(buf.getvalue())

    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries = b""
    for img, blob in zip(images, encoded):
        w = 0 if img.width >= 256 else img.width      # 0 means 256 in the ICO format
        h = 0 if img.height >= 256 else img.height
        entries += struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)

    path.write_bytes(header + entries + b"".join(encoded))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(img: Image.Image, name: str) -> None:
        img.save(OUT / name)
        written.append(name)

    # --- .ico for the unpackaged exe, window title bar and taskbar ---
    ico_sizes = (16, 24, 32, 48, 64, 128, 256)
    write_ico([unplated(n) for n in ico_sizes], OUT / "AppIcon.ico")
    written.append("AppIcon.ico")

    # --- MSIX assets ---
    # Square tiles, plated, across the standard scale factors.
    for base, dim in (("Square150x150Logo", 150), ("Square44x44Logo", 44),
                      ("Square71x71Logo", 71), ("Square310x310Logo", 310),
                      ("StoreLogo", 50)):
        for scale in (100, 125, 150, 200, 400):
            px = round(dim * scale / 100)
            save(plated(px), f"{base}.scale-{scale}.png")
        save(plated(dim), f"{base}.png")

    # Wide tile.
    for scale in (100, 125, 150, 200, 400):
        save(wide(round(310 * scale / 100), round(150 * scale / 100)),
             f"Wide310x150Logo.scale-{scale}.png")
    save(wide(310, 150), "Wide310x150Logo.png")

    # Target-size variants drive the taskbar, Alt-Tab and File Explorer. The unplated
    # forms are what Windows uses when it draws its own background behind the icon.
    for px in (16, 24, 32, 48, 256):
        save(plated(px), f"Square44x44Logo.targetsize-{px}.png")
        save(unplated(px), f"Square44x44Logo.targetsize-{px}_altform-unplated.png")

    # Splash screen: monochrome artwork, drawn once and downsampled per scale. Unlike the
    # tiles this is never seen small, so it can afford a composition rather than a logo.
    master = _splash_master()
    for scale in (100, 125, 150, 200, 400):
        size = (round(SPLASH_W * scale / 100), round(SPLASH_H * scale / 100))
        save(master.resize(size, Image.LANCZOS), f"SplashScreen.scale-{scale}.png")

    print(f"wrote {len(written)} files to {OUT}")
    total = sum((OUT / n.split(' ')[0]).stat().st_size for n in written) / 1024
    print(f"total {total:.0f} KB")


if __name__ == "__main__":
    main()
