"""Generate the Sunno app icon: a .ico for the window/taskbar and the full MSIX
asset set.

Design: a speech bubble containing three caption lines, the last one short — the shape a
real caption block makes as it wraps. Chosen because the silhouette is a single solid mass,
so it survives being drawn at 16 px where thin strokes and gradients turn to mud.

Two treatments:
  * plated  — white bubble on the accent-teal tile, for Start menu tiles and the Store.
  * unplated — teal bubble on transparency, for the taskbar and title bar, where Windows
    supplies its own background and a coloured square would look wrong.

Everything is drawn at 8x and downsampled with LANCZOS; drawing directly at 16 px produces
visibly broken geometry.

All original artwork.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(r"D:\Code\Live Speech to Text\app\Assets")
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

    # Splash screen: the mark centred on the accent field.
    for scale in (100, 125, 150, 200, 400):
        w, h = round(620 * scale / 100), round(300 * scale / 100)
        img = Image.new("RGBA", (w, h), TEAL)
        mark = _bubble(int(h * 0.46), bubble_fill=WHITE, line_fill=TEAL, background=None,
                       pad_ratio=0.02)
        img.paste(mark, ((w - mark.width) // 2, (h - mark.height) // 2), mark)
        save(img, f"SplashScreen.scale-{scale}.png")

    print(f"wrote {len(written)} files to {OUT}")
    total = sum((OUT / n.split(' ')[0]).stat().st_size for n in written) / 1024
    print(f"total {total:.0f} KB")


if __name__ == "__main__":
    main()
