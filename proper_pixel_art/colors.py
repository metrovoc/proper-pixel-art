"""Image color utilities."""

from collections import Counter

import numpy as np
from PIL import Image, ImageColor

RGB = tuple[int, int, int]


def _rgb_dist(a: RGB, b: RGB) -> int:
    """Squared RGB distance."""
    dr, dg, db = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return dr**2 + dg**2 + db**2


def _top_opaque_colors(
    img: Image.Image, alpha_threshold: int, limit: int = 8
) -> list[RGB]:
    """Return the most common opaque colors up to limit."""
    rgba = img.convert("RGBA").copy()
    rgba.thumbnail((160, 160))
    counts = Counter()
    for r, g, b, a in rgba.getdata():
        if a >= alpha_threshold:
            counts[(r, g, b)] += 1
    return [c for c, _ in counts.most_common(limit)]


def _pick_background(colors: list[RGB]) -> RGB:
    """Pick a background color far from the common colors."""
    candidates: list[RGB] = [
        (0, 255, 255),    # cyan
        (255, 255, 255),  # white
        (255, 0, 0),      # red
        (0, 255, 0),      # green
        (0, 0, 255),      # blue
        (255, 255, 0),    # yellow
        (255, 0, 255),    # magenta
        (255, 128, 0),    # orange
        (128, 0, 255),    # violet
        (0, 128, 255),    # sky
        (0, 255, 128),    # mint
        (255, 0, 128),    # pink
    ]

    if not colors:
        return (255, 255, 255)

    best, best_score = candidates[0], -1
    for c in candidates:
        score = min(_rgb_dist(c, col) for col in colors)
        if score > best_score:
            best, best_score = c, score
    return best


def clamp_alpha(
    image: Image.Image,
    alpha_threshold: int = 128,
    mode: str = "RGB",
    background_hex: str | None = None,
) -> Image.Image:
    """
    Replace transparent pixels with a background color.

    If background_hex is None, auto-pick a color far from the image's common colors.
    """
    if mode not in ("RGB", "L"):
        raise ValueError("mode must be 'RGB' or 'L'")

    if background_hex is None:
        common = _top_opaque_colors(image, alpha_threshold)
        bg_rgb = _pick_background(common)
    else:
        bg_rgb = ImageColor.getrgb(background_hex)

    base = image.convert(mode)
    alpha = image.getchannel("A")
    mask = alpha.point(lambda p: 255 if p >= alpha_threshold else 0)
    background = Image.new("RGB", image.size, bg_rgb).convert(mode)

    return Image.composite(base, background, mask)


def get_cell_color(cell_pixels: np.ndarray) -> RGB:
    """Return the most frequent RGB color in a cell."""
    flat = list(map(tuple, cell_pixels.reshape(-1, 3)))
    return Counter(flat).most_common(1)[0][0]


def most_common_boundary_color(image: Image.Image) -> RGB:
    """Return the most common color on the image boundary."""
    rgb = image.convert("RGB")
    w, h = rgb.size

    top = list(rgb.crop((0, 0, w, 1)).getdata())
    bottom = list(rgb.crop((0, h - 1, w, h)).getdata())
    left = [rgb.getpixel((0, y)) for y in range(1, h - 1)]
    right = [rgb.getpixel((w - 1, y)) for y in range(1, h - 1)]

    counts = Counter(top + bottom + left + right)
    return counts.most_common(1)[0][0]


def make_background_transparent(image: Image.Image) -> Image.Image:
    """Make the most common boundary color transparent."""
    bg_color = most_common_boundary_color(image)
    rgba = image.convert("RGBA")
    px = list(rgba.getdata())

    out = []
    for r, g, b, a in px:
        if (r, g, b) == bg_color:
            out.append((r, g, b, 0))
        else:
            out.append((r, g, b, a or 255))

    rgba.putdata(out)
    return rgba
