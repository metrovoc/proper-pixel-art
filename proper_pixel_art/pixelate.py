"""Main pixelation algorithm."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from proper_pixel_art import colors, mesh, utils
from proper_pixel_art.quantize import quantize_pil
from proper_pixel_art.utils import Mesh


def downsample(
    image: Image.Image,
    mesh_lines: Mesh,
    center_ratio: float = 0.5,
) -> Image.Image:
    """
    Downsample image to one pixel per mesh cell using mode color.

    Args:
        image: Input image
        mesh_lines: (lines_x, lines_y) grid coordinates
        center_ratio: Sample center portion of each cell (0.5-1.0). Reduces edge noise.
    """
    lines_x, lines_y = mesh_lines
    rgb_array = np.array(image.convert("RGB"))
    h_new, w_new = len(lines_y) - 1, len(lines_x) - 1
    out = np.zeros((h_new, w_new, 3), dtype=np.uint8)

    for j in range(h_new):
        for i in range(w_new):
            x0, x1 = lines_x[i], lines_x[i + 1]
            y0, y1 = lines_y[j], lines_y[j + 1]

            # Apply center_ratio to reduce edge noise
            if center_ratio < 1.0:
                cell_w, cell_h = x1 - x0, y1 - y0
                margin_x = int(cell_w * (1 - center_ratio) / 2)
                margin_y = int(cell_h * (1 - center_ratio) / 2)
                x0, x1 = x0 + margin_x, x1 - margin_x
                y0, y1 = y0 + margin_y, y1 - margin_y
                # Fallback if cell too small
                if x1 <= x0:
                    x0, x1 = lines_x[i], lines_x[i + 1]
                if y1 <= y0:
                    y0, y1 = lines_y[j], lines_y[j + 1]

            cell = rgb_array[y0:y1, x0:x1]
            out[j, i] = colors.get_cell_color(cell)

    return Image.fromarray(out, mode="RGB")


def pixelate(
    image: Image.Image,
    num_colors: int = 16,
    initial_upscale_factor: int = 2,
    scale_result: int | None = None,
    transparent_background: bool = False,
    intermediate_dir: Path | None = None,
    pixel_width: int | None = None,
    # New parameters
    downsample_first: bool = False,
    quantizer: Callable[[Image.Image], Image.Image] | None = None,
    center_ratio: float = 0.5,
) -> Image.Image:
    """
    Convert noisy pixel-art-style image to true pixel resolution.

    Args:
        image: Input PIL image
        num_colors: Colors for quantization (ignored if quantizer provided)
        initial_upscale_factor: Upscale for better mesh detection
        scale_result: Upscale final result by this factor
        transparent_background: Make boundary color transparent
        intermediate_dir: Save intermediate images for debugging
        pixel_width: Manual pixel width (None = auto-detect)
        downsample_first: If True, downsample then quantize (better colors).
                          If False, quantize then downsample (original algorithm).
        quantizer: Custom quantization function (Image -> Image).
                   If None, uses quantize_pil with num_colors.
        center_ratio: Sample center portion of cells (0.5-1.0). Reduces edge noise.

    Returns:
        Pixelated image with clean colors.
    """
    image_rgba = image.convert("RGBA")

    # Default quantizer
    if quantizer is None:
        quantizer = lambda img: quantize_pil(img, num_colors)

    # Detect mesh
    mesh_lines, upscale_factor = mesh.compute_mesh_with_scaling(
        image_rgba,
        initial_upscale_factor,
        output_dir=intermediate_dir,
        pixel_width=pixel_width,
    )

    # Preprocess: replace semi-transparent pixels with smart background
    image_rgb = colors.clamp_alpha(image_rgba, mode="RGB")

    # Pipeline
    if downsample_first:
        # Improved: downsample -> quantize (cleaner colors)
        scaled = utils.scale_img(image_rgb, upscale_factor)
        raw = downsample(scaled, mesh_lines, center_ratio)
        if intermediate_dir:
            raw.save(intermediate_dir / "downsampled_raw.png")
        result = quantizer(raw)
    else:
        # Original: quantize -> downsample (center_ratio=1.0 for original method)
        quantized = quantizer(image_rgb)
        if intermediate_dir:
            quantized.save(intermediate_dir / "quantized_full.png")
        scaled = utils.scale_img(quantized, upscale_factor)
        result = downsample(scaled, mesh_lines, 1.0)

    # Post-process
    if transparent_background:
        result = colors.make_background_transparent(result)

    if scale_result is not None:
        result = utils.scale_img(result, int(scale_result))

    return result
