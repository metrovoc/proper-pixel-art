"""Main functions for pixelating an image with the pixelate function"""

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from proper_pixel_art import colors, mesh, utils
from proper_pixel_art.quantize import PILQuantizer, Quantizer
from proper_pixel_art.utils import Mesh

PipelineOrder = Literal["quantize_first", "downsample_first"]


def downsample(
    image: Image.Image,
    mesh_lines: Mesh,
    center_ratio: float = 1.0,
) -> Image.Image:
    """
    Downsample the image by looping over each cell in mesh and
    using the most common color as the pixel color.

    Args:
        image: Input image
        mesh_lines: (lines_x, lines_y) grid coordinates
        center_ratio: Sample only the center portion of each cell (0.0-1.0).
                      1.0 = entire cell, 0.5 = center 50%. Helps reduce edge noise.
    """
    lines_x, lines_y = mesh_lines
    rgb = image.convert("RGB")
    rgb_array = np.array(rgb)
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
                # Ensure at least 1 pixel
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
    # New parameters for pipeline configuration
    pipeline: PipelineOrder = "quantize_first",
    quantizer: Quantizer | None = None,
    center_ratio: float = 1.0,
) -> Image.Image:
    """
    Computes the true resolution pixel art image.

    Args:
        image: A PIL image to pixelate.

        num_colors: The number of colors to use when quantizing the image.
            Only used if quantizer is None. If too high, pixels that should be
            the same color will be different colors. If too low, pixels that
            should be different colors will be the same color.

        scale_result: Upsample result by this factor after algorithm is complete.

        initial_upscale_factor: Upsample original image by this factor for
            better mesh detection.

        transparent_background: If True, makes the most common boundary color
            transparent in the result.

        intermediate_dir: Directory to save images visualizing intermediate steps.

        pixel_width: If set, skips automatic pixel width detection and uses
            this value instead.

        pipeline: Processing order for quantization and downsampling.
            - "quantize_first": Original algorithm. Quantize colors on the full
              image, then downsample. Fast but may produce "compromise colors".
            - "downsample_first": Improved algorithm. Downsample first to extract
              representative colors, then quantize. Better color accuracy.

        quantizer: Custom quantization strategy. If None, uses PILQuantizer with
            num_colors. For advanced usage, pass a ClusterQuantizer for LAB-space
            clustering with automatic color count detection.

        center_ratio: When downsampling, only sample the center portion of each
            cell (0.0-1.0). 1.0 = entire cell, 0.5 = center 50%. Reduces edge
            noise from grid misalignment.

    Returns:
        The true pixelated image.

    Examples:
        # Original behavior (backward compatible)
        result = pixelate(image, num_colors=16)

        # Improved: downsample first, then quantize
        result = pixelate(image, num_colors=16, pipeline="downsample_first")

        # Advanced: LAB clustering with auto color detection
        from proper_pixel_art.quantize import ClusterQuantizer
        result = pixelate(
            image,
            pipeline="downsample_first",
            quantizer=ClusterQuantizer(distance_threshold=5.0)
        )
    """
    image_rgba = image.convert("RGBA")

    # Create default quantizer if not provided
    if quantizer is None:
        quantizer = PILQuantizer(num_colors=num_colors)

    # Calculate the pixel mesh lines
    mesh_lines, upscale_factor = mesh.compute_mesh_with_scaling(
        image_rgba,
        initial_upscale_factor,
        output_dir=intermediate_dir,
        pixel_width=pixel_width,
    )

    # Preprocess: replace semi-transparent pixels with a smart background color.
    # This prevents alpha artifacts from polluting the color quantization.
    # Critical for GPT-4o images which often have imperfect transparency.
    image_rgb = colors.clamp_alpha(image_rgba, mode="RGB")

    if pipeline == "quantize_first":
        # Original flow: quantize -> scale -> downsample
        result = _pipeline_quantize_first(
            image_rgb,
            mesh_lines,
            upscale_factor,
            quantizer,
            center_ratio,
            intermediate_dir,
        )
    else:
        # Improved flow: scale -> downsample -> quantize
        result = _pipeline_downsample_first(
            image_rgb,
            mesh_lines,
            upscale_factor,
            quantizer,
            center_ratio,
            intermediate_dir,
        )

    # Apply transparency
    if transparent_background:
        result = colors.make_background_transparent(result)

    # Upscale the result if requested
    if scale_result is not None:
        result = utils.scale_img(result, int(scale_result))

    return result


def _pipeline_quantize_first(
    image: Image.Image,
    mesh_lines: Mesh,
    upscale_factor: int,
    quantizer: Quantizer,
    center_ratio: float,
    intermediate_dir: Path | None,
) -> Image.Image:
    """
    Original pipeline: quantize on full image, then downsample.

    Flow: image -> quantize -> scale -> downsample -> result
    """
    # Quantize colors on the full image
    quantized = quantizer(image)

    if intermediate_dir is not None:
        quantized.save(intermediate_dir / "quantized_full.png")

    # Scale to match mesh dimensions
    scaled = utils.scale_img(quantized, upscale_factor)

    # Downsample by taking mode color per cell
    result = downsample(scaled, mesh_lines, center_ratio)

    return result


def _pipeline_downsample_first(
    image: Image.Image,
    mesh_lines: Mesh,
    upscale_factor: int,
    quantizer: Quantizer,
    center_ratio: float,
    intermediate_dir: Path | None,
) -> Image.Image:
    """
    Improved pipeline: downsample first to extract clean colors, then quantize.

    Flow: image -> scale -> downsample -> quantize -> result

    This reduces noise because:
    1. Mode extraction removes within-cell color variations
    2. Quantization operates on clean, representative colors
    """
    # Scale original image to match mesh dimensions
    scaled = utils.scale_img(image, upscale_factor)

    # Downsample by taking mode color per cell (noise removal)
    raw_result = downsample(scaled, mesh_lines, center_ratio)

    if intermediate_dir is not None:
        raw_result.save(intermediate_dir / "downsampled_raw.png")

    # Quantize the clean downsampled image
    result = quantizer(raw_result)

    if intermediate_dir is not None:
        result.save(intermediate_dir / "quantized_final.png")

    return result
