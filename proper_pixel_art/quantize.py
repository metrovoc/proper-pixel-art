"""
Color quantization functions for pixel art.

Two approaches:
- quantize_pil: Fast PIL-based quantization
- quantize_cluster: LAB-space clustering with optional auto color detection
"""

import numpy as np
from PIL import Image
from PIL.Image import Quantize
from skimage import color as skcolor
from sklearn.cluster import AgglomerativeClustering


def quantize_pil(image: Image.Image, num_colors: int = 16) -> Image.Image:
    """
    Fast PIL-based color quantization.

    Args:
        image: Input image
        num_colors: Target number of colors (8, 16, 32, 64 typically work well)

    Returns:
        Quantized RGB image
    """
    rgb = image.convert("RGB")
    quantized = rgb.quantize(
        colors=num_colors,
        method=Quantize.MAXCOVERAGE,
        dither=Image.Dither.NONE,
    )
    return quantized.convert("RGB")


def quantize_cluster(
    image: Image.Image,
    num_colors: int | None = None,
    threshold: float = 5.0,
) -> Image.Image:
    """
    LAB-space clustering quantization.

    Uses hierarchical agglomerative clustering in perceptually uniform LAB space.
    Picks the most frequent original color in each cluster (no "compromise colors").

    Args:
        image: Input image
        num_colors: Fixed number of colors. If None, auto-determine using threshold.
        threshold: Max color distance (LAB Delta E) within a cluster.
                   Only used when num_colors is None. Typical values: 3.0-8.0

    Returns:
        Quantized RGB image
    """
    rgb = image.convert("RGB")
    pixels = np.array(rgb).reshape(-1, 3)

    # Get unique colors and their counts
    unique_colors, inverse_indices, counts = np.unique(
        pixels, axis=0, return_inverse=True, return_counts=True
    )

    if len(unique_colors) <= 1:
        return rgb

    # Convert to LAB for perceptually uniform distance
    unique_float = unique_colors.astype(np.float64) / 255.0
    unique_lab = skcolor.rgb2lab(unique_float.reshape(1, -1, 3)).reshape(-1, 3)

    # Cluster
    if num_colors is not None:
        n_clusters = min(num_colors, len(unique_colors))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean",
            linkage="complete",
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="euclidean",
            linkage="complete",
        )

    labels = clustering.fit_predict(unique_lab)
    n_clusters = labels.max() + 1

    # Pick most frequent original color per cluster
    palette = np.zeros((n_clusters, 3), dtype=np.uint8)
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_counts = counts[mask]
        best_idx = np.argmax(cluster_counts)
        palette[cluster_id] = unique_colors[mask][best_idx]

    # Map pixels to palette
    result_pixels = palette[labels[inverse_indices]]
    result_array = result_pixels.reshape(np.array(rgb).shape)

    return Image.fromarray(result_array, mode="RGB")
