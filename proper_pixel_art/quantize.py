"""
Quantization strategies for pixel art color reduction.

This module provides different approaches to reduce colors in an image:
- PILQuantizer: Fast, uses PIL's built-in quantization (MAXCOVERAGE/MEDIANCUT/FASTOCTREE)
- ClusterQuantizer: Advanced, uses LAB color space + hierarchical clustering
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from PIL import Image
from PIL.Image import Quantize
from skimage import color as skcolor
from sklearn.cluster import AgglomerativeClustering

RGB = tuple[int, int, int]


class Quantizer(ABC):
    """Abstract base class for color quantization strategies."""

    @abstractmethod
    def __call__(self, image: Image.Image) -> Image.Image:
        """Quantize the image colors and return the result."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this quantizer."""
        ...


class PILQuantizer(Quantizer):
    """
    PIL-based color quantization.

    Uses PIL's built-in quantize() method with configurable algorithm.
    Fast but may produce "compromise colors" not present in the original.
    """

    def __init__(
        self,
        num_colors: int = 16,
        method: int = Quantize.MAXCOVERAGE,
    ):
        """
        Args:
            num_colors: Target number of colors (8, 16, 32, 64 typically work well)
            method: PIL quantization algorithm (MAXCOVERAGE, MEDIANCUT, FASTOCTREE)
        """
        self.num_colors = num_colors
        self.method = method

    @property
    def name(self) -> str:
        method_names = {
            Quantize.MAXCOVERAGE: "MAXCOVERAGE",
            Quantize.MEDIANCUT: "MEDIANCUT",
            Quantize.FASTOCTREE: "FASTOCTREE",
        }
        return f"PIL({method_names.get(self.method, 'UNKNOWN')}, {self.num_colors})"

    def __call__(self, image: Image.Image) -> Image.Image:
        image_rgb = image.convert("RGB")
        quantized = image_rgb.quantize(
            colors=self.num_colors,
            method=self.method,
            dither=Image.Dither.NONE,
        )
        return quantized.convert("RGB")


class ClusterQuantizer(Quantizer):
    """
    Clustering-based color quantization using LAB color space.

    Uses hierarchical agglomerative clustering in perceptually uniform LAB space.
    Can automatically determine the number of colors based on a distance threshold.
    """

    def __init__(
        self,
        num_colors: int | None = None,
        distance_threshold: float = 5.0,
        color_space: Literal["rgb", "lab"] = "lab",
        representative: Literal["centroid", "most_frequent"] = "most_frequent",
    ):
        """
        Args:
            num_colors: Fixed number of colors. If None, auto-determine using distance_threshold.
            distance_threshold: Max color distance within a cluster (LAB Delta E).
                               Only used when num_colors is None. Typical values: 3.0-8.0
            color_space: Color space for distance calculation ("lab" recommended)
            representative: How to pick the final color for each cluster:
                           "centroid" = cluster center (may create new colors)
                           "most_frequent" = most common original color (preserves originals)
        """
        self.num_colors = num_colors
        self.distance_threshold = distance_threshold
        self.color_space = color_space
        self.representative = representative

    @property
    def name(self) -> str:
        if self.num_colors is not None:
            return f"Cluster({self.color_space.upper()}, k={self.num_colors})"
        return f"Cluster({self.color_space.upper()}, threshold={self.distance_threshold})"

    def __call__(self, image: Image.Image) -> Image.Image:
        image_rgb = image.convert("RGB")
        pixels = np.array(image_rgb).reshape(-1, 3)

        # Get unique colors and their counts for efficiency
        unique_colors, inverse_indices, counts = np.unique(
            pixels, axis=0, return_inverse=True, return_counts=True
        )

        if len(unique_colors) <= 1:
            return image_rgb

        # Convert to working color space
        if self.color_space == "lab":
            # skimage expects float [0,1] for rgb2lab
            unique_float = unique_colors.astype(np.float64) / 255.0
            unique_working = skcolor.rgb2lab(unique_float.reshape(1, -1, 3)).reshape(
                -1, 3
            )
        else:
            unique_working = unique_colors.astype(np.float64)

        # Clustering
        if self.num_colors is not None:
            n_clusters = min(self.num_colors, len(unique_colors))
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="euclidean",
                linkage="complete",
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="euclidean",
                linkage="complete",
            )

        labels = clustering.fit_predict(unique_working)
        n_clusters = labels.max() + 1

        # Determine representative color for each cluster
        palette_rgb = np.zeros((n_clusters, 3), dtype=np.uint8)

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_colors = unique_colors[mask]
            cluster_counts = counts[mask]

            if self.representative == "most_frequent":
                # Pick the most frequent original color in this cluster
                best_idx = np.argmax(cluster_counts)
                palette_rgb[cluster_id] = cluster_colors[best_idx]
            else:
                # Centroid: average in working space, convert back
                cluster_working = unique_working[mask]
                centroid = np.mean(cluster_working, axis=0)

                if self.color_space == "lab":
                    centroid_rgb = skcolor.lab2rgb(centroid.reshape(1, 1, 3)).reshape(3)
                    palette_rgb[cluster_id] = (centroid_rgb * 255).astype(np.uint8)
                else:
                    palette_rgb[cluster_id] = centroid.astype(np.uint8)

        # Map each pixel to its cluster's representative color
        pixel_labels = labels[inverse_indices]
        result_pixels = palette_rgb[pixel_labels]
        result_array = result_pixels.reshape(np.array(image_rgb).shape)

        return Image.fromarray(result_array, mode="RGB")


def create_quantizer(
    method: Literal["pil", "cluster"] = "pil",
    num_colors: int | None = 16,
    **kwargs,
) -> Quantizer:
    """
    Factory function to create a quantizer.

    Args:
        method: "pil" for PILQuantizer, "cluster" for ClusterQuantizer
        num_colors: Number of colors (None for auto in cluster mode)
        **kwargs: Additional arguments passed to the quantizer constructor

    Returns:
        A Quantizer instance
    """
    if method == "pil":
        if num_colors is None:
            num_colors = 16
        return PILQuantizer(num_colors=num_colors, **kwargs)
    elif method == "cluster":
        return ClusterQuantizer(num_colors=num_colors, **kwargs)
    else:
        raise ValueError(f"Unknown quantizer method: {method}")
