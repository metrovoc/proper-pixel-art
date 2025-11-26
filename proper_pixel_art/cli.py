"""Command line interface."""

import argparse
from pathlib import Path

from PIL import Image

from proper_pixel_art import pixelate
from proper_pixel_art.quantize import quantize_cluster, quantize_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pixel-art-style images to true pixel resolution."
    )
    parser.add_argument(
        "input_path", type=Path, nargs="?", help="Input image path"
    )
    parser.add_argument(
        "-i", "--input", dest="input_path_flag", type=Path, help="Input image path"
    )
    parser.add_argument(
        "-o", "--output", dest="out_path", type=Path, default=".",
        help="Output path (directory or file). Default: current directory"
    )
    parser.add_argument(
        "-c", "--colors", dest="num_colors", type=int, default=16,
        help="Number of colors (1-256). Default: 16"
    )
    parser.add_argument(
        "-s", "--scale-result", dest="scale_result", type=int, default=1,
        help="Upscale output by this factor. Default: 1"
    )
    parser.add_argument(
        "-t", "--transparent", action="store_true",
        help="Make background transparent"
    )
    parser.add_argument(
        "-w", "--pixel-width", dest="pixel_width", type=int, default=None,
        help="Manual pixel width (default: auto-detect)"
    )
    parser.add_argument(
        "-u", "--initial-upscale", dest="initial_upscale", type=int, default=2,
        help="Upscale factor for mesh detection. Default: 2"
    )

    # Algorithm options (new defaults: downsample_first + LAB clustering + auto colors)
    parser.add_argument(
        "--no-downsample-first", dest="downsample_first", action="store_false",
        help="Quantize then downsample (original behavior)"
    )
    parser.add_argument(
        "--no-cluster", dest="use_cluster", action="store_false",
        help="Use PIL quantization instead of LAB clustering"
    )
    parser.add_argument(
        "--no-auto-colors", dest="auto_colors", action="store_false",
        help="Use fixed color count instead of auto-detection"
    )
    parser.add_argument(
        "--threshold", dest="threshold", type=float, default=1.0,
        help="Color distance threshold for auto-colors (LAB Delta E). Lower = more colors, higher = purer colors. Default: 1.0"
    )
    parser.add_argument(
        "--center-ratio", dest="center_ratio", type=float, default=0.5,
        help="Sample center portion of cells (0.5-1.0). Default: 0.5"
    )
    parser.set_defaults(downsample_first=True, use_cluster=True, auto_colors=True)

    args = parser.parse_args()

    # Resolve input path
    if args.input_path is None and args.input_path_flag is None:
        parser.error("Input path required (positional or -i)")
    args.input_path = args.input_path or args.input_path_flag

    return args


def resolve_output_path(out_path: Path, input_path: Path) -> Path:
    """If out_path is a directory, create filename from input stem."""
    if out_path.suffix:
        return out_path
    return out_path / f"{input_path.stem}_pixelated.png"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()
    out_path = resolve_output_path(Path(args.out_path), input_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    # LAB clustering requires downsample_first (O(n²) memory on full image)
    if args.use_cluster and not args.downsample_first:
        print("Warning: LAB clustering disabled (requires --downsample-first due to O(n²) memory)")
        args.use_cluster = False

    # Build quantizer
    if args.use_cluster:
        num_colors = None if args.auto_colors else args.num_colors
        quantizer = lambda img: quantize_cluster(img, num_colors, args.threshold)
    else:
        quantizer = lambda img: quantize_pil(img, args.num_colors)

    img = Image.open(input_path)
    result = pixelate.pixelate(
        img,
        num_colors=args.num_colors,
        scale_result=args.scale_result if args.scale_result > 1 else None,
        transparent_background=args.transparent,
        pixel_width=args.pixel_width,
        initial_upscale_factor=args.initial_upscale,
        downsample_first=args.downsample_first,
        quantizer=quantizer,
        center_ratio=args.center_ratio,
    )
    result.save(out_path)


if __name__ == "__main__":
    main()
