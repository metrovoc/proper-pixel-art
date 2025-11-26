"""Command line interface"""

import argparse
from pathlib import Path

from PIL import Image

from proper_pixel_art import pixelate
from proper_pixel_art.quantize import ClusterQuantizer, PILQuantizer, Quantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a true-resolution pixel-art image from a source image."
    )
    parser.add_argument(
        "input_path", type=Path, nargs="?", help="Path to the source input file."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path_flag",
        type=Path,
        help="Path to the source input file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_path",
        type=Path,
        default=".",
        help="Path where the pixelated image will be saved. Can be either a directory or a file path.",
    )
    parser.add_argument(
        "-c",
        "--colors",
        dest="num_colors",
        type=int,
        default=16,
        help="Number of colors to quantize the image to. From 1 to 256. Ignored if --quantizer=cluster and --auto-colors is set.",
    )
    parser.add_argument(
        "-s",
        "--scale-result",
        dest="scale_result",
        type=int,
        default=1,
        help="Width of the 'pixels' in the output image (default: 1).",
    )
    parser.add_argument(
        "-t",
        "--transparent",
        dest="transparent",
        action="store_true",
        default=False,
        help="Produce a transparent background in the output if set.",
    )
    parser.add_argument(
        "-w",
        "--pixel-width",
        dest="pixel_width",
        type=int,
        default=None,
        help="Width of the pixels in the input image. If not set, it will be determined automatically.",
    )
    parser.add_argument(
        "-u",
        "--initial-upscale",
        dest="initial_upscale",
        type=int,
        default=2,
        help=(
            "Initial image upscale factor in mesh detection algorithm. "
            "If the detected spacing is too large, "
            "it may be useful to increase this value."
        ),
    )

    # New pipeline configuration options
    parser.add_argument(
        "-p",
        "--pipeline",
        dest="pipeline",
        choices=["quantize_first", "downsample_first"],
        default="quantize_first",
        help=(
            "Processing order. 'quantize_first' (default): original algorithm. "
            "'downsample_first': improved algorithm with better color accuracy."
        ),
    )
    parser.add_argument(
        "-q",
        "--quantizer",
        dest="quantizer",
        choices=["pil", "cluster"],
        default="pil",
        help=(
            "Quantization method. 'pil' (default): fast PIL-based. "
            "'cluster': LAB-space clustering."
        ),
    )
    parser.add_argument(
        "--auto-colors",
        dest="auto_colors",
        action="store_true",
        default=False,
        help="Automatically determine the number of colors (only with --quantizer=cluster).",
    )
    parser.add_argument(
        "--color-threshold",
        dest="color_threshold",
        type=float,
        default=5.0,
        help="Color distance threshold for auto color detection (LAB Delta E). Default: 5.0",
    )
    parser.add_argument(
        "--center-ratio",
        dest="center_ratio",
        type=float,
        default=1.0,
        help="Sample only center portion of each cell (0.0-1.0). Reduces edge noise. Default: 1.0",
    )

    args = parser.parse_args()

    # Either take the input as the first argument or use the -i flag
    if args.input_path is None and args.input_path_flag is None:
        parser.error("You must provide an input path (positional or with -i).")
    args.input_path = (
        args.input_path if args.input_path is not None else args.input_path_flag
    )

    return args


def create_quantizer(args: argparse.Namespace) -> Quantizer:
    """Create quantizer based on CLI arguments."""
    if args.quantizer == "pil":
        return PILQuantizer(num_colors=args.num_colors)

    num_colors = None if args.auto_colors else args.num_colors
    return ClusterQuantizer(
        num_colors=num_colors,
        distance_threshold=args.color_threshold,
        color_space="lab",
        representative="most_frequent",
    )


def resolve_output_path(
    out_path: Path, input_path: Path, suffix: str = "_pixelated"
) -> Path:
    """
    If outpath is a directory, make it a file path
    with filename e.g. (input stem)_pixelated.png
    """
    if out_path.suffix:
        return out_path
    filename = f"{input_path.stem}{suffix}.png"
    return out_path / filename


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()

    out_path = resolve_output_path(Path(args.out_path), input_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    quantizer = create_quantizer(args)

    img = Image.open(input_path)
    pixelated = pixelate.pixelate(
        img,
        num_colors=args.num_colors,
        scale_result=args.scale_result,
        transparent_background=args.transparent,
        pixel_width=args.pixel_width,
        initial_upscale_factor=args.initial_upscale,
        pipeline=args.pipeline,
        quantizer=quantizer,
        center_ratio=args.center_ratio,
    )

    pixelated.save(out_path)


if __name__ == "__main__":
    main()
