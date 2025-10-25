"""Visual output tests"""
from pathlib import Path
from PIL import Image
from proper_pixel_art import pixelate

def test_pixelate_pngs(pixelate_png_test_params: dict):
    """Test that pixelate algorithm runs without error and generates output."""
    # Create output directory for this test asset
    for name, params in pixelate_png_test_params.items():
        output_dir = Path.cwd() / "tests" / "outputs" / str(name)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the image
        img = Image.open(params["path"])

        # Pixelate
        result = pixelate.pixelate(
            img,
            num_colors=params["num_colors"],
            scale_result=params["result_scale"],
            transparent_background=params["transparent_background"],
            intermediate_dir=output_dir,
        )

        # Save the result
        result_path = output_dir / "result.png"
        result.save(result_path)

        # Verify the output exists and has reasonable properties
        assert result_path.exists(), f"Output file not created for {name}"
        assert result.width > 0 and result.height > 0, f"Invalid dimensions for {name}"

        print(f"Generated output for {name}: {result_path}")
