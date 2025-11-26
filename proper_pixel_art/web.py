"""Web interface for Proper Pixel Art using Gradio."""

from PIL import Image

from proper_pixel_art.pixelate import pixelate


def process(
    image: Image.Image | None,
    num_colors: int,
    transparent: bool,
    scale: int,
    initial_upscale: int,
    pixel_width: int,
) -> Image.Image | None:
    """Process image through pixelation pipeline."""
    if image is None:
        return None
    return pixelate(
        image,
        num_colors=num_colors,
        transparent_background=transparent,
        scale_result=scale if scale > 1 else None,
        initial_upscale_factor=initial_upscale,
        pixel_width=pixel_width if pixel_width > 0 else None,
    )


def create_demo():
    """Create Gradio demo interface."""
    import gradio as gr

    return gr.Interface(
        fn=process,
        inputs=[
            gr.Image(type="pil", label="Input", format="png", image_mode="RGBA"),
            gr.Slider(2, 64, value=16, step=1, label="Colors"),
            gr.Checkbox(value=False, label="Transparent Background"),
            gr.Slider(1, 20, value=1, step=1, label="Scale Result"),
            gr.Slider(1, 4, value=2, step=1, label="Initial Upscale (for mesh detection)"),
            gr.Slider(0, 50, value=0, step=1, label="Pixel Width (0 = auto)"),
        ],
        outputs=gr.Image(type="pil", label="Output", format="png", image_mode="RGBA"),
        title="Proper Pixel Art",
        description="Convert AI-generated pixel art to true pixel resolution",
        flagging_mode="never",
    )


def main():
    """Entry point for ppa-web command."""
    demo = create_demo()
    demo.launch()


if __name__ == "__main__":
    main()
