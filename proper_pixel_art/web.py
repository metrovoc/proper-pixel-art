"""Web interface for Proper Pixel Art using Gradio."""

from PIL import Image

from proper_pixel_art.pixelate import pixelate
from proper_pixel_art.quantize import ClusterQuantizer, PILQuantizer, Quantizer

IMG_HEIGHT = 512


def create_quantizer(
    method: str,
    num_colors: int,
    auto_colors: bool,
    color_threshold: float,
) -> Quantizer:
    """Create quantizer based on UI selections."""
    if method == "pil":
        return PILQuantizer(num_colors=num_colors)

    return ClusterQuantizer(
        num_colors=None if auto_colors else num_colors,
        distance_threshold=color_threshold,
        color_space="lab",
        representative="most_frequent",
    )


def process(
    image: Image.Image | None,
    preset: str,
    num_colors: int,
    transparent: bool,
    scale: int,
    initial_upscale: int,
    pixel_width: int,
    center_ratio: float,
    auto_colors: bool,
    color_threshold: float,
) -> Image.Image | None:
    """Process image through pixelation pipeline."""
    if image is None:
        return None

    # Determine pipeline and quantizer from preset
    if preset == "fast":
        pipeline = "quantize_first"
        quantizer_method = "pil"
    elif preset == "balanced":
        pipeline = "downsample_first"
        quantizer_method = "pil"
    else:  # advanced
        pipeline = "downsample_first"
        quantizer_method = "cluster"

    quantizer = create_quantizer(
        method=quantizer_method,
        num_colors=num_colors,
        auto_colors=auto_colors,
        color_threshold=color_threshold,
    )

    return pixelate(
        image,
        num_colors=num_colors,
        transparent_background=transparent,
        scale_result=scale if scale > 1 else None,
        initial_upscale_factor=initial_upscale,
        pixel_width=pixel_width if pixel_width > 0 else None,
        pipeline=pipeline,
        quantizer=quantizer,
        center_ratio=center_ratio,
    )


def create_demo():
    """Create Gradio demo interface."""
    import gradio as gr

    # Preset descriptions
    preset_info = {
        "fast": "**Fast**: Original algorithm. Quick but may produce slightly inaccurate colors.",
        "balanced": "**Balanced**: Improved color accuracy. Recommended for most use cases.",
        "advanced": "**Advanced**: LAB color space + clustering. Best quality, auto color detection available.",
    }

    with gr.Blocks(title="Proper Pixel Art") as demo:
        gr.Markdown(
            "# Proper Pixel Art\n"
            "Convert AI-generated pixel art to true pixel resolution"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    type="pil",
                    label="Input",
                    format="png",
                    image_mode="RGBA",
                    height=IMG_HEIGHT,
                )
            with gr.Column(scale=1):
                output_img = gr.Image(
                    type="pil",
                    label="Output",
                    format="png",
                    image_mode="RGBA",
                    height=IMG_HEIGHT,
                    interactive=False,
                )

        # Main controls
        with gr.Row():
            preset = gr.Radio(
                choices=["fast", "balanced", "advanced"],
                value="balanced",
                label="Mode",
                info="Select processing mode",
            )

        preset_desc = gr.Markdown(preset_info["balanced"])

        with gr.Row():
            num_colors = gr.Slider(
                2,
                64,
                value=16,
                step=1,
                label="Colors",
                info="Target number of colors",
            )
            scale = gr.Slider(
                1, 20, value=1, step=1, label="Scale", info="Output upscaling factor"
            )

        with gr.Row():
            transparent = gr.Checkbox(
                value=False, label="Transparent Background", scale=1
            )
            btn = gr.Button("Pixelate", variant="primary", scale=2)

        # Advanced options (collapsible)
        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                initial_upscale = gr.Slider(
                    1,
                    4,
                    value=2,
                    step=1,
                    label="Initial Upscale",
                    info="Upscale factor for mesh detection",
                )
                pixel_width = gr.Slider(
                    0,
                    50,
                    value=0,
                    step=1,
                    label="Pixel Width",
                    info="0 = auto detect",
                )
                center_ratio = gr.Slider(
                    0.5,
                    1.0,
                    value=0.5,
                    step=0.1,
                    label="Center Ratio",
                    info="Sample center portion of each cell. Rarely needs adjustment.",
                )

            # Cluster-specific options
            cluster_options = gr.Group(visible=False)
            with cluster_options:
                gr.Markdown("**Cluster Quantizer Options**")
                with gr.Row():
                    auto_colors = gr.Checkbox(
                        value=False,
                        label="Auto Colors",
                        info="Automatically determine color count",
                    )
                    color_threshold = gr.Slider(
                        1.0,
                        15.0,
                        value=5.0,
                        step=0.5,
                        label="Color Threshold",
                        info="LAB Delta E threshold for merging colors (lower = more colors)",
                    )

        # Event handlers
        def update_preset_desc(preset_value):
            return preset_info[preset_value]

        def update_cluster_visibility(preset_value):
            return gr.update(visible=(preset_value == "advanced"))

        def update_colors_interactive(auto_colors_value, preset_value):
            # Disable num_colors when auto_colors is enabled in advanced mode
            if preset_value == "advanced" and auto_colors_value:
                return gr.update(interactive=False)
            return gr.update(interactive=True)

        preset.change(
            fn=update_preset_desc,
            inputs=[preset],
            outputs=[preset_desc],
        )

        preset.change(
            fn=update_cluster_visibility,
            inputs=[preset],
            outputs=[cluster_options],
        )

        auto_colors.change(
            fn=update_colors_interactive,
            inputs=[auto_colors, preset],
            outputs=[num_colors],
        )

        preset.change(
            fn=lambda p: update_colors_interactive(False, p),
            inputs=[preset],
            outputs=[num_colors],
        )

        btn.click(
            fn=process,
            inputs=[
                input_img,
                preset,
                num_colors,
                transparent,
                scale,
                initial_upscale,
                pixel_width,
                center_ratio,
                auto_colors,
                color_threshold,
            ],
            outputs=output_img,
        )

    return demo


def main():
    """Entry point for ppa-web command."""
    demo = create_demo()
    demo.launch()


if __name__ == "__main__":
    main()
