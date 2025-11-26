"""Web interface using Gradio."""

from PIL import Image

from proper_pixel_art.pixelate import pixelate
from proper_pixel_art.quantize import quantize_cluster, quantize_pil


def process(
    image: Image.Image | None,
    num_colors: int,
    transparent: bool,
    scale: int,
    initial_upscale: int,
    pixel_width: int,
    downsample_first: bool,
    use_cluster: bool,
    auto_colors: bool,
    threshold: float,
    center_ratio: float,
) -> Image.Image | None:
    """Process image through pixelation pipeline."""
    if image is None:
        return None

    # Build quantizer
    if use_cluster:
        n = None if auto_colors else num_colors
        quantizer = lambda img: quantize_cluster(img, n, threshold)
    else:
        quantizer = lambda img: quantize_pil(img, num_colors)

    return pixelate(
        image,
        num_colors=num_colors,
        transparent_background=transparent,
        scale_result=scale if scale > 1 else None,
        initial_upscale_factor=initial_upscale,
        pixel_width=pixel_width if pixel_width > 0 else None,
        downsample_first=downsample_first,
        quantizer=quantizer,
        center_ratio=center_ratio,
    )


def create_demo():
    """Create Gradio demo interface."""
    import gradio as gr

    with gr.Blocks(title="Proper Pixel Art") as demo:
        gr.Markdown(
            "# Proper Pixel Art\n"
            "Convert AI-generated pixel art to true pixel resolution"
        )

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(
                    type="pil", label="Input", format="png",
                    image_mode="RGBA", height=512,
                )
            with gr.Column():
                output_img = gr.Image(
                    type="pil", label="Output", format="png",
                    image_mode="RGBA", height=512, interactive=False,
                )

        # Main controls
        with gr.Row():
            num_colors = gr.Slider(2, 64, value=16, step=1, label="Colors", interactive=False)
            scale = gr.Slider(1, 20, value=20, step=1, label="Scale")

        with gr.Row():
            transparent = gr.Checkbox(value=False, label="Transparent Background")
            btn = gr.Button("Pixelate", variant="primary", scale=2)

        # Algorithm options
        with gr.Accordion("Algorithm Options", open=False):
            with gr.Row():
                downsample_first = gr.Checkbox(
                    value=True, label="Downsample First",
                    info="Better color accuracy. Disable to quantize first (original behavior)."
                )
                use_cluster = gr.Checkbox(
                    value=True, label="LAB Clustering",
                    info="Perceptually uniform color quantization"
                )

            with gr.Row():
                auto_colors = gr.Checkbox(
                    value=True, label="Auto Colors",
                    info="Auto-detect color count (requires LAB Clustering)",
                    interactive=True,
                )
                threshold = gr.Slider(
                    1.0, 15.0, value=1.0, step=0.5,
                    label="Color Threshold",
                    info="Lower = more colors, higher = purer colors (requires Auto Colors)",
                    interactive=True,
                )

        # Advanced options
        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                initial_upscale = gr.Slider(
                    1, 4, value=2, step=1, label="Initial Upscale",
                    info="Upscale factor for mesh detection"
                )
                pixel_width = gr.Slider(
                    0, 50, value=0, step=1, label="Pixel Width",
                    info="Manual pixel width (0 = auto detect)"
                )
                center_ratio = gr.Slider(
                    0.5, 1.0, value=0.5, step=0.1, label="Center Ratio",
                    info="Sample center of each cell to reduce edge noise. You probably don't need to change this (requires Downsample First)",
                    interactive=True,  # downsample_first default True
                )

        # Unified UI state update
        def update_ui(ds_first, cluster, auto):
            # LAB clustering requires downsample_first (O(n²) memory on full image)
            cluster_allowed = ds_first
            effective_cluster = cluster_allowed and cluster
            effective_auto = effective_cluster and auto
            return (
                gr.update(interactive=not effective_auto),           # num_colors
                gr.update(value=effective_cluster, interactive=cluster_allowed),  # use_cluster
                gr.update(interactive=effective_cluster),            # auto_colors
                gr.update(interactive=effective_auto),               # threshold
                gr.update(interactive=ds_first),                     # center_ratio
            )

        for ctrl in [downsample_first, use_cluster, auto_colors]:
            ctrl.change(
                fn=update_ui,
                inputs=[downsample_first, use_cluster, auto_colors],
                outputs=[num_colors, use_cluster, auto_colors, threshold, center_ratio],
            )

        btn.click(
            fn=process,
            inputs=[
                input_img, num_colors, transparent, scale,
                initial_upscale, pixel_width, downsample_first,
                use_cluster, auto_colors, threshold, center_ratio,
            ],
            outputs=output_img,
        )

    return demo


def main():
    """Entry point for ppa-web command."""
    create_demo().launch()


if __name__ == "__main__":
    main()
