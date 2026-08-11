"""TripoSplat UI with TRELLIS-style native multi-image sampling."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr
import torch

from multiview import run_multi_image
from triposplat import TripoSplatPipeline


PIPE = None
GENERATION_LOCK = threading.RLock()

OUT_ROOT = Path("gradio_outputs").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
VIEWER_HTML = Path("static/viewer/viewer.html").resolve()
EXAMPLES_DIR = Path("static/example_inputs").resolve()
MULTIVIEW_EXAMPLES_DIR = Path("static/trellis_multiview_examples").resolve()
EXAMPLES = [
    str(EXAMPLES_DIR / "creature_butterfly.webp"),
    str(EXAMPLES_DIR / "building_stone_house.webp"),
    str(EXAMPLES_DIR / "vehicle_pirate_ship.webp"),
    str(EXAMPLES_DIR / "plant_water_lily.webp"),
]
MULTIVIEW_EXAMPLE_SETS = [
    ("Character", ["character_1.png", "character_2.png", "character_3.png"]),
    ("Mushroom", ["mushroom_1.png", "mushroom_2.png", "mushroom_3.png"]),
    ("Orange Guy", ["orangeguy_1.png", "orangeguy_2.png", "orangeguy_3.png"]),
    ("Pop Mart", ["popmart_1.png", "popmart_2.png", "popmart_3.png"]),
    ("Rabbit", ["rabbit_1.png", "rabbit_2.png", "rabbit_3.png"]),
    ("Tiger", ["tiger_1.png", "tiger_2.png", "tiger_3.png"]),
    ("Yoimiya", ["yoimiya_1.png", "yoimiya_2.png", "yoimiya_3.png"]),
]
MULTIVIEW_EXAMPLES = [
    (label, [str(MULTIVIEW_EXAMPLES_DIR / filename) for filename in filenames])
    for label, filenames in MULTIVIEW_EXAMPLE_SETS
]

PLACEHOLDER_HTML = (
    "<div style='display:flex;align-items:center;justify-content:center;height:520px;"
    "color:#94a3b8;font:16px system-ui;background:#111318;border-radius:12px'>"
    "3D viewer will appear here after generation</div>"
)


def _gr_file(path: Path) -> str:
    return f"/gradio_api/file={path.as_posix()}"


def _viewer_iframe(ply_path: Path) -> str:
    src = f"{_gr_file(VIEWER_HTML)}?ply={_gr_file(ply_path)}&source=tripo&ts={time.time()}"
    return (
        f"<iframe src='{src}' "
        "style='width:100%;height:520px;border:0;border-radius:12px;background:#0a0b0e'></iframe>"
    )


def _load_tripo_pipe():
    global PIPE
    if PIPE is None:
        PIPE = TripoSplatPipeline(
            ckpt_path="ckpts/diffusion_models/triposplat_fp16.safetensors",
            decoder_path="ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
            dinov3_path="ckpts/clip_vision/dino_v3_vit_h.safetensors",
            flux2_vae_encoder_path="ckpts/vae/flux2-vae.safetensors",
            rmbg_path="ckpts/background_removal/birefnet.safetensors",
            device="cuda",
        )
    return PIPE


def _save_gaussian(gaussian, output_format: str):
    out_dir = OUT_ROOT / uuid4().hex[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "splat.ply"
    gaussian.save_ply(str(ply_path))

    output_format = str(output_format).lower()
    if output_format == "ply":
        download_path = ply_path
    elif output_format == "splat":
        download_path = out_dir / "splat.splat"
        gaussian.save_splat(str(download_path))
    else:
        raise gr.Error(f"Unknown output format: {output_format}")
    return _viewer_iframe(ply_path), str(download_path)


def _upload_path(upload):
    if isinstance(upload, (str, bytes, Path)) or hasattr(upload, "__fspath__"):
        return os.fspath(upload)
    for attribute in ("path", "name"):
        value = getattr(upload, attribute, None)
        if value:
            return os.fspath(value)
    return os.fspath(upload)


def collect_multiview_entries(front, back, left, right, free_uploads):
    """Return guided views in cyclic order followed by arbitrary-angle views."""
    guided_candidates = [
        ("Front-ish", front),
        ("Right-ish", right),
        ("Back-ish", back),
        ("Left-ish", left),
    ]
    entries = [
        (label, _upload_path(image), "guided")
        for label, image in guided_candidates
        if image is not None
    ]
    entries.extend(
        (f"Free angle {index + 1}", _upload_path(upload), "free")
        for index, upload in enumerate(list(free_uploads or []))
        if upload is not None
    )
    return entries


def free_upload_thumbnails(free_uploads):
    """Return thumbnail cards in the exact order used for generation."""
    uploads = [
        _upload_path(upload)
        for upload in list(free_uploads or [])
        if upload is not None
    ]
    return [
        (
            path,
            "1 - Reference orientation when no guided view is used"
            if index == 0
            else f"{index + 1} - Detail view",
        )
        for index, path in enumerate(uploads)
    ]


def _fusion_mode(label: str) -> str:
    return "multidiffusion" if str(label).startswith("MultiDiffusion") else "stochastic"


def generate_single(
    image,
    seed: int,
    steps: int,
    guidance_scale: float,
    num_gaussians: int,
    output_format: str,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Please upload an image first.")

    with GENERATION_LOCK:
        progress(0, desc="Loading TripoSplat...")
        pipe = _load_tripo_pipe()
        started = time.time()
        prepared = pipe.preprocess_image(image)
        generator = torch.Generator(device=pipe._device).manual_seed(int(seed))
        cond = pipe.encode_image(prepared, generator=generator)
        output = pipe.sample_latent(
            cond,
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            show_progress=True,
        )
        gaussian = pipe.decode_latent(output["latent"], num_gaussians=int(num_gaussians))
        viewer, download_path = _save_gaussian(gaussian, output_format)
        info = (
            f"TripoSplat single image · {gaussian.get_xyz.shape[0]:,} gaussians · "
            f"generation: {time.time() - started:.1f}s · saved: {Path(download_path).name}"
        )
        return (
            [(prepared, "Single image")],
            viewer,
            gr.update(value=download_path, interactive=True),
            info,
        )


def generate_multiview(
    front,
    back,
    left,
    right,
    free_uploads,
    fusion_label: str,
    detail_influence: float,
    seed: int,
    steps: int,
    guidance_scale: float,
    shift: float,
    num_gaussians: int,
    output_format: str,
    progress=gr.Progress(track_tqdm=True),
):
    entries = collect_multiview_entries(front, back, left, right, free_uploads)
    if not entries:
        raise gr.Error(
            "Add at least one guided side or one free-angle image. You can use either section or both."
        )

    with GENERATION_LOCK:
        progress(0, desc="Loading TripoSplat and preparing all views...")
        pipe = _load_tripo_pipe()
        mode = _fusion_mode(fusion_label)
        roles = [role for _label, _path, role in entries]
        anchor_index = next(
            (index for index, (label, _path, role) in enumerate(entries)
             if role == "guided" and label == "Front-ish"),
            next((index for index, role in enumerate(roles) if role == "guided"), None),
        )
        started = time.time()

        def on_step(step, total):
            progress(
                min(0.94, 0.10 + 0.80 * step / max(1, total)),
                desc=f"TripoSplat multi-view sampling {step}/{total}",
            )

        gaussian, prepared, metadata = run_multi_image(
            pipe,
            [path for _label, path, _role in entries],
            roles=roles,
            anchor_index=anchor_index,
            detail_influence=float(detail_influence),
            seed=int(seed),
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            shift=float(shift),
            num_gaussians=int(num_gaussians),
            mode=mode,
            show_progress=True,
            callback=on_step,
        )
        viewer, download_path = _save_gaussian(gaussian, output_format)
        guided_count = roles.count("guided")
        free_count = roles.count("free")
        if guided_count and free_count:
            mix = f"{guided_count} guided + {free_count} free-angle"
        elif guided_count:
            mix = f"{guided_count} guided"
        else:
            mix = f"{free_count} free-angle"
        step_note = ""
        if metadata["effective_steps"] != int(steps):
            step_note = " Steps were raised so every image is visited."
        info = (
            f"TripoSplat native multi-view · {len(entries)} views ({mix}) · "
            f"{metadata['method']} · {gaussian.get_xyz.shape[0]:,} gaussians · "
            f"generation: {time.time() - started:.1f}s · saved: {Path(download_path).name}.{step_note} "
            "TripoSplat generated and decoded this splat; no TRELLIS model weights were used."
        )
        gallery = [
            (image, label) for image, (label, _path, _role) in zip(prepared, entries)
        ]
        return (
            gallery,
            viewer,
            gr.update(value=download_path, interactive=True),
            info,
        )


with gr.Blocks(title="TripoSplat Native Multi-view") as demo:
    gr.Markdown("# TripoSplat Native Multi-view")
    gr.Markdown(
        "Both tabs use **TripoSplat** as the generative model. Multi-view adapts "
        "TRELLIS 1's tuning-free MultiDiffusion sampling strategy to TripoSplat's own latent "
        "and decoder; it does not load or generate with TRELLIS."
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Single image"):
                    image_in = gr.Image(
                        label="Input image", type="pil", image_mode="RGBA", height=320
                    )
                    gr.Examples(
                        examples=[[path] for path in EXAMPLES],
                        inputs=[image_in],
                        label="Examples (click to load)",
                        examples_per_page=10,
                        cache_examples=False,
                    )
                    with gr.Accordion("Single-image settings", open=False):
                        single_seed_in = gr.Number(label="Seed", value=42, precision=0)
                        single_steps_in = gr.Slider(
                            label="Inference steps", minimum=1, maximum=50, step=1, value=20
                        )
                        single_cfg_in = gr.Slider(
                            label="Guidance scale", minimum=1.0, maximum=10.0, step=0.5, value=3.0
                        )
                        single_num_g_in = gr.Dropdown(
                            label="Number of gaussians",
                            choices=["32768", "65536", "131072", "262144"],
                            value="262144",
                        )
                        single_fmt_in = gr.Dropdown(
                            label="Download format", choices=["ply", "splat"], value="ply"
                        )
                    single_btn = gr.Button("Generate from one image", variant="primary")

                with gr.Tab("Multi-view (TripoSplat)"):
                    gr.Markdown(
                        "Use guided slots, any number of free-angle images, or both. Side labels "
                        "can be approximate. TripoSplat infers a separate camera state for every "
                        "view while generating one shared object latent. If you use only free-angle "
                        "images, the first uploaded image sets the reference orientation."
                    )
                    with gr.Accordion("Guided side slots (optional)", open=True):
                        with gr.Row():
                            front_in = gr.Image(
                                label="Front-ish", type="filepath", image_mode="RGBA", height=190
                            )
                            back_in = gr.Image(
                                label="Back-ish", type="filepath", image_mode="RGBA", height=190
                            )
                        with gr.Row():
                            left_in = gr.Image(
                                label="Left-ish", type="filepath", image_mode="RGBA", height=190
                            )
                            right_in = gr.Image(
                                label="Right-ish", type="filepath", image_mode="RGBA", height=190
                            )
                    gr.Markdown("### Upload and order free-angle images")
                    free_in = gr.File(
                        label="Additional free-angle images - drag the file cards to reorder",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                        allow_reordering=True,
                        height=220,
                    )
                    free_order_out = gr.Gallery(
                        label="Generation order (thumbnail 1 is the free-only reference)",
                        columns=4,
                        rows=2,
                        height=240,
                        object_fit="contain",
                        allow_preview=True,
                        buttons=[],
                        interactive=False,
                        type="filepath",
                    )
                    gr.Markdown(
                        "Drag the file cards in the uploader to change the generation order. "
                        "The thumbnail strip mirrors that order. When no guided image is used, "
                        "thumbnail 1 establishes the reference orientation."
                    )
                    gr.Examples(
                        examples=[[files] for _label, files in MULTIVIEW_EXAMPLES],
                        inputs=[free_in],
                        label="Multi-view test sets from the TRELLIS repository — click one to load all 3 images",
                        example_labels=[label for label, _files in MULTIVIEW_EXAMPLES],
                        examples_per_page=7,
                        cache_examples=False,
                    )
                    gr.Markdown(
                        "The test images are copied unchanged from "
                        "[Microsoft TRELLIS](https://github.com/microsoft/TRELLIS/tree/main/assets/example_multi_image) "
                        "under its MIT license. Only the images and sampling idea are reused—not its model."
                    )
                    with gr.Accordion("TripoSplat multi-view settings", open=False):
                        fusion_in = gr.Radio(
                            label="Multi-view sampling method",
                            choices=[
                                "MultiDiffusion (recommended, strongest consistency)",
                                "Stochastic (faster)",
                            ],
                            value="MultiDiffusion (recommended, strongest consistency)",
                        )
                        detail_influence_in = gr.Slider(
                            label="Free-angle detail influence",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.45,
                            info="Used late in sampling; guided views remain structural anchors early on.",
                        )
                        multi_seed_in = gr.Number(label="Seed", value=42, precision=0)
                        multi_steps_in = gr.Slider(
                            label="Inference steps", minimum=1, maximum=50, step=1, value=20
                        )
                        multi_cfg_in = gr.Slider(
                            label="Guidance scale", minimum=1.0, maximum=10.0, step=0.5, value=3.0
                        )
                        multi_shift_in = gr.Slider(
                            label="Sampling schedule shift", minimum=1.0, maximum=5.0, step=0.25, value=3.0
                        )
                        multi_num_g_in = gr.Dropdown(
                            label="Number of gaussians",
                            choices=["32768", "65536", "131072", "262144"],
                            value="262144",
                        )
                        multi_fmt_in = gr.Dropdown(
                            label="Download format", choices=["ply", "splat"], value="ply"
                        )
                    multiview_btn = gr.Button(
                        "Generate with TripoSplat from all views", variant="primary"
                    )

            prepared_out = gr.Gallery(
                label="Images actually supplied to TripoSplat",
                columns=4,
                rows=2,
                height=300,
                object_fit="contain",
            )
            info_out = gr.Markdown()

        with gr.Column(scale=2):
            viewer_out = gr.HTML(value=PLACEHOLDER_HTML, label="Spark.js viewer")
            file_out = gr.DownloadButton(label="Download", value=None, interactive=False)

    common_outputs = [prepared_out, viewer_out, file_out, info_out]
    single_btn.click(
        fn=generate_single,
        inputs=[
            image_in,
            single_seed_in,
            single_steps_in,
            single_cfg_in,
            single_num_g_in,
            single_fmt_in,
        ],
        outputs=common_outputs,
    )
    free_in.change(
        fn=free_upload_thumbnails,
        inputs=[free_in],
        outputs=[free_order_out],
    )
    multiview_btn.click(
        fn=generate_multiview,
        inputs=[
            front_in,
            back_in,
            left_in,
            right_in,
            free_in,
            fusion_in,
            detail_influence_in,
            multi_seed_in,
            multi_steps_in,
            multi_cfg_in,
            multi_shift_in,
            multi_num_g_in,
            multi_fmt_in,
        ],
        outputs=common_outputs,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[
            str(VIEWER_HTML.parent),
            str(OUT_ROOT),
            str(EXAMPLES_DIR),
            str(MULTIVIEW_EXAMPLES_DIR),
        ],
    )
