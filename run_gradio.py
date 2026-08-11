"""TripoSplat UI with TRELLIS-style native multi-image sampling."""

from __future__ import annotations

import html
import json
import threading
import time
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import gradio as gr
import torch

from multiview import run_multi_image
from triposplat import TripoSplatPipeline
from ui_ordering import (
    apply_index_order,
    move_selected,
    normalize_uploads,
    selection_choices,
    sync_upload_order,
    upload_path,
)


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

FREE_SORT_CSS = """
#free-sorter .free-sort-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(132px, 1fr));
  gap: 10px;
  min-height: 90px;
}
#free-sorter .free-sort-card {
  position: relative;
  overflow: hidden;
  border: 2px solid var(--border-color-primary);
  border-radius: 10px;
  background: var(--background-fill-secondary);
  cursor: grab;
  user-select: none;
  transition: border-color 120ms ease, opacity 120ms ease, transform 120ms ease;
}
#free-sorter .free-sort-card:hover { border-color: var(--color-accent); }
#free-sorter .free-sort-card.free-dragging { opacity: 0.45; transform: scale(0.98); }
#free-sorter .free-sort-card img {
  display: block;
  width: 100%;
  height: 118px;
  object-fit: contain;
  pointer-events: none;
  background: #111318;
}
#free-sorter .free-sort-caption {
  padding: 7px 9px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
}
#free-sorter .free-sort-number {
  position: absolute;
  top: 6px;
  left: 6px;
  min-width: 25px;
  padding: 3px 6px;
  border-radius: 999px;
  color: white;
  background: rgba(0, 0, 0, 0.78);
  font-weight: 700;
  text-align: center;
}
#free-sorter .free-sort-empty {
  display: flex;
  min-height: 90px;
  align-items: center;
  justify-content: center;
  border: 1px dashed var(--border-color-primary);
  border-radius: 10px;
  color: var(--body-text-color-subdued);
}
"""

FREE_SORT_JS = r"""() => {
  const inputFor = (id) => {
    const root = document.getElementById(id);
    if (!root) return null;
    if (root.matches("textarea,input")) return root;
    return root.querySelector("textarea,input");
  };
  const buttonFor = (id) => {
    const root = document.getElementById(id);
    if (!root) return null;
    if (root.matches("button")) return root;
    return root.querySelector("button");
  };
  const setFrameworkValue = (input, value) => {
    const proto = input instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    setter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const commit = (list) => {
    const indices = Array.from(list.querySelectorAll(".free-sort-card"))
      .map((card) => Number(card.dataset.orderIndex));
    const input = inputFor("free-order-json");
    const button = buttonFor("free-order-apply");
    if (!input || !button) return;
    setFrameworkValue(input, JSON.stringify(indices));
    window.setTimeout(() => button.click(), 40);
  };
  const bind = () => {
    document.querySelectorAll("#free-sorter .free-sort-list").forEach((list) => {
      if (list.dataset.sortBound === "1") return;
      list.dataset.sortBound = "1";
      let dragged = null;
      list.addEventListener("dragstart", (event) => {
        const card = event.target.closest(".free-sort-card");
        if (!card) return;
        dragged = card;
        card.classList.add("free-dragging");
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", card.dataset.orderIndex);
      });
      list.addEventListener("dragover", (event) => {
        if (!dragged) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        const target = event.target.closest(".free-sort-card");
        if (!target || target === dragged) return;
        const box = target.getBoundingClientRect();
        const verticalMove = Math.abs(event.clientY - (box.top + box.height / 2)) > box.height / 4;
        const after = verticalMove
          ? event.clientY > box.top + box.height / 2
          : event.clientX > box.left + box.width / 2;
        list.insertBefore(dragged, after ? target.nextSibling : target);
      });
      list.addEventListener("drop", (event) => {
        if (!dragged) return;
        event.preventDefault();
        dragged.classList.remove("free-dragging");
        dragged = null;
        commit(list);
      });
      list.addEventListener("dragend", () => {
        if (dragged) dragged.classList.remove("free-dragging");
        dragged = null;
      });
    });
  };
  bind();
  new MutationObserver(bind).observe(document.body, { childList: true, subtree: true });
} """

FREE_SORT_HEAD = f"""<script>
(() => {{
  const installFreeSorter = {FREE_SORT_JS};
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", installFreeSorter, {{ once: true }});
  }} else {{
    installFreeSorter();
  }}
}})();
</script>"""


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



def collect_multiview_entries(front, back, left, right, free_uploads):
    """Return guided views in cyclic order followed by arbitrary-angle views."""
    guided_candidates = [
        ("Front-ish", front),
        ("Right-ish", right),
        ("Back-ish", back),
        ("Left-ish", left),
    ]
    entries = [
        (label, upload_path(image), "guided")
        for label, image in guided_candidates
        if image is not None
    ]
    entries.extend(
        (f"Free angle {index + 1}", upload_path(upload), "free")
        for index, upload in enumerate(list(free_uploads or []))
        if upload is not None
    )
    return entries


def free_order_html(order):
    """Render the canonical generation order as draggable image cards."""
    paths = normalize_uploads(order)
    if not paths:
        return '<div class="free-sort-empty">Upload images to create the ordering board.</div>'
    cards = []
    for index, path in enumerate(paths):
        url = f"/gradio_api/file={quote(Path(path).as_posix(), safe='/:')}"
        name = html.escape(Path(path).name)
        reference = " (reference)" if index == 0 else ""
        cards.append(
            '<div class="free-sort-card" draggable="true" '
            f'data-order-index="{index}" title="Drag to reorder {name}">'
            f'<span class="free-sort-number">{index + 1}</span>'
            f'<img src="{html.escape(url, quote=True)}" alt="{name}">'
            f'<div class="free-sort-caption">{name}{reference}</div></div>'
        )
    return '<div class="free-sort-list" role="list">' + "".join(cards) + "</div>"


def _order_choice_update(order, selected=None):
    choices = selection_choices(order)
    value = None if selected is None else str(selected)
    return gr.update(choices=choices, value=value)


def sync_free_uploads(current_order, known_uploads, free_uploads):
    order, known = sync_upload_order(current_order, known_uploads, free_uploads)
    status = (
        f"{len(order)} image(s). Drag the thumbnail cards, or use the move buttons below."
        if order
        else "No free-angle images uploaded yet."
    )
    return order, known, free_order_html(order), _order_choice_update(order), "", status


def apply_free_angle_order(current_order, order_json):
    try:
        indices = json.loads(order_json or "[]")
        order = apply_index_order(current_order, indices)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise gr.Error(str(error)) from error
    return (
        order,
        free_order_html(order),
        _order_choice_update(order),
        "",
        "Thumbnail order updated. This exact order will be sent to TripoSplat.",
    )


def move_free_angle_order(current_order, selected, action):
    try:
        order, new_index, status = move_selected(current_order, selected, action)
    except (TypeError, ValueError) as error:
        raise gr.Error(str(error)) from error
    return order, free_order_html(order), _order_choice_update(order, new_index), "", status


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
                        label="Additional free-angle images",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                        height=220,
                    )
                    free_order_state = gr.State([])
                    free_known_uploads_state = gr.State([])
                    free_order_out = gr.HTML(
                        value=free_order_html([]),
                        label="Generation order",
                        elem_id="free-sorter",
                    )
                    free_order_select = gr.Dropdown(
                        label="Select an image for the move buttons",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    with gr.Row():
                        free_first_btn = gr.Button("Move first")
                        free_earlier_btn = gr.Button("Move earlier")
                        free_later_btn = gr.Button("Move later")
                    free_order_status = gr.Markdown("No free-angle images uploaded yet.")
                    free_order_json = gr.Textbox(
                        value="", visible="hidden", elem_id="free-order-json"
                    )
                    free_order_apply = gr.Button(
                        "Apply thumbnail order", visible="hidden", elem_id="free-order-apply"
                    )
                    gr.Markdown(
                        "Drag the **thumbnail cards** above to reorder them. If dragging is not "
                        "available on your device, select an image and use the move buttons. "
                        "Card 1 establishes the reference orientation when no guided image is used."
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
        fn=sync_free_uploads,
        inputs=[free_order_state, free_known_uploads_state, free_in],
        outputs=[
            free_order_state,
            free_known_uploads_state,
            free_order_out,
            free_order_select,
            free_order_json,
            free_order_status,
        ],
        api_name="sync_free_angle_uploads",
    )
    free_order_apply.click(
        fn=apply_free_angle_order,
        inputs=[free_order_state, free_order_json],
        outputs=[
            free_order_state,
            free_order_out,
            free_order_select,
            free_order_json,
            free_order_status,
        ],
        queue=False,
        api_name="apply_free_angle_order",
    )
    for button, action in (
        (free_first_btn, "first"),
        (free_earlier_btn, "earlier"),
        (free_later_btn, "later"),
    ):
        button.click(
            fn=lambda order, selected, action=action: move_free_angle_order(
                order, selected, action
            ),
            inputs=[free_order_state, free_order_select],
            outputs=[
                free_order_state,
                free_order_out,
                free_order_select,
                free_order_json,
                free_order_status,
            ],
            queue=False,
            api_name=f"move_free_angle_{action}",
        )
    multiview_btn.click(
        fn=generate_multiview,
        inputs=[
            front_in,
            back_in,
            left_in,
            right_in,
            free_order_state,
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
        css=FREE_SORT_CSS,
        head=FREE_SORT_HEAD,
        allowed_paths=[
            str(VIEWER_HTML.parent),
            str(OUT_ROOT),
            str(EXAMPLES_DIR),
            str(MULTIVIEW_EXAMPLES_DIR),
        ],
    )
