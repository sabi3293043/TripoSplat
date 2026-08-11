# Usage

## Input modes

### Single image

Use the Single image tab for the original TripoSplat workflow. Upload one image,
choose the output size, and generate.

### Free-angle multi-view

Upload any number of images in any camera order. Uploaded images appear on a
sortable thumbnail board. Drag the thumbnail cards to reorder them. For devices
or browsers where dragging is unavailable, choose a thumbnail in the selection
list and use **Move first**, **Move earlier**, or **Move later**. The board is the
canonical order sent to TripoSplat, so it no longer depends on the uploader's
internal file order. The first image establishes the early reference orientation
when no guided view is used; it does not need to be a front view. Later in
sampling, all images contribute missing surface detail.

For the most stable result:

- show the same object in every image;
- use sharp, unobstructed views with similar scale and framing;
- keep major shape, clothing, accessories, and colors consistent;
- remove images with severe perspective distortion or unrelated objects;
- start with two to four useful views, then add only images that reveal new
  information.

### Guided sides

Front-ish, Back-ish, Left-ish, and Right-ish are approximate organizational
slots. Images do not need to be perfectly orthographic. The labels select a
safe scheduling anchor only; they are not written into TripoSplat's learned
camera token.

### Guided plus free-angle

You can fill any guided slots and also upload arbitrary free-angle images. Both
sets condition the same shared TripoSplat latent. Guided images dominate early
structure, while free-angle influence grows later to recover details.

## Recommended settings

| Setting | Starting value | Notes |
|---|---:|---|
| Sampling method | MultiDiffusion | Best consistency; cost grows with view count. |
| Free-angle detail influence | 0.45 | Lower if free views distort the main shape; raise carefully for missing detail. |
| Steps | 20 | Use 10-20 for previews and 20-30 for a final attempt. |
| Guidance | 3.0 | Higher is not always better. |
| Shift | 3.0 | Default used by the tested implementation. |
| Gaussians | 32,768 preview / 262,144 final | Higher counts use more memory and create larger files. |
| Format | PLY | Broad compatibility; SPLAT is smaller for compatible viewers. |

Stochastic mode evaluates one view per step and is faster. If the requested
step count is smaller than the view count, it is automatically raised so every
image is visited at least once.

## Outputs

Each run creates a downloadable `.ply` or `.splat` file and previews it in the
embedded Spark viewer. Generated files are written under `gradio_outputs/`,
which is intentionally excluded from Git.

Compatible external viewers include [SparkJS](https://sparkjs.dev) and
[SuperSplat](https://superspl.at/editor).

## Python integration

`multiview.run_multi_image` accepts an initialized `TripoSplatPipeline`, a list
of images, optional `guided`/`free` roles, and normal TripoSplat sampling
settings. It returns the decoded Gaussian object, preprocessed images, and
provenance metadata.

```python
from PIL import Image

from multiview import run_multi_image
from triposplat import TripoSplatPipeline

pipe = TripoSplatPipeline(
    ckpt_path="ckpts/diffusion_models/triposplat_fp16.safetensors",
    decoder_path="ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
    dinov3_path="ckpts/clip_vision/dino_v3_vit_h.safetensors",
    flux2_vae_encoder_path="ckpts/vae/flux2-vae.safetensors",
    rmbg_path="ckpts/background_removal/birefnet.safetensors",
    device="cuda",
)

images = [Image.open("angle_1.png"), Image.open("angle_2.png")]
gaussian, prepared, metadata = run_multi_image(
    pipe,
    images,
    roles=["free", "free"],
    mode="multidiffusion",
    seed=42,
    steps=20,
    guidance_scale=3.0,
    shift=3.0,
    num_gaussians=262144,
)
gaussian.save_ply("result.ply")
print(metadata)
```
