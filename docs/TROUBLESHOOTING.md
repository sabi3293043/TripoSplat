# Troubleshooting

## The result contains duplicate or mangled bodies

1. Confirm every image shows the same unchanged object.
2. Put the clearest, most representative image first in free-angle-only mode,
   or place it in Front-ish.
3. Use MultiDiffusion rather than Stochastic.
4. Remove the weakest or most distorted image and retry.
5. If guided and free images are mixed, reduce free-angle detail influence from
   `0.45` toward `0.25`.
6. Match crop, scale, background, and object position as closely as practical.

The guided labels can be approximate. Do not rotate or mirror an image merely
to force it into a slot.

## Details from later views are missing

- Increase free-angle detail influence gradually, for example from `0.45` to
  `0.55`.
- Add a sharp view that clearly exposes the missing surface.
- Use 20-30 steps for the final run.
- Check that the extra view is not heavily occluded or inconsistent with the
  structural view.

## CUDA out of memory

- Preview with 32,768 or 65,536 Gaussians.
- Use fewer input views.
- Close other GPU-heavy applications.
- Restart the app after an out-of-memory failure to clear model allocations.
- Avoid running multiple generations concurrently; the UI serializes requests
  intentionally.

## The app reports missing checkpoint files

Ensure the complete Hugging Face repository was downloaded into `ckpts/` and
that the file layout matches the tree in the installation guide. Re-run the
Pinokio **Update** action or the manual `huggingface-cli download` command.

## PyTorch says CUDA is unavailable

Run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the second value is `False`, reinstall a CUDA-enabled PyTorch wheel matched
to the installed driver. Use the official PyTorch selector rather than a random
wheel URL.

## The 3D viewer is blank but a file was generated

- Download the PLY and open it in SparkJS or SuperSplat.
- Enable WebGL/hardware acceleration in the browser.
- Try PLY output before SPLAT output.
- Refresh the page after the generation has completed.

## The raw PLY appears rotated in another viewer

Viewers use different world-up conventions. The bundled viewer applies the
orientation used by TripoSplat. Rotate the scene or change the viewer's up axis;
this is separate from multiview alignment.

## Generation is slow

MultiDiffusion performs one conditional TripoSplat prediction per image at each
step, plus a shared unconditional prediction when guidance is greater than one.
Runtime therefore increases roughly linearly with view count. Use Stochastic
mode or fewer steps for previews.

## Is TRELLIS generating the output?

No. The UI status reports the generator provenance after every run. TRELLIS is
credited for the sampling pattern and bundled example images only; no TRELLIS
model service or weights are required.
