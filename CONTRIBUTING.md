# Contributing

Bug reports and improvements are welcome.

## Before opening an issue

- Confirm the same object appears in every input image.
- Retry with the bundled Character example.
- Record GPU model, VRAM, operating system, Python, PyTorch, CUDA, view count,
  sampling mode, step count, Gaussian count, and the complete error message.
- Do not upload private input images without permission.

## Development checks

From an activated environment:

```bash
python -m py_compile multiview.py guided_multiview.py run_gradio.py
python -m unittest discover -s tests -p "test_*.py"
```

Tests must not require model weights or a GPU unless explicitly marked as an
integration test. Keep TRELLIS runtime dependencies out of the active path.

## Design constraints

- TripoSplat must remain the model and Gaussian decoder.
- Multi-view inputs must share one object latent.
- Each input may retain its own learned TripoSplat camera state.
- Do not assign undocumented geometric meaning to camera-token channels.
- Guided and free-angle inputs must remain composable.
- Generated outputs, checkpoints, environments, and caches must not be
  committed.
