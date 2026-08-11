# TripoSplat-native multi-view

Multi-view generation uses TripoSplat for every learned component: image
conditioning, flow denoising, latent representation, Gaussian decoding, and
PLY/SPLAT export. TRELLIS model weights and TRELLIS's 3D decoder are not used.

The adaptation follows TRELLIS 1's tuning-free multi-image strategy. At every
flow step, the same noisy TripoSplat object latent is evaluated once for each
uploaded image. The predicted object velocities are blended and one shared
latent is updated. Stochastic mode instead cycles through the images.

TripoSplat also generates a five-channel camera token. Its channel semantics
are not documented, so guided labels are never written into that token. Each
image receives its own learned camera state, allowing approximate and arbitrary
viewpoints to align from image content. A guided front view acts only as a safe
early structural anchor; free-angle images gain influence later for detail.

When only free-angle images are supplied, the first uploaded image defines the
early reference frame regardless of its actual angle. The weights transition
to an equal TRELLIS-style average late in sampling, so every other image still
adds missing surface detail.

The number of free-angle images is not hard-limited by the UI. Runtime grows
roughly with image count in MultiDiffusion mode, and the practical limit is GPU
memory and patience.
