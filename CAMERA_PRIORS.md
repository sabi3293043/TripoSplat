# Guided-view behavior

Front-ish, Right-ish, Back-ish, and Left-ish are organizational hints. They do
not need to be exact orthographic views.

The previous experiment assumed the first three values of TripoSplat's learned
five-channel camera token were an XYZ direction and pushed them toward fixed
side vectors. The checkpoint does not document that representation, so the
assumption could corrupt camera denoising and has been removed.

The corrected method lets TripoSplat infer one camera state per image. A guided
front image, when supplied, is weighted more strongly only during the early
structure steps. All guided views and free-angle images remain part of the same
shared-latent generation.
