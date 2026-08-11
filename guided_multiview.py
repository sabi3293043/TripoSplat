"""Compatibility exports for the corrected TripoSplat multi-view sampler.

The former camera-direction prior was removed because TripoSplat does not
document its five camera-token channels as an XYZ direction. Guided behavior is
now implemented through safe reference-frame scheduling in :mod:`multiview`.
"""

from multiview import (
    MULTIVIEW_MODES,
    MultiViewFlowEulerCfgSampler,
    build_view_weight_schedule,
    effective_multiview_steps,
    interpolate_view_weights,
    run_multi_image,
    sample_latent_multi_view,
)


__all__ = [
    "MULTIVIEW_MODES",
    "MultiViewFlowEulerCfgSampler",
    "build_view_weight_schedule",
    "effective_multiview_steps",
    "interpolate_view_weights",
    "run_multi_image",
    "sample_latent_multi_view",
]
