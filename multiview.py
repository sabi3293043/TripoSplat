"""TRELLIS-style tuning-free multi-image sampling for TripoSplat.

This module uses only TripoSplat's image encoders, flow model, latent space,
decoder, and Gaussian representation.  It adapts TRELLIS 1's MultiDiffusion
idea by evaluating the shared noisy object latent once per input view at every
step, then averaging those object-velocity predictions.

TripoSplat additionally denoises a learned camera token.  MultiDiffusion keeps
one camera state per image so arbitrary viewpoints do not get collapsed into a
single contradictory camera.  Guided labels influence safe scheduling only;
no undocumented meaning is assigned to the five camera-token channels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from triposplat import FlowEulerCfgSampler


MULTIVIEW_MODES = {"stochastic", "multidiffusion"}


def effective_multiview_steps(steps: int, view_count: int, mode: str) -> int:
    """Ensure stochastic sampling visits every supplied image at least once."""
    if view_count < 1:
        raise ValueError("view_count must be at least 1")
    mode = str(mode).lower()
    if mode not in MULTIVIEW_MODES:
        raise ValueError("mode must be 'stochastic' or 'multidiffusion'")
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    return max(steps, view_count) if mode == "stochastic" else steps


def _normalized(weights: Sequence[float]) -> list[float]:
    values = [max(0.0, float(value)) for value in weights]
    total = sum(values)
    if total <= 0:
        raise ValueError("at least one view weight must be greater than zero")
    return [value / total for value in values]


def build_view_weight_schedule(
    roles: Sequence[str],
    anchor_index: int | None = None,
    detail_influence: float = 0.45,
) -> tuple[list[float], list[float]]:
    """Build early-structure and late-detail MultiDiffusion weights.

    Free-only uploads use the exact equal averaging from TRELLIS.  When guided
    slots are present, the first available guided view (normally Front-ish)
    anchors the noisy early structure without pretending that TripoSplat's
    undocumented camera-token channels are literal XYZ directions.  Free-angle
    views gain their requested detail influence as denoising approaches zero.
    """
    roles = [str(role).lower() for role in roles]
    if not roles or any(role not in {"guided", "free"} for role in roles):
        raise ValueError("roles must contain one 'guided' or 'free' value per view")
    guided = [index for index, role in enumerate(roles) if role == "guided"]
    free = [index for index, role in enumerate(roles) if role == "free"]
    count = len(roles)
    early = [0.0] * count
    late = [0.0] * count

    if not guided:
        equal = [1.0 / count] * count
        if count == 1:
            return equal, equal.copy()
        # TripoSplat's object latent remains partly view-relative even though it
        # has a learned camera token. Pick the first arbitrary view as a
        # reference frame early, then converge to equal averaging for detail.
        # That first image may be at any angle; it defines orientation, not a
        # semantic "front".
        if anchor_index is None or not 0 <= int(anchor_index) < count:
            anchor_index = 0
        early[anchor_index] = 0.65
        for index in range(count):
            if index != anchor_index:
                early[index] = 0.35 / (count - 1)
        return _normalized(early), equal.copy()

    if anchor_index not in guided:
        anchor_index = guided[0]

    if not free:
        for index in guided:
            late[index] = 1.0 / len(guided)
        if len(guided) == 1:
            early[anchor_index] = 1.0
        else:
            early[anchor_index] = 0.60
            remainder = 0.40 / (len(guided) - 1)
            for index in guided:
                if index != anchor_index:
                    early[index] = remainder
        return _normalized(early), _normalized(late)

    detail_influence = min(1.0, max(0.0, float(detail_influence)))
    early_guided_total = 0.80
    early_free_total = 0.20
    if len(guided) == 1:
        early[anchor_index] = early_guided_total
    else:
        early[anchor_index] = early_guided_total * 0.65
        remainder = early_guided_total * 0.35 / (len(guided) - 1)
        for index in guided:
            if index != anchor_index:
                early[index] = remainder
    for index in free:
        early[index] = early_free_total / len(free)

    guided_total = 1.0 - detail_influence
    for index in guided:
        late[index] = guided_total / len(guided)
    for index in free:
        late[index] = detail_influence / len(free)
    return _normalized(early), _normalized(late)


def interpolate_view_weights(
    early_weights: Sequence[float], late_weights: Sequence[float], timestep: float
) -> list[float]:
    """Smoothly transition from structure weights at t=1 to detail at t=0."""
    early = _normalized(early_weights)
    late = _normalized(late_weights)
    if len(early) != len(late):
        raise ValueError("early and late weights must have the same length")
    t = min(1.0, max(0.0, float(timestep)))
    blend = t * t * (3.0 - 2.0 * t)
    return _normalized([
        blend * early_value + (1.0 - blend) * late_value
        for early_value, late_value in zip(early, late)
    ])


class MultiViewFlowEulerCfgSampler(FlowEulerCfgSampler):
    """TripoSplat Euler sampler with TRELLIS-style multi-image prediction fusion."""

    @staticmethod
    def _stochastic_view_schedule(steps: int, weights: Sequence[float]) -> list[int]:
        count = len(weights)
        schedule = list(range(min(count, steps)))
        used = [0] * count
        for index in schedule:
            used[index] += 1
        for step in range(len(schedule), steps):
            target = step + 1
            index = max(
                range(count),
                key=lambda candidate: target * weights[candidate] - used[candidate],
            )
            schedule.append(index)
            used[index] += 1
        return schedule

    @staticmethod
    def _guidance_value(guidance_scale, key: str) -> float:
        if isinstance(guidance_scale, dict):
            return float(guidance_scale.get(key, 1.0))
        if guidance_scale is None:
            return 1.0
        return float(guidance_scale)

    @torch.no_grad()
    def sample_multi_view(
        self,
        model,
        noise,
        conds,
        neg_cond,
        *,
        steps=50,
        shift=1.0,
        guidance_scale=None,
        mode="multidiffusion",
        early_weights=None,
        late_weights=None,
        camera_noises=None,
        show_progress=False,
        callback=None,
    ):
        if not conds:
            raise ValueError("at least one image condition is required")
        mode = str(mode).lower()
        if mode not in MULTIVIEW_MODES:
            raise ValueError("mode must be 'stochastic' or 'multidiffusion'")
        count = len(conds)
        equal = [1.0 / count] * count
        early_weights = _normalized(early_weights or equal)
        late_weights = _normalized(late_weights or equal)
        if len(early_weights) != count or len(late_weights) != count:
            raise ValueError("view weights must contain one value per condition")

        steps = effective_multiview_steps(steps, count, mode)
        t_values = np.linspace(1, 0, steps + 1)
        t_seq = shift * t_values / (1 + (shift - 1) * t_values)
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))
        iterator = tqdm(t_pairs, desc="TripoSplat multi-view", total=steps) if show_progress else t_pairs

        if mode == "stochastic":
            sample = {key: value.clone() for key, value in noise.items()}
            schedule_weights = _normalized([
                (early + late) * 0.5 for early, late in zip(early_weights, late_weights)
            ])
            schedule = self._stochastic_view_schedule(steps, schedule_weights)
            for step_index, (t, t_prev) in enumerate(iterator):
                view_index = schedule[step_index]
                x_t = {key: value.clone() for key, value in sample.items()}
                pred_v = self._cfg_prediction(
                    model, x_t, t, conds[view_index], neg_cond, guidance_scale
                )
                dt = t - t_prev
                for key in sample:
                    sample[key] = sample[key] - pred_v[key] * dt
                if callback is not None:
                    callback(step_index + 1, steps)
            return sample

        shared_latent = noise["latent"].clone()
        has_camera = "camera" in noise
        if has_camera:
            if camera_noises is None:
                camera_states = [noise["camera"].clone() for _ in range(count)]
            elif len(camera_noises) != count:
                raise ValueError("camera_noises must contain one tensor per condition")
            else:
                camera_states = [camera.clone() for camera in camera_noises]

        for step_index, (t, t_prev) in enumerate(iterator):
            weights = interpolate_view_weights(early_weights, late_weights, t)
            conditional_predictions = []
            for view_index, cond in enumerate(conds):
                x_t = {"latent": shared_latent.clone()}
                if has_camera:
                    x_t["camera"] = camera_states[view_index].clone()
                conditional_predictions.append(self._inference_model(model, x_t, t, cond))

            averaged_latent = torch.zeros_like(conditional_predictions[0]["latent"])
            for weight, prediction in zip(weights, conditional_predictions):
                averaged_latent.add_(prediction["latent"], alpha=weight)

            latent_guidance = self._guidance_value(guidance_scale, "latent")
            if latent_guidance > 1.0:
                negative_state = {"latent": shared_latent.clone()}
                if has_camera:
                    blended_camera = torch.zeros_like(camera_states[0])
                    for weight, camera in zip(weights, camera_states):
                        blended_camera.add_(camera, alpha=weight)
                    negative_state["camera"] = blended_camera
                negative_prediction = self._inference_model(
                    model, negative_state, t, neg_cond
                )["latent"]
                averaged_latent = (
                    latent_guidance * averaged_latent
                    - (latent_guidance - 1.0) * negative_prediction
                )

            dt = t - t_prev
            shared_latent = shared_latent - averaged_latent * dt
            if has_camera:
                # Each image denoises its own camera token from its own condition.
                # This is the Tripo-specific extension to TRELLIS MultiDiffusion.
                camera_states = [
                    camera - prediction["camera"] * dt
                    for camera, prediction in zip(camera_states, conditional_predictions)
                ]
            if callback is not None:
                callback(step_index + 1, steps)

        result = {"latent": shared_latent}
        if has_camera:
            final_weights = interpolate_view_weights(early_weights, late_weights, 0.0)
            result["camera"] = torch.zeros_like(camera_states[0])
            for weight, camera in zip(final_weights, camera_states):
                result["camera"].add_(camera, alpha=weight)
        return result


@torch.no_grad()
def sample_latent_multi_view(
    flow_model,
    conds,
    *,
    steps: int = 50,
    guidance_scale: float = 7.0,
    shift: float = 3.0,
    mode: str = "multidiffusion",
    early_weights=None,
    late_weights=None,
    generator: torch.Generator | None = None,
    show_progress: bool = False,
    callback=None,
) -> dict:
    conds = list(conds)
    if not conds:
        raise ValueError("at least one image condition is required")
    neg_cond = {key: torch.zeros_like(value) for key, value in conds[0].items()}
    device = flow_model.device
    noise = {
        "latent": torch.randn(
            1,
            flow_model.q_token_length,
            flow_model.in_channels,
            device=device,
            generator=generator,
        )
    }
    camera_noises = None
    if flow_model.cam_channels is not None:
        noise["camera"] = torch.randn(
            1, 1, flow_model.cam_channels, device=device, generator=generator
        )
        if str(mode).lower() == "multidiffusion":
            camera_noises = [noise["camera"].clone()]
            camera_noises.extend(
                torch.randn(
                    1, 1, flow_model.cam_channels, device=device, generator=generator
                )
                for _ in range(len(conds) - 1)
            )

    sampler = MultiViewFlowEulerCfgSampler()
    return sampler.sample_multi_view(
        flow_model,
        noise,
        conds,
        neg_cond,
        steps=steps,
        guidance_scale=guidance_scale,
        shift=shift,
        mode=mode,
        early_weights=early_weights,
        late_weights=late_weights,
        camera_noises=camera_noises,
        show_progress=show_progress,
        callback=callback,
    )


@torch.no_grad()
def run_multi_image(
    pipeline,
    images,
    *,
    roles=None,
    anchor_index: int | None = None,
    detail_influence: float = 0.45,
    seed: int = 42,
    steps: int = 20,
    guidance_scale: float = 3.0,
    shift: float = 3.0,
    num_gaussians=262144,
    erode_radius: int = 1,
    mode: str = "multidiffusion",
    show_progress: bool = False,
    callback=None,
):
    """Generate one TripoSplat object from an arbitrary number of images."""
    images = list(images)
    if not images:
        raise ValueError("at least one image is required")
    roles = list(roles or ["free"] * len(images))
    if len(roles) != len(images):
        raise ValueError("roles must contain one value per image")
    if isinstance(num_gaussians, (list, tuple)):
        counts = [pipeline._validate_num_gaussians(value) for value in num_gaussians]
    else:
        counts = [pipeline._validate_num_gaussians(num_gaussians)]

    prepared = [
        pipeline.preprocess_image(image, erode_radius=erode_radius)
        for image in images
    ]
    # Use the same VAE posterior-noise realization for every view.  This avoids
    # injecting view-index-dependent randomness into cross-view differences.
    conds = []
    for image in prepared:
        encode_generator = torch.Generator(device=pipeline._device).manual_seed(int(seed))
        conds.append(pipeline.encode_image(image, generator=encode_generator))

    early_weights, late_weights = build_view_weight_schedule(
        roles, anchor_index=anchor_index, detail_influence=detail_influence
    )
    sampling_generator = torch.Generator(device=pipeline._device).manual_seed(int(seed))
    mode = str(mode).lower()
    if len(conds) == 1:
        output = pipeline.sample_latent(
            conds[0],
            steps=steps,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=sampling_generator,
            show_progress=show_progress,
            callback=callback,
        )
    else:
        output = sample_latent_multi_view(
            pipeline.flow_model,
            conds,
            steps=steps,
            guidance_scale=guidance_scale,
            shift=shift,
            mode=mode,
            early_weights=early_weights,
            late_weights=late_weights,
            generator=sampling_generator,
            show_progress=show_progress,
            callback=callback,
        )
    gaussians = [pipeline.decode_latent(output["latent"], num_gaussians=count) for count in counts]
    metadata = {
        "generator": "TripoSplat",
        "method": "TRELLIS-style MultiDiffusion" if mode == "multidiffusion" else "TRELLIS-style stochastic",
        "mode": mode,
        "effective_steps": effective_multiview_steps(steps, len(images), mode),
        "early_weights": early_weights,
        "late_weights": late_weights,
    }
    if isinstance(num_gaussians, (list, tuple)):
        return gaussians, prepared, metadata
    return gaussians[0], prepared, metadata
