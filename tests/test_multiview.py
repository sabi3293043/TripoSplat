import unittest

import torch

from multiview import (
    MultiViewFlowEulerCfgSampler,
    build_view_weight_schedule,
    effective_multiview_steps,
    interpolate_view_weights,
)


class FakeFlowModel:
    def __init__(self):
        self.seen = []

    def __call__(self, x_t, _t, cond):
        value = float(cond["value"].item())
        self.seen.append(value)
        return {
            "latent": torch.full_like(x_t["latent"], value),
            "camera": torch.full_like(x_t["camera"], value),
        }


def condition(value):
    return {"value": torch.tensor([value], dtype=torch.float32)}


class MultiViewSamplerTests(unittest.TestCase):
    def test_free_only_uses_reference_early_and_equal_detail_late(self):
        early, late = build_view_weight_schedule(["free", "free", "free"])
        self.assertAlmostEqual(early[0], 0.65)
        self.assertAlmostEqual(early[1], 0.175)
        self.assertAlmostEqual(early[2], 0.175)
        self.assertEqual(late, [1 / 3, 1 / 3, 1 / 3])

    def test_guided_anchor_yields_to_free_detail_late(self):
        early, late = build_view_weight_schedule(
            ["guided", "guided", "free", "free"],
            anchor_index=0,
            detail_influence=0.45,
        )
        self.assertAlmostEqual(sum(early[:2]), 0.80)
        self.assertAlmostEqual(sum(early[2:]), 0.20)
        self.assertAlmostEqual(sum(late[:2]), 0.55)
        self.assertAlmostEqual(sum(late[2:]), 0.45)
        self.assertGreater(early[0], early[1])

    def test_weight_interpolation_reaches_both_endpoints(self):
        early = [0.8, 0.2]
        late = [0.4, 0.6]
        self.assertEqual(interpolate_view_weights(early, late, 1.0), early)
        self.assertEqual(interpolate_view_weights(early, late, 0.0), late)

    def test_multidiffusion_averages_object_and_tracks_separate_cameras(self):
        sampler = MultiViewFlowEulerCfgSampler()
        model = FakeFlowModel()
        noise = {
            "latent": torch.zeros(1, 1, 1),
            "camera": torch.zeros(1, 1, 1),
        }
        output = sampler.sample_multi_view(
            model,
            noise,
            [condition(2), condition(6)],
            condition(0),
            steps=1,
            shift=1.0,
            guidance_scale=1.0,
            mode="multidiffusion",
            early_weights=[0.75, 0.25],
            late_weights=[0.75, 0.25],
            camera_noises=[torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)],
        )
        self.assertTrue(torch.allclose(output["latent"], torch.tensor([[[-3.0]]])))
        self.assertTrue(torch.allclose(output["camera"], torch.tensor([[[-3.0]]])))
        self.assertEqual(model.seen, [2.0, 6.0])

    def test_multidiffusion_runs_only_one_unconditional_pass_per_step(self):
        sampler = MultiViewFlowEulerCfgSampler()
        model = FakeFlowModel()
        noise = {
            "latent": torch.zeros(1, 1, 1),
            "camera": torch.zeros(1, 1, 1),
        }
        sampler.sample_multi_view(
            model,
            noise,
            [condition(2), condition(6)],
            condition(0),
            steps=1,
            shift=1.0,
            guidance_scale=3.0,
            mode="multidiffusion",
            camera_noises=[torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)],
        )
        self.assertEqual(model.seen, [2.0, 6.0, 0.0])

    def test_stochastic_mode_visits_every_view(self):
        sampler = MultiViewFlowEulerCfgSampler()
        model = FakeFlowModel()
        noise = {
            "latent": torch.zeros(1, 1, 1),
            "camera": torch.zeros(1, 1, 1),
        }
        totals = []
        sampler.sample_multi_view(
            model,
            noise,
            [condition(1), condition(2), condition(3)],
            condition(0),
            steps=1,
            shift=1.0,
            guidance_scale=1.0,
            mode="stochastic",
            callback=lambda step, total: totals.append((step, total)),
        )
        self.assertEqual(effective_multiview_steps(1, 3, "stochastic"), 3)
        self.assertEqual(model.seen, [1.0, 2.0, 3.0])
        self.assertEqual(totals[-1], (3, 3))


if __name__ == "__main__":
    unittest.main()
