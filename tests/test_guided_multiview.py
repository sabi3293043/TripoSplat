import unittest

from multiview import build_view_weight_schedule


class GuidedSchedulingTests(unittest.TestCase):
    def test_anchor_is_selected_only_from_guided_views(self):
        early, _late = build_view_weight_schedule(
            ["free", "guided", "guided"], anchor_index=0
        )
        self.assertGreater(early[1], early[2])

    def test_camera_token_is_not_assigned_undocumented_xyz_directions(self):
        from pathlib import Path

        source = (Path(__file__).parents[1] / "multiview.py").read_text(encoding="utf-8")
        self.assertNotIn("GUIDED_CAMERA_DIRECTIONS", source)
        self.assertNotIn("apply_camera_direction_prior", source)

    def test_invalid_roles_are_rejected(self):
        with self.assertRaises(ValueError):
            build_view_weight_schedule(["guided", "sideways"])


if __name__ == "__main__":
    unittest.main()
