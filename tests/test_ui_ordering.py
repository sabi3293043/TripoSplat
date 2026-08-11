import unittest

from ui_ordering import apply_index_order, move_selected, sync_upload_order


class UploadOrderingTests(unittest.TestCase):
    def test_initial_upload_uses_uploader_order(self):
        order, known = sync_upload_order([], [], ["a.png", "b.png", "c.png"])
        self.assertEqual(order, ["a.png", "b.png", "c.png"])
        self.assertEqual(known, order)

    def test_custom_order_survives_new_upload(self):
        order, known = sync_upload_order(
            ["c.png", "a.png", "b.png"],
            ["a.png", "b.png", "c.png"],
            ["a.png", "b.png", "c.png", "d.png"],
        )
        self.assertEqual(order, ["c.png", "a.png", "b.png", "d.png"])
        self.assertEqual(known, ["a.png", "b.png", "c.png", "d.png"])

    def test_removed_upload_leaves_order_without_that_card(self):
        order, _known = sync_upload_order(
            ["c.png", "a.png", "b.png"],
            ["a.png", "b.png", "c.png"],
            ["a.png", "c.png"],
        )
        self.assertEqual(order, ["c.png", "a.png"])

    def test_native_uploader_reorder_is_honored(self):
        order, known = sync_upload_order(
            ["a.png", "b.png", "c.png"],
            ["a.png", "b.png", "c.png"],
            ["c.png", "a.png", "b.png"],
        )
        self.assertEqual(order, ["c.png", "a.png", "b.png"])
        self.assertEqual(known, order)

    def test_browser_drag_permutation_is_applied(self):
        self.assertEqual(
            apply_index_order(["a.png", "b.png", "c.png"], [2, 0, 1]),
            ["c.png", "a.png", "b.png"],
        )

    def test_invalid_browser_permutation_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_index_order(["a.png", "b.png"], [0, 0])

    def test_button_move_changes_canonical_order(self):
        order, selected, _status = move_selected(
            ["a.png", "b.png", "c.png"], "2", "first"
        )
        self.assertEqual(order, ["c.png", "a.png", "b.png"])
        self.assertEqual(selected, 0)


if __name__ == "__main__":
    unittest.main()
