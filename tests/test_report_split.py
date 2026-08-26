import unittest

from scripts.build_report_split import (
    GroupStats,
    assign_groups,
    derived_parent_stem,
    scene_group_key,
)


class ReportSplitTests(unittest.TestCase):
    def test_pan_crops_share_original_scene(self):
        a = "01-PAN-20240420-113-325-L00000010882-CCD3_5_crop1"
        b = "01-PAN-20240420-113-325-L00000010882-CCD4_2_crop9"
        self.assertEqual(scene_group_key(a), scene_group_key(b))

    def test_coordinate_crops_share_l1_scene(self):
        a = "E120.2_N14.7_20190309_L1A0003875932-PAN10_crop1"
        b = "E120.2_N14.7_20190309_L1A0003875932-PAN17_crop4"
        self.assertEqual(scene_group_key(a), scene_group_key(b))

    def test_fsc_map_sources_share_location(self):
        a = "fsc_AGZ-N23.44-E120.40-lv20-Bing_crop0001"
        b = "fsc_AGZ-N23.44-E120.40-lv20-Google_crop0003"
        self.assertEqual(scene_group_key(a), scene_group_key(b))

    def test_mar20_uses_contiguous_blocks(self):
        self.assertEqual(scene_group_key("MAR20_1", mar20_block_size=20), "mar20:000000")
        self.assertEqual(scene_group_key("MAR20_20", mar20_block_size=20), "mar20:000000")
        self.assertEqual(scene_group_key("MAR20_21", mar20_block_size=20), "mar20:000001")

    def test_augmented_fsc_tracks_its_source(self):
        self.assertEqual(
            derived_parent_stem("fsc_aug2_fsc_TG-N22.33-E120.62-lv20-Google_crop00002"),
            "fsc_TG-N22.33-E120.62-lv20-Google_crop00002",
        )

    def test_assignment_keeps_groups_atomic_and_validation_at_least_30_percent(self):
        groups = []
        for idx in range(60):
            counts = [0, 0, 0]
            counts[idx % 3] = 3
            groups.append(
                GroupStats(
                    key=f"g{idx}",
                    sample_indices=(idx,),
                    image_count=2,
                    class_counts=tuple(counts),
                )
            )

        assignment = assign_groups(groups, ratios=(0.60, 0.10, 0.30), seed=2026, restarts=32)
        split_images = {name: 0 for name in ("train", "calibration", "validation")}
        for group in groups:
            split_images[assignment[group.key]] += group.image_count

        self.assertEqual(set(assignment), {group.key for group in groups})
        self.assertGreaterEqual(split_images["validation"] / sum(split_images.values()), 0.30)
        self.assertLessEqual(abs(split_images["calibration"] / sum(split_images.values()) - 0.10), 0.05)


if __name__ == "__main__":
    unittest.main()
