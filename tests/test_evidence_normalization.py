import unittest

from vlmsearch.evidence_normalization import deduplicate_observations


class ObservationDeduplicationTests(unittest.TestCase):
    def test_repeated_observations_are_removed_and_logged(self):
        text = "The horse is visible at (400, 500).\nThere is a horse visible at (410, 505).\nThe horse is visible at (400, 500)."
        cleaned, report = deduplicate_observations(text)
        self.assertEqual(cleaned, "The horse is visible at (400, 500).")
        self.assertEqual(report["removed_count"], 2)
        self.assertEqual(report["input_spans"], 3)
        self.assertNotEqual(report["input_sha256"], report["output_sha256"])

    def test_counts_negations_and_colors_cannot_be_merged(self):
        examples = [
            "Two giraffes are at (100, 100).\nThree giraffes are at (100, 100).",
            "A red ball is at (100, 100).\nA blue ball is at (100, 100).",
            "The person is standing at (100, 100).\nThe person is not standing at (100, 100).",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(deduplicate_observations(text)[1]["removed_count"], 0)

    def test_different_objects_regions_and_point_associations_are_preserved(self):
        examples = [
            "A horse is at (100, 100).\nA horse is at (700, 700).",
            "A man chases a dog at (100, 100).\nA dog chases a man at (100, 100).",
            "The man is left of the dog.\nThe dog is left of the man.",
            "Dog (100, 100) near person (700, 700).\nDog (700, 700) near person (100, 100).",
            "Dog at (100, 100) near person.\nDog near person at (100, 100).",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(deduplicate_observations(text)[1]["removed_count"], 0)

    def test_new_localization_information_is_not_dropped(self):
        text = "A dog is visible.\nA dog is visible at (500, 500)."
        self.assertEqual(deduplicate_observations(text)[1]["removed_count"], 0)

    def test_numbers_in_claims_are_preserved_and_sentence_boundaries_are_stable(self):
        text = "There are two horses. There are three horses. There are two horses."
        cleaned, report = deduplicate_observations(text)
        self.assertEqual(cleaned, "There are two horses.\nThere are three horses.")
        self.assertEqual(report["removed_count"], 1)


if __name__ == "__main__":
    unittest.main()
