import unittest

from smartrupee.engine import _final_verdict


class FinalVerdictTests(unittest.TestCase):
    def test_experimental_counterfeit_label_does_not_override_reference_match(self):
        model_result = {
            "label": "Consistent with genuine references",
            "authenticity_classifier": {
                "label": "POSSIBLE COUNTERFEIT"
            }
        }

        verdict = _final_verdict(
            "Good",
            True,
            True,
            model_result,
            5,
            0
        )

        self.assertEqual(verdict[0], "LIKELY GENUINE")

    def test_experimental_genuine_label_does_not_override_inconclusive_model(self):
        model_result = {
            "label": "Inconclusive",
            "authenticity_classifier": {
                "label": "LIKELY GENUINE"
            }
        }

        verdict = _final_verdict(
            "Good",
            True,
            True,
            model_result,
            7,
            0
        )

        self.assertEqual(verdict[0], "INCONCLUSIVE")

    def test_visual_mismatch_prevents_genuine_verdict(self):
        model_result = {
            "label": "Consistent with genuine references"
        }

        verdict = _final_verdict(
            "Good",
            True,
            True,
            model_result,
            7,
            1
        )

        self.assertEqual(verdict[0], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
