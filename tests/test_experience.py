import os
import tempfile
import unittest

from cognition.experience import ExperienceModel


class ExperienceModelTests(unittest.TestCase):
    def test_success_rate_and_sequences_persist(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "learned.json")
            model = ExperienceModel(path)
            model.observe("task_completed", "open_app", "Spotify", True)
            model.observe("task_completed", "diagnostic", "Spotify", True)
            model.observe("task_failed", "close_app_dynamic", "Spotify", False)

            ctx = model.context()
            spotify = next(x for x in ctx["reliable_actions"] if x["target"] == "spotify")
            self.assertEqual(spotify["successes"], 2)
            self.assertEqual(spotify["failures"], 1)
            self.assertGreater(spotify["success_rate"], 0.6)
            self.assertTrue(any(x["from"] == "spotify" for x in ctx["common_action_sequences"]))

            restored = ExperienceModel(path)
            self.assertEqual(restored.successes["spotify"], 2)
            self.assertEqual(restored.failures["spotify"], 1)


if __name__ == "__main__":
    unittest.main()
