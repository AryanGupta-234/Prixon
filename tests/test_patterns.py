import json
import tempfile
import unittest

from cognition.patterns import PatternMemory


class PatternMemoryTests(unittest.TestCase):
    def test_repeated_actions_become_context(self):
        with tempfile.TemporaryDirectory() as d:
            path = d + "/patterns.json"
            memory = PatternMemory(path)
            for _ in range(3):
                memory.observe_action("Spotify.exe", "open_app", True)
            ctx = memory.context()
            self.assertTrue(ctx["habits"])
            self.assertEqual(ctx["habits"][0]["target"], "spotify.exe")
            self.assertEqual(ctx["habits"][0]["success_rate"], 1.0)

    def test_failed_actions_do_not_become_positive_habit(self):
        with tempfile.TemporaryDirectory() as d:
            path = d + "/patterns.json"
            memory = PatternMemory(path)
            for _ in range(3):
                memory.observe_action("bad.exe", "open_app", False)
            ctx = memory.context()
            self.assertEqual(ctx["habits"][0]["success_rate"], 0.0)

    def test_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as d:
            path = d + "/patterns.json"
            first = PatternMemory(path)
            first.observe_preference("response_style", "concise")
            first.observe_preference("response_style", "concise")
            second = PatternMemory(path)
            self.assertIn("response_style=concise", second.context()["stable_preferences"])


if __name__ == "__main__":
    unittest.main()
