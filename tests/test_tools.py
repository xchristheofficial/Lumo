"""Tests for the tool layer: safe calculator eval, memory, knowledge search."""
import json
import tempfile
import unittest
from pathlib import Path

from lumo import tools
from lumo.corpus import PASSAGES


class TestCalculator(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(tools.tool_calculator("17 * 23"), "391")
        self.assertEqual(tools.tool_calculator("12 + 34"), "46")
        self.assertEqual(tools.tool_calculator("144 / 12"), "12")
        self.assertEqual(tools.tool_calculator("2 + 3 * 4"), "14")  # precedence
        self.assertEqual(tools.tool_calculator("sqrt(144)"), "12")
        self.assertEqual(tools.tool_calculator("17 x 23"), "391")

    def test_conversions_via_expr(self):
        self.assertEqual(tools.tool_calculator("100 * 9 / 5 + 32"), "212")

    def test_rejects_unsafe_input(self):
        for bad in ["__import__('os')", "open('/etc/passwd')",
                    "().__class__", "1/0", "lambda: 1", "import os", ""]:
            res = tools.tool_calculator(bad)
            self.assertTrue(res.startswith("error"),
                            f"expected error for {bad!r}, got {res!r}")

    def test_too_large_result(self):
        self.assertTrue(tools.tool_calculator("10 ** 60").startswith("error"))


class TestMemory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = tools.MEMORY_PATH
        tools.MEMORY_PATH = Path(self._tmp.name) / "memory.json"

    def tearDown(self):
        tools.MEMORY_PATH = self._orig
        self._tmp.cleanup()

    def test_remember_recall_roundtrip(self):
        self.assertEqual(tools.tool_remember("favorite color = teal"),
                         "stored favorite color = teal")
        self.assertEqual(tools.tool_recall("favorite color"), "teal")

    def test_fuzzy_recall(self):
        tools.tool_remember("favorite city = mombasa")
        self.assertEqual(tools.tool_recall("my favorite city"), "mombasa")

    def test_missing_key(self):
        res = tools.tool_recall("nonexistent key")
        self.assertIn("no memory", res)

    def test_bad_remember(self):
        self.assertTrue(tools.tool_remember("no equals sign").startswith("error"))

    def test_persists_to_disk(self):
        tools.tool_remember("a = b")
        data = json.loads(tools.MEMORY_PATH.read_text())
        self.assertEqual(data["a"], "b")


class TestKnowledge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tools.init_knowledge(PASSAGES)

    def test_finds_relevant_passage(self):
        text, score = tools._KB.search("black holes gravity")
        self.assertGreater(score, 0.15)
        self.assertIn("black hole", text.lower())

    def test_plural_stemming(self):
        self.assertIn("giraffe", tools.tool_search("giraffes").lower())
        self.assertIn("giraffe", tools.tool_search("giraffe").lower())

    def test_capital_lookup(self):
        self.assertIn("paris", tools.tool_search("capital of france").lower())

    def test_irrelevant_query_flagged(self):
        res = tools.tool_search("xylophone quantum banana split")
        # either a weak match or the explicit no-result string — never garbage
        self.assertTrue(res == "no relevant passage found" or len(res) > 20)


class TestTime(unittest.TestCase):
    def test_format(self):
        import re

        res = tools.tool_time("now")
        self.assertRegex(res, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} utc \( \w+ \)$")


if __name__ == "__main__":
    unittest.main()
