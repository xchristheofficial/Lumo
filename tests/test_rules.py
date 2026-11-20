"""Tests for the deterministic rule planner's intent routing."""
import unittest

from lumo.engine import rule_plan
from lumo.tools import init_knowledge
from lumo.corpus import PASSAGES

init_knowledge(PASSAGES)


class TestRulePlanner(unittest.TestCase):
    def test_math_intents(self):
        cases = {
            "what is 17 * 23 ?": ("calculator", "17 * 23"),
            "calculate 9 * 8": ("calculator", "9 * 8"),
            "what is 12 plus 34 ?": ("calculator", "12 + 34"),
            "how much is 1000 / 8 ?": ("calculator", "1000 / 8"),
            "what is 15 percent of 200 ?": ("calculator", "200 * (15 / 100)"),
        }
        for msg, expected in cases.items():
            p = rule_plan(msg)
            self.assertEqual(p.actions, [expected], msg)

    def test_conversion_intents(self):
        cases = {
            "convert 5 km to miles": ("5 * 0.621371", "5", "km", "miles"),
            "how many miles in 10 km ?": ("10 * 0.621371", "10", "km", "miles"),
            "12 pounds to kg": ("12 * 0.453592", "12", "pounds", "kg"),
            "convert 100 celsius to fahrenheit": ("100 * 9 / 5 + 32", "100",
                                                  "celsius", "fahrenheit"),
        }
        for msg, (expr, v, u1, u2) in cases.items():
            p = rule_plan(msg)
            self.assertEqual(p.actions, [("calculator", expr)], msg)
            self.assertEqual(p.conv, (v, u1, u2), msg)

    def test_memory_intents(self):
        p = rule_plan("remember that my favorite color is teal")
        self.assertEqual(p.actions, [("remember", "favorite color = teal")])
        p = rule_plan("what is my favorite color ?")
        self.assertEqual(p.actions, [("recall", "favorite color")])

    def test_time_intent(self):
        for msg in ["what time is it ?", "what is today's date ?",
                    "what day is it today ?"]:
            p = rule_plan(msg)
            self.assertEqual(p.actions, [("time", "now")], msg)

    def test_knowledge_intent(self):
        p = rule_plan("tell me about black holes")
        self.assertEqual(len(p.actions), 1)
        self.assertEqual(p.actions[0][0], "search")

    def test_small_talk_is_direct_not_fallback(self):
        for msg in ["hello", "thanks", "who are you ?", "what can you do ?",
                    "goodbye", "how are you ?"]:
            p = rule_plan(msg)
            self.assertIsNotNone(p.direct_answer, msg)
            self.assertFalse(p.fallback, msg)

    def test_unknown_is_flagged_fallback(self):
        p = rule_plan("xyzzy plugh frobnicate")
        self.assertIsNotNone(p.direct_answer)
        self.assertTrue(p.fallback)

    def test_no_false_math_on_plain_text(self):
        p = rule_plan("tell me about giraffes")
        self.assertEqual(p.actions[0][0], "search")
        p = rule_plan("who was leonardo da vinci")
        self.assertEqual(p.actions[0][0], "search")


if __name__ == "__main__":
    unittest.main()
