import io
import sys
import unittest
from contextlib import redirect_stderr
from unittest import mock

from neurnet.arch.llm.inference import GenerationResult
from scripts.llm import gen_policy


class GenerationPolicyScriptTests(unittest.TestCase):
    def test_parse_arguments_accepts_multiple_prompts_and_custom_policy(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "gen_policy.py",
                "--model",
                "gpt2",
                "--prompt",
                "First prompt",
                "--prompt",
                "Second prompt",
                "--policy",
                '{"name": "greedy", "temperature": 0, "repetition_penalty": 1}',
            ],
        ):
            args = gen_policy.parse_arguments()

        self.assertEqual(args.prompts, ["First prompt", "Second prompt"])
        self.assertEqual(len(args.policies), 1)
        self.assertEqual(args.policies[0].name, "greedy")
        self.assertEqual(args.policies[0].policy.temperature, 0)

    def test_measure_generation_counts_repeated_ngrams(self) -> None:
        metrics = gen_policy.measure_generation(
            GenerationResult("", (1, 2, 1, 2, 1), 0.5),
            ngram_size=2,
        )

        self.assertEqual(metrics.token_count, 5)
        self.assertEqual(metrics.repeated_ngram_count, 2)
        self.assertEqual(metrics.unique_token_ratio, 0.4)
        self.assertEqual(metrics.tokens_per_second, 10.0)

    def test_custom_policy_requires_a_name(self) -> None:
        with (
            self.assertRaisesRegex(SystemExit, "2"),
            redirect_stderr(io.StringIO()),
            mock.patch.object(
                sys,
                "argv",
                [
                    "gen_policy.py",
                    "--model",
                    "gpt2",
                    "--prompt",
                    "Prompt",
                    "--policy",
                    '{"temperature": 0}',
                ],
            ),
        ):
            gen_policy.parse_arguments()


if __name__ == "__main__":
    unittest.main()
