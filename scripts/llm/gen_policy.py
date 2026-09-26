from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from neurnet.arch.llm.inference import (
    GenerationConfig,
    GenerationPolicy,
    GenerationResult,
    InferenceRuntime,
)
from neurnet.device import Device, DeviceType
from scripts.llm.models import MODEL_CHOICES, load_model_runtime

DEVICE_CHOICES = [device_type.value for device_type in DeviceType]
DEFAULT_MAX_NEW_TOKENS = 64


@dataclass(frozen=True)
class NamedPolicy:
    name: str
    policy: GenerationPolicy


@dataclass(frozen=True)
class GenerationMetrics:
    token_count: int
    unique_token_ratio: float
    repeated_ngram_count: int
    tokens_per_second: float


@dataclass(frozen=True)
class Trial:
    prompt: str
    policy: NamedPolicy
    seed: int
    result: GenerationResult
    metrics: GenerationMetrics


DEFAULT_POLICIES: tuple[NamedPolicy, ...] = (
    NamedPolicy(
        "greedy",
        GenerationPolicy(
            temperature=0,
            repetition_penalty=1,
            max_consecutive_repeats=0,
            repeat_ngram_size=0,
        ),
    ),
    NamedPolicy(
        "greedy-penalty-1.05",
        GenerationPolicy(
            temperature=0,
            repetition_penalty=1.05,
            max_consecutive_repeats=0,
            repeat_ngram_size=0,
        ),
    ),
    NamedPolicy(
        "sample-k50-p90",
        GenerationPolicy(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1,
            max_consecutive_repeats=0,
            repeat_ngram_size=0,
        ),
    ),
    NamedPolicy(
        "sample-k50-p90-penalty-1.05",
        GenerationPolicy(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.05,
            max_consecutive_repeats=0,
            repeat_ngram_size=0,
        ),
    ),
    NamedPolicy(
        "sample-k50-p90-penalty-1.05-guards",
        GenerationPolicy(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.05,
            max_consecutive_repeats=3,
            repeat_ngram_size=6,
        ),
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compare generation policies using the same prompts and seeds.",
        epilog=(
            "Each --prompt is sent to the selected model verbatim. Repeat --prompt "
            "to test multiple prompts. With no --policy arguments, the built-in "
            "policy matrix is used.\n\n"
            "Custom policy example:\n"
            '  --policy \'{"name": "custom", "temperature": 0.7, '
            '"top_k": 50, "top_p": 0.9, "repetition_penalty": 1.05, '
            '"max_consecutive_repeats": 3, "repeat_ngram_size": 6}\''
        ),
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        required=True,
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        required=True,
        help="Prompt to complete; may be supplied multiple times.",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default=DeviceType.CPU.value,
        help="Requested device. 'gpu' selects the backend's available GPU.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum completion length for every trial.",
    )
    parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=0,
        help="First seed to evaluate.",
    )
    parser.add_argument(
        "--num-seeds",
        type=_positive_int,
        default=5,
        help="Number of consecutive seeds to evaluate per prompt and policy.",
    )
    parser.add_argument(
        "--metric-ngram-size",
        type=_ngram_size,
        default=3,
        help="N-gram size used to measure emitted repetition.",
    )
    parser.add_argument(
        "--policy",
        dest="policies",
        action="append",
        type=_parse_policy,
        default=[],
        metavar="JSON",
        help="Named GenerationPolicy as a JSON object; may be supplied multiple times.",
    )
    parser.add_argument(
        "--show-completions",
        action="store_true",
        help="Print the completion for every trial after the summary.",
    )
    parser.add_argument("--compile", action="store_true", help="Compile Torch models.")
    parser.add_argument(
        "--reasoning", action="store_true", help="Use Qwen reasoning model variant."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    policies = tuple(args.policies) or DEFAULT_POLICIES
    _ensure_unique_policy_names(policies)

    device = Device(DeviceType.from_string(args.device))
    runtime = load_model_runtime(
        args.model,
        device,
        compile=args.compile,
        reasoning=args.reasoning,
    )
    trials = run_experiment(
        runtime,
        prompts=args.prompts,
        policies=policies,
        max_new_tokens=args.max_new_tokens,
        seeds=range(args.seed, args.seed + args.num_seeds),
        metric_ngram_size=args.metric_ngram_size,
    )
    print_summary(trials, metric_ngram_size=args.metric_ngram_size)
    if args.show_completions:
        print_completions(trials)


def run_experiment(
    runtime: InferenceRuntime,
    *,
    prompts: Sequence[str],
    policies: Sequence[NamedPolicy],
    max_new_tokens: int,
    seeds: Sequence[int],
    metric_ngram_size: int,
) -> list[Trial]:
    """Generate every prompt-policy-seed combination through one runtime."""
    trials: list[Trial] = []
    for prompt in prompts:
        for named_policy in policies:
            config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                policy=named_policy.policy,
            )
            for seed in seeds:
                runtime.set_seed(seed)
                result = runtime.generate(prompt, config)
                trials.append(
                    Trial(
                        prompt=prompt,
                        policy=named_policy,
                        seed=seed,
                        result=result,
                        metrics=measure_generation(result, metric_ngram_size),
                    )
                )
    return trials


def measure_generation(
    result: GenerationResult, ngram_size: int = 3
) -> GenerationMetrics:
    if ngram_size < 2:
        raise ValueError("ngram_size must be at least two")

    token_ids = result.token_ids
    token_count = len(token_ids)
    unique_token_ratio = len(set(token_ids)) / token_count if token_count else 0.0
    return GenerationMetrics(
        token_count=token_count,
        unique_token_ratio=unique_token_ratio,
        repeated_ngram_count=_repeated_ngram_count(token_ids, ngram_size),
        tokens_per_second=(
            token_count / result.elapsed_seconds if result.elapsed_seconds > 0 else 0.0
        ),
    )


def print_summary(trials: Sequence[Trial], *, metric_ngram_size: int) -> None:
    for prompt in dict.fromkeys(trial.prompt for trial in trials):
        prompt_trials = [trial for trial in trials if trial.prompt == prompt]
        print(f"\nPrompt: {prompt!r}")
        print(
            "policy                                     runs  tokens  "
            f"repeated-{metric_ngram_size}grams  repeat-runs  unique  tok/s"
        )
        for policy in dict.fromkeys(trial.policy for trial in prompt_trials):
            policy_trials = [trial for trial in prompt_trials if trial.policy == policy]
            metrics = [trial.metrics for trial in policy_trials]
            repeated_runs = sum(metric.repeated_ngram_count > 0 for metric in metrics)
            print(
                f"{policy.name:40.40} "
                f"{len(metrics):>4} "
                f"{mean(metric.token_count for metric in metrics):>7.1f} "
                f"{mean(metric.repeated_ngram_count for metric in metrics):>16.2f} "
                f"{repeated_runs:>11}/{len(metrics):<3} "
                f"{mean(metric.unique_token_ratio for metric in metrics):>6.3f} "
                f"{mean(metric.tokens_per_second for metric in metrics):>5.1f}"
            )


def print_completions(trials: Sequence[Trial]) -> None:
    for trial in trials:
        print(f"\n[{trial.policy.name}; seed={trial.seed}; prompt={trial.prompt!r}]")
        print(trial.result.text)


def _parse_policy(value: str) -> NamedPolicy:
    try:
        data: Any = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("--policy must be valid JSON") from error

    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("--policy must be a JSON object")

    name = data.pop("name", None)
    if not isinstance(name, str) or not name:
        raise argparse.ArgumentTypeError("--policy requires a non-empty 'name'")

    try:
        policy = GenerationPolicy(**data)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"invalid --policy: {error}") from error
    return NamedPolicy(name, policy)


def _positive_int(value: str) -> int:
    parsed = _non_negative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _ngram_size(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("n-gram size must be at least two")
    return parsed


def _ensure_unique_policy_names(policies: Sequence[NamedPolicy]) -> None:
    names = [policy.name for policy in policies]
    if len(names) != len(set(names)):
        raise ValueError("policy names must be unique")


def _repeated_ngram_count(token_ids: Sequence[int], ngram_size: int) -> int:
    ngrams = [
        tuple(token_ids[index : index + ngram_size])
        for index in range(len(token_ids) - ngram_size + 1)
    ]
    return len(ngrams) - len(set(ngrams))


if __name__ == "__main__":
    main()
