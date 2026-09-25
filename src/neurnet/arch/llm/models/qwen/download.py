from pathlib import Path
from typing import Literal

from neurnet.arch.llm.utils import download_file


def download_qwen3_small(
    kind: Literal["base", "reasoning"],
    out_dir: str | Path,
    *,
    tokenizer_only: bool = False,
) -> None:
    files = {
        "base": {"model": "qwen3-0.6B-base.pth", "tokenizer": "tokenizer-base.json"},
        "reasoning": {
            "model": "qwen3-0.6B-reasoning.pth",
            "tokenizer": "tokenizer-reasoning.json",
        },
    }
    if kind not in files:
        raise ValueError("kind must be 'base' or 'reasoning'")

    repo = "rasbt/qwen3-from-scratch"
    hf_fmt = "https://huggingface.co/{repo}/resolve/main/{file}"
    backup_root = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"
    targets = ["tokenizer"] if tokenizer_only else ["model", "tokenizer"]

    for key in targets:
        fname = files[kind][key]
        primary = hf_fmt.format(repo=repo, file=fname)
        backup = f"{backup_root}/{fname}"
        download_file(primary, out_dir=out_dir, backup_url=backup)


def download_qwen3_grpo_checkpoints(
    grpo_type: Literal[
        "no_kl", "tracking", "clip_ratio", "kl", "format_reward"
    ] = "no_kl",
    step: str | int = "00050",
    out_dir: str = ".",
) -> Path:
    mapper = {
        "no_kl": "grpo_original_no_kl",
        "tracking": "7_3_plus_tracking/checkpoints",
        "clip_ratio": "7_4_plus_clip_ratio/checkpoints",
        "kl": "7_5_plus_kl/checkpoints",
        "format_reward": "7_6_plus_format_reward/checkpoints",
    }
    if grpo_type not in mapper:
        raise ValueError(f"only grpo_type in {mapper.keys()} are supported for now")

    repo = "rasbt/qwen3-from-scratch-grpo-checkpoints"
    step = str(step)
    if step.isdigit():
        step = step.zfill(5)
    fname = f"qwen3-0.6B-rlvr-grpo-step{step}.pth"
    primary = f"https://huggingface.co/{repo}/resolve/main/{mapper[grpo_type]}/{fname}"

    backup = None
    if grpo_type == "no_kl" and step == "00050":
        backup_root = (
            "https://f001.backblazeb2.com/file/"
            "reasoning-from-scratch/qwen3-0.6B-checkpoints"
        )
        fname = "grpo_original_no_kl/qwen3-0.6B-rlvr-grpo-step00050.pth"
        backup = f"{backup_root}/{fname}"

    return download_file(primary, out_dir=out_dir, backup_url=backup)


def download_qwen3_distill_checkpoints(
    distill_type: Literal["deepseek_r1", "qwen3_235b_a22b"] = "deepseek_r1",
    step: str | int = "06682",
    out_dir: str = ".",
) -> Path:
    mapper = {
        "deepseek_r1": {
            "06682": "qwen3-0.6B-distill-step06682-epoch1.pth",
            "13364": "qwen3-0.6B-distill-step13364-epoch2.pth",
            "20046": "qwen3-0.6B-distill-step20046-epoch3.pth",
        },
        "qwen3_235b_a22b": {
            "05746": "qwen3-0.6B-distill-step05746-epoch1.pth",
            "11492": "qwen3-0.6B-distill-step11492-epoch2.pth",
            "17238": "qwen3-0.6B-distill-step17238-epoch3.pth",
        },
    }
    folder_map = {
        "deepseek_r1": "ch08_distill_deepseek_r1/checkpoints",
        "qwen3_235b_a22b": "ch08_distill_qwen3_235b_a22b/checkpoints",
    }
    if distill_type not in mapper:
        raise ValueError(f"only distill_type in {mapper.keys()} are supported for now")

    step = str(step)
    if step.isdigit():
        step = step.zfill(5)
    if step not in mapper[distill_type]:
        raise ValueError(
            f"only step in {mapper[distill_type].keys()} are supported for {distill_type}"
        )

    repo = "rasbt/qwen3-from-scratch-distill-checkpoints"
    fname = mapper[distill_type][step]
    primary = (
        f"https://huggingface.co/{repo}/resolve/main/{folder_map[distill_type]}/{fname}"
    )
    return download_file(primary, out_dir=out_dir)
