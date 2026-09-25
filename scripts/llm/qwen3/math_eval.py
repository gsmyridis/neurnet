import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import torch
from neurnet.arch.llm.generate import generate_text_stream_concat

from neurnet.arch.llm.gym.math import grade_answer
from neurnet.arch.llm.gym.math.parse import extract_final_candidate
from neurnet.arch.llm.models.qwen.torch import Qwen3Tokenizer, Qwen3TorchModel
from neurnet.arch.llm.models.qwen.utils import load_model_and_tokenizer
from neurnet.utils.gpu import get_torch_device
from neurnet.utils.progress import eta_progress_message


def render_prompt(prompt: str) -> str:
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template


class MathExample(TypedDict):
    problem: str
    answer: str


def mini_eval_demo(
    model: Qwen3TorchModel,
    tokenizer: Qwen3Tokenizer,
    device: str | torch.device,
) -> None:
    ex: MathExample = {  # Test example with "problem" and "answer" fields
        "problem": "Compute 1/2 + 1/6.",
        "answer": "2/3",
    }
    prompt = render_prompt(ex["problem"])  # 1. Apply prompt template
    gen_text = generate_text_stream_concat(  # 2. Generate response
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=64,
    )
    pred_answer = extract_final_candidate(gen_text)  # 3. Extract and normalize answer
    is_correct = grade_answer(  # 4. Grade answer
        pred_answer, ex["answer"]
    )
    print(f"Device: {device}")
    print(f"Prediction: {pred_answer}")
    print(f"Ground truth: {ex['answer']}")
    print(f"Correct: {is_correct}")


def evaluate_math500_stream(
    model: Qwen3TorchModel,
    tokenizer: Qwen3Tokenizer,
    device: str | torch.device,
    math_data: Sequence[MathExample],
    out_path: str | Path | None = None,
    max_new_tokens: int = 512,
    verbose: bool = False,
) -> tuple[int, int, float]:

    if out_path is None:
        dev_name = str(device).replace(
            ":", "-"
        )  # Make filename compatible with Windows
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    total_len = 0  # Calculates the average response length (see exercise 3.2)
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:  # Save results for inspection
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])  # 1. Apply prompt template
            gen_text = generate_text_stream_concat(  # 2. Generate response
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            total_len += len(tokenizer.encode(gen_text))

            extracted = extract_final_candidate(  # 3. Extract and normalize answer
                gen_text
            )
            is_correct = grade_answer(  # 4. Grade answer
                extracted, row["answer"]
            )
            num_correct += int(is_correct)

            record = {  # Record to be saved for inspection
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:  # Print responses during the generation process
                print(
                    f"\n\n{'=' * 50}\n{progress_msg}\n"
                    f"{'=' * 50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-' * 50}"
                )

    # Print summary information
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc * 100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed / 60:.1f} min")
    avg_len = total_len / num_examples
    print(f"Average response length: {avg_len:.2f} tokens")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc


def main() -> None:
    qwen_dir = os.path.join("models", "qwen3")
    device = get_torch_device()
    model, tokenizer = load_model_and_tokenizer("base", device, False, qwen_dir)

    prompt = (
        r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
        r"what is the value of $a^2+b^2$?"
    )

    answer = generate_text_stream_concat(model, tokenizer, prompt, device, 2048, False)
    print(extract_final_candidate(answer))


if __name__ == "__main__":
    main()
