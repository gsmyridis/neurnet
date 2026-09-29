# `neurnet`

Experiments with different neural network architectures, and deep learning frameworks.
The different frameworks that I explored are:

| Frameworks |
| ---------- |
| PyTorch    |
| MLX        |

The experiments that have been carried out are grouped by architecture.

## MLP

| Model          | Framework | Dataset | Comments                                               |
| -------------- | --------- | ------- | ------------------------------------------------------ |
| MLP Classifier | MLX       | MNIST   | Trained and evaluated a simple multi-layer perceptron. |

## CNN

| Model   | Framework | Dataset                                                                              | Comments                                                                        |
| ------- | --------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| LeNet   | PyTorch   | CIFAR-10                                                                             | • Trained and evaluated the model.<br>• Inspected its feature maps.             |
| AlexNet | PyTorch   | [Kaggle Hymenoptera Data](https://www.kaggle.com/datasets/ajayrana/hymenoptera-data) | • Fine-tuned an ImageNet-pretrained model.<br>• Demonstrated transfer learning. |

## LLM

| Model | Framework | Dataset          | Comments               |
| ----- | --------- | ---------------- | ---------------------- |
| GPT-2 | MLX       | Tiny Shakespeare | Training and inference |
| Qwen3 | PyTorch   | —                | Inference              |

### Generation policies

The generation policies explored are:

- Greedy decoding.
- Temperature sampling.
- Top-k sampling.
- Nucleus (top-p) sampling.
- Repetition penalties.
- Repetition guards.

### REPL inference

Run interactive LLM inference with:

```bash
uv run -m scripts.llm.chat
```

Append `--help` to view the available models and generation options.

### GPT-2 training

```bash
uv run -m scripts.llm.gpt2.train --dtype mixed --verbose
```

The training script compiles each optimizer step and tunes batch size using
warm, synchronized token throughput. `--batch-size N` skips tuning, and
`--no-compile` provides an eager comparison. `--dtype float32` is the default
precision baseline. `--dtype mixed` computes with bfloat16 parameters while
keeping float32 master weights and AdamW state; `--dtype bfloat16` uses bfloat16
for both parameters and optimizer state and should be checked for convergence.
The loss is evaluated in float32 in all modes. `--verbose` reports each epoch's
training loss and throughput.

GPT-2 and Qwen3 inference use backend-specific, fixed-capacity KV caches.
Prefill runs once; single-token decoding updates the allocated cache without
growing its shape. `--compile` compiles the stable-shape decoder (not prefill),
avoiding a new graph for every decode position. Cache capacity is rounded up to
256-token buckets, capped by the model context length. Qwen3 uses bfloat16
weights and cache entries by default while its normalization and RoPE
calculations retain float32 precision. Cached generation requires the model to
provide a fixed-cache implementation; there is no growing-cache fallback.

## Development

Run the tests with `uv run python -m unittest discover -s tests`. After
installing [prek](https://prek.j178.dev/), run `uv sync` and `prek install` to
enable the commit hooks. The hooks run the tests, Ruff lint and formatting, and
standard file checks. Run them manually with `prek run --all-files`.
