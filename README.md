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

Run the MLP with the same batch size, learning rate, epoch count, and hidden
width as the MNIST energy model:

```bash
uv run -m scripts.mlp.mnist --epochs 5
```

## CNN

| Model   | Framework | Dataset                                                                              | Comments                                                                        |
| ------- | --------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| LeNet   | PyTorch   | CIFAR-10                                                                             | • Trained and evaluated the model.<br>• Inspected its feature maps.             |
| AlexNet | PyTorch   | [Kaggle Hymenoptera Data](https://www.kaggle.com/datasets/ajayrana/hymenoptera-data) | • Fine-tuned an ImageNet-pretrained model.<br>• Demonstrated transfer learning. |

## Energy-based model

The MLX MNIST energy classifier uses the same ten-output network as the MLP,
interpreting each energy as the negative of a class logit. Inference selects the
label with the lowest energy. The training loss can be chosen independently:

| `--loss` | Per-example objective, where `y` is correct and `E_j` is class `j`'s energy |
| --- | --- |
| `nll` (default) | `E_y + log(sum_j exp(-E_j))` |
| `perceptron` | `E_y - min_j E_j` |
| `hinge` | `max(0, margin + E_y - min_{j != y} E_j)` |

These are the multiclass forms described in [A Tutorial on Energy-Based
Learning](https://yann.lecun.org/exdb/publis/pdf/lecun-06.pdf). The perceptron
loss enforces no margin, so tied energies can give zero loss. `--margin` controls
the hinge gap and defaults to 1.0.

```bash
uv run -m scripts.ebm.mnist --loss nll --epochs 5
uv run -m scripts.ebm.mnist --loss perceptron --epochs 5
uv run -m scripts.ebm.mnist --loss hinge --margin 1.0 --epochs 5
```

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

Chat with locally trained GPT-2 weights or an explicit Hugging Face variant:

```bash
uv run -m scripts.llm.chat --model gpt2 --checkpoint-path models/gpt2/shakespeare.safetensors --device gpu
uv run -m scripts.llm.chat --hugging-face gpt2-medium --device gpu
```

`--checkpoint-path` and `--hugging-face` are mutually exclusive. `--model`
selects the local checkpoint's architecture, defaulting to GPT-2 124M when
omitted. With `--hugging-face`, the model is inferred from the named variant;
an explicit `--model` must match it. Supported variants are `gpt2`,
`gpt2-medium`, `gpt2-large`, and `gpt2-xl`. Local weights must be native MLX
`.safetensors` or `.npz` files and retain their saved precision. Existing
`--model` commands keep their default pretrained source, including Qwen3.

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
training loss and throughput. Async evaluation is enabled by default, with at
most two training steps in flight; `--no-async-eval` selects synchronous
evaluation for comparison. Batch-size tuning uses the selected evaluation mode
and waits for all measured updates to finish before recording throughput.

Each epoch displays a tqdm progress bar with completed batches, processing rate,
elapsed time, and ETA. The display refreshes at most twice per second, and
`--no-progress` disables it independently of `--verbose`.

Add `--save-path models/gpt2/shakespeare.safetensors` to save the model weights
after training. Both `.safetensors` and `.npz` are supported, and parent
directories are created automatically. This saves model weights only, without
optimizer state. Omit the option to skip saving.

Use the chat CLI above for generation. Inspect GPT-2 parameters separately with:

```bash
uv run -m scripts.llm.gpt2.pretrained --checkpoint-path models/gpt2/shakespeare.safetensors
uv run -m scripts.llm.gpt2.pretrained --hugging-face gpt2 --explore-parameters
```

The inspection script prints weight shapes by default. `--explore-parameters`
plots positional embeddings; add `--print-state` to do both. For local weights
from a larger GPT-2 model, pass the matching `--model-type`.

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
