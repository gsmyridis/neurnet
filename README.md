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

| Model | Framework | Dataset | Comments  |
| ----- | --------- | ------- | --------- |
| GPT-2 | MLX       | —       | Inference |
| Qwen3 | PyTorch   | —       | Inference |

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
