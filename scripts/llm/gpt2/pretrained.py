from copy import deepcopy

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline, set_seed

# from neurnet.arch.llm.models.gpt2 import GPT2_CONFIG_124M, GPT2ModelType
# from neurnet.arch.llm.models.gpt2.mlx import GPT2


def print_gpt2_state():
    # Load the pretrained 124M parameter model from HuggingFace
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="models")
    state = gpt2.state_dict()

    for key, value in state.items():
        print(key, value.shape)


def explore_pretrained_parameters():
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="models")
    state = gpt2.state_dict()

    # Explore the weights of positional encoding weights
    plt.imshow(state["transformer.wpe.weight"])
    plt.show()

    # In later architectures the weights of positional encodings
    # are fixed to sinusoidal with different frequences, but in GPT2
    # all weights were trained.
    # Below we look at random channels, which kind or look sinusoidal.
    plt.plot(state["transformer.wpe.weight"][:, 150])
    plt.plot(state["transformer.wpe.weight"][:, 200])
    plt.plot(state["transformer.wpe.weight"][:, 250])
    plt.show()


def generate_sample_text():
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="models")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="models")
    generator = pipeline("text-generation", model=gpt2, tokenizer=tokenizer)

    generation_config = deepcopy(generator.generation_config)
    generation_config.max_new_tokens = None
    generation_config.max_length = 30
    generation_config.num_return_sequences = 5

    set_seed(42)
    outputs = generator(
        "Hello, I am a language model,",
        generation_config=generation_config,
        clean_up_tokenization_spaces=False,
    )

    for output in outputs:
        print("-", output["generated_text"], "\n")


def main():
    if True:
        print_gpt2_state()
    if False:
        explore_pretrained_parameters()
    if True:
        generate_sample_text()


if __name__ == "__main__":
    main()
