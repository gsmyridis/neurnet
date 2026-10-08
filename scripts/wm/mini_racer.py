import matplotlib.pyplot as plt
import numpy as np

from neurnet.gym import MiniRacer, MiniRacerAction

BYTES_IN_MB = 1e6


def plot_frames(frames: np.ndarray, n: int) -> None:
    fig, axes = plt.subplots(1, 8, figsize=(12, 1.9))
    for i in range(8):
        axes[i].imshow(frames[i, i * 9])
        axes[i].axis("off")
    fig.suptitle("MiniRacer — eight raw observations from the dataset", y=1.12)
    plt.tight_layout()
    plt.savefig("plot_env_frames.png", dpi=150, bbox_inches="tight")
    plt.show()


def main() -> None:
    EPISODES, TIME_MAX = 150, 150
    SIZE_X, SIZE_Y = 24, 24
    RANDOM_SEED = 123

    env = MiniRacer(
        size_x=SIZE_X,
        size_y=SIZE_Y,
        radius=8,
        street_width=4.4,
        kerb_width=0.45,
        speed=1.5,
        time_max=TIME_MAX,
        seed=RANDOM_SEED,
    )

    frames = np.zeros((EPISODES, TIME_MAX, SIZE_X, SIZE_Y, 3), dtype=np.float32)
    actions = np.zeros((EPISODES, TIME_MAX), dtype=MiniRacerAction)
    rewards = np.zeros((EPISODES, TIME_MAX), dtype=np.float32)

    for i_episode in range(EPISODES):
        obs = env.reset()
        for t in range(TIME_MAX):
            action = MiniRacerAction.random(RANDOM_SEED)
            frames[i_episode, t], actions[i_episode, t] = obs, action
            obs, reward, _done = env.step(action)
            rewards[i_episode, t] = reward.value

    dataset_size = frames.nbytes // BYTES_IN_MB
    print(f"Dataset: {EPISODES} episodes, {TIME_MAX} time steps, {dataset_size}")


if __name__ == "__main__":
    main()
