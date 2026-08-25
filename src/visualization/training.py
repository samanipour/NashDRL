from __future__ import annotations


def plot_training_history(history: list[dict[str, float]]) -> None:
    import matplotlib.pyplot as plt
    if not history:
        return
    values = [item["mean_reward"] for item in history]
    plt.plot(values)
    plt.xlabel("Step")
    plt.ylabel("Mean reward")
    plt.tight_layout()
    plt.show()
