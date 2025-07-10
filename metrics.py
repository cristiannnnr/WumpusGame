import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(avg_rewards, avg_successes, avg_lyapunovs, avg_steps, avg_losses, scenario):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    fig, axs = plt.subplots(5, 1, figsize=(7, 9), sharex=True)
    fig.suptitle(f"Training Metrics — {scenario.title()} Scenario (All seeds)", fontsize=16)

    axs[0].plot(avg_rewards, color="blue", linewidth=2)
    axs[0].set_ylabel("Avg. Reward")
    axs[0].grid(True)

    axs[1].plot(np.array(avg_successes) * 100, color="green", linewidth=2)
    axs[1].set_ylabel("Success Rate (%)")
    axs[1].set_ylim(0, 100)
    axs[1].grid(True)

    if np.all(np.isnan(avg_lyapunovs)) or np.all(avg_lyapunovs == 0):
        axs[2].set_title("No Lyapunov data available", fontsize=10)
        axs[2].plot([], [])
    else:
        axs[2].plot(avg_lyapunovs, color="red", linewidth=2)
        axs[2].set_ylabel("Avg. Lyapunov")
        axs[2].grid(True)

    axs[3].plot(avg_steps, color="purple", linewidth=2)
    axs[3].set_ylabel("Avg. Steps")
    axs[3].grid(True)

    axs[4].plot(avg_losses, color="orange", linewidth=2)
    axs[4].set_ylabel("Avg. Loss")
    axs[4].set_xlabel("Episode")
    axs[4].grid(True)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = f"plots/{scenario}_metrics.png"
    plt.savefig(output_path)
    plt.show()
    plt.close()

    print(f"📊 Combined subplot saved: {output_path}")
