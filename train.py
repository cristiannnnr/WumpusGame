import argparse
import pygame
import numpy as np
import matplotlib.pyplot as plt
from env_wumpus import WumpusCyberEnv
from agent_dqn import CyberneticAgent
import os


def run_episode(env, agent, render=False, clock=None, max_steps=100):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    lyapunov_list = []

    while not done and steps < max_steps:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return total_reward, steps, False, [], True
            env.render()
            if clock:
                clock.tick(20)

        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        agent.update(state, action, reward, next_state, done)
        total_reward += reward
        lyapunov_list.append(next_state['chaos'][1])

        state = next_state
        steps += 1

    success = info.get("event") == "success"
    return total_reward, steps, success, lyapunov_list, False


def evaluate_scenario(scenario, episodes_per_seed=200, seeds=[0, 1, 2], render=False, size=4):
    rewards_all, successes_all, lyapunovs_all = [], [], []

    for seed in seeds:
        print(f"\n🌱 Seed: {seed} — Scenario: {scenario}")
        env = WumpusCyberEnv(mode=scenario, size=size, seed=seed)
        agent = CyberneticAgent(env)
        clock = pygame.time.Clock() if render else None

        rewards = []
        successes = []
        lyapunov_means = []

        for ep in range(episodes_per_seed):
            should_render = render and (ep % 50 == 0)
            total_reward, steps, success, lyapunov_list, fatal = run_episode(
                env, agent, render=should_render, clock=clock
            )
            if fatal:
                break

            rewards.append(total_reward)
            successes.append(success)

            if lyapunov_list:
                lyapunov_means.append(np.mean(lyapunov_list))
            else:
                lyapunov_means.append(0.0)

            if ep % 10 == 0:
                print(f"  🎯 Ep {ep} — Reward: {total_reward} — Success: {success}")

        rewards_all.append(rewards)
        successes_all.append(successes)
        lyapunovs_all.append(lyapunov_means)

        env.close()

    avg_rewards = np.mean(rewards_all, axis=0)
    avg_successes = np.mean(successes_all, axis=0)
    avg_lyapunovs = np.mean(lyapunovs_all, axis=0)

    return avg_rewards, avg_successes, avg_lyapunovs


def plot_metrics(avg_rewards, avg_successes, avg_lyapunovs, scenario):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Training Metrics — {scenario.title()} Scenario", fontsize=16)

    # Subplot 1: Recompensa promedio
    axs[0].plot(avg_rewards, color="blue", linewidth=2)
    axs[0].set_ylabel("Avg. Reward")
    axs[0].grid(True)

    # Subplot 2: Tasa de éxito (%)
    axs[1].plot(np.array(avg_successes) * 100, color="green", linewidth=2)
    axs[1].set_ylabel("Success Rate (%)")
    axs[1].set_ylim(0, 100)
    axs[1].grid(True)

    # Subplot 3: Lyapunov (si está disponible)
    if np.all(np.isnan(avg_lyapunovs)) or np.all(avg_lyapunovs == 0):
        axs[2].set_title("No Lyapunov data available", fontsize=10)
        axs[2].plot([], [])
    else:
        axs[2].plot(avg_lyapunovs, color="red", linewidth=2)
        axs[2].set_ylabel("Avg. Lyapunov")
        axs[2].grid(True)

    axs[2].set_xlabel("Episode")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = f"plots/{scenario}_metrics.png"
    plt.savefig(output_path)
    plt.close()

    print(f"📊 Combined subplot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="static", choices=["static", "random", "custom"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--size", type=int, default=4)
    args = parser.parse_args()

    print("✅ Running train.py")
    print("🚀 Starting training...")

    avg_rewards, avg_successes, avg_lyapunovs = evaluate_scenario(
        scenario=args.scenario,
        episodes_per_seed=args.episodes,
        seeds=args.seeds,
        render=args.render,
        size=args.size
    )

    plot_metrics(avg_rewards, avg_successes, avg_lyapunovs, args.scenario)



if __name__ == "__main__":
    main()
