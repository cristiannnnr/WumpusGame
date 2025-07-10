import pygame
import time
import numpy as np
from agents.dqn_agent import CyberneticAgent
from envs.wumpus_env import WumpusCyberEnv

def run_episode(env, agent, render=False, clock=None, max_steps=100, delay=0.0):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    lyapunov_list = []
    losses = []

    while not done and steps < max_steps:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return total_reward, steps, False, [], True
            env.render()
            if delay > 0:
                time.sleep(delay)
            if clock:
                clock.tick(60)

        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        lyapunov_list.append(next_state['chaos'][1])
        losses.append(agent.update(state, action, reward, next_state, done))
        state = next_state
        steps += 1

    success = info.get("event") == "success"

    if not success:
        print(f"💀 El agente murió por: {info.get('event', 'máximo de pasos')}")
    return total_reward, steps, success, lyapunov_list, losses, False

def evaluate_scenario(scenario, episodes_per_seed=200, seeds=[0, 1, 2], render=False, size=4, delay=0.0, max_steps=100, save_model=False, load_model=False, model_path="model.pt"):
    rewards_all, successes_all, lyapunovs_all, losses_all, steps_all = [], [], [], [], []

    best_agent = None
    best_avg_reward = float('-inf')

    for seed in seeds:        
        print(f"\n Seed: {seed} — Scenario: {scenario}")
        env = WumpusCyberEnv(mode=scenario, size=size, seed=seed)
        agent = CyberneticAgent(env)

        if load_model:
            agent.load(model_path)

        clock = pygame.time.Clock() if render else None

        rewards = []
        successes = []
        lyapunov_means = []
        steps_list = []
        losses_means = []

        for ep in range(episodes_per_seed):
            should_render = render
            total_reward, steps, success, lyapunov_list, losses, fatal = run_episode(
                env, agent, render=should_render, clock=clock, delay=delay, max_steps=max_steps
            )
            if fatal:
                break
            
            rewards.append(total_reward)
            successes.append(success)
            steps_list.append(steps)

            if lyapunov_list:
                lyapunov_means.append(np.mean(lyapunov_list))
            else:
                lyapunov_means.append(0.0)

            if losses:
                losses_means.append(np.mean(losses))
            else:
                losses_means.append(0.0)

            if ep % 10 == 0:
                print(f"  🎯 Ep {ep} — Reward: {total_reward} — Success: {success} — Steps: {steps}")

        # Evaluar si este agente es el mejor
        avg_reward_seed = np.mean(rewards)
        if avg_reward_seed > best_avg_reward:
            best_avg_reward = avg_reward_seed
            best_agent = agent

        rewards_all.append(rewards)
        successes_all.append(successes)
        lyapunovs_all.append(lyapunov_means)
        steps_all.append(steps_list)
        losses_all.append(losses_means)

        env.close()

    # Guardar solo el mejor modelo consolidado
    if save_model and best_agent:
        best_agent.save(model_path)

    avg_rewards = np.mean(rewards_all, axis=0)
    avg_successes = np.mean(successes_all, axis=0)
    avg_lyapunovs = np.mean(lyapunovs_all, axis=0)
    avg_steps = np.mean(steps_all, axis=0)
    avg_losses = np.mean(losses_all, axis=0)

    return avg_rewards, avg_successes, avg_lyapunovs, avg_steps, avg_losses
