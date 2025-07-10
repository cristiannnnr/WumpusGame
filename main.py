import argparse
from trainer import evaluate_scenario
from metrics import plot_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="static", choices=["static", "random", "custom"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.0, help="Delay (seconds) between agent actions during render")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--save_model", action="store_true", help="Save the model after training")
    parser.add_argument("--load_model", action="store_true", help="Load the model before training")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to save/load the model")
    args = parser.parse_args()

    print("Running main.py")
    print("Starting training...")

    avg_rewards, avg_successes, avg_lyapunovs, avg_steps = evaluate_scenario(
        scenario=args.scenario,
        episodes_per_seed=args.episodes,
        seeds=args.seeds,
        render=args.render,
        size=args.size,
        delay=args.delay,
        max_steps=args.max_steps,
        save_model=args.save_model,
        load_model=args.load_model,
        model_path=args.model_path
    )

    plot_metrics(avg_rewards, avg_successes, avg_lyapunovs, avg_steps, args.scenario)

if __name__ == "__main__":
    main()
