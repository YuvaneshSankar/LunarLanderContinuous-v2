import argparse
from scripts import train, evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="Mode: train or eval")
    parser.add_argument("--config", default="configs/td3_config.yaml", help="Path to config file")
    parser.add_argument("--model", help="Path to saved model for evaluation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    if args.mode == "train":
        train.train(args.config)
    elif args.mode == "eval":
        if args.model is None:
            print("Please specify --model for evaluation mode")
        else:
            evaluate.evaluate(args.config, args.model, args.episodes)
