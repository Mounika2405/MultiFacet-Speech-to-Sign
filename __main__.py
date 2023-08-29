import argparse

# from training import train, test
from training_speech2sign import train, test


def main():

    # Example options:
    # train ./Configs/Base.yaml
    # test ./Configs/Base.yaml

    ap = argparse.ArgumentParser("Progressive Transformers")

    # Choose between Train and Test
    ap.add_argument("mode", choices=["train", "test"],
                    help="train a model or test")
    # Path to Config
    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")
    ap.add_argument("--ckpt", type=str, default = None,
                    help="path to ckpt file")

    args = ap.parse_args()

    # If Train
    if args.mode == "train":
        train(cfg_file=args.config_path, ckpt=args.ckpt)
    # If Test
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
