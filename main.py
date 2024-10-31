import argparse
from TrajLearn.utils import setup_environment, get_dataset, test_model, train_model
from TrajLearn.config_loader import load_config_with_defaults


def main() -> None:
    """
    Main function to handle argument parsing and execute training or testing.
    """
    parser = argparse.ArgumentParser(description='Trajectory Prediction Learning')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()

    config_list = load_config_with_defaults(args.config)

    for name, config in config_list.items():
        setup_environment(config["seed"])

        dataset = get_dataset(config, test_mode=args.test)

        if args.test:
            test_model(name, dataset, config)
        else:
            train_model(name, dataset, config)


if __name__ == '__main__':
    main()
