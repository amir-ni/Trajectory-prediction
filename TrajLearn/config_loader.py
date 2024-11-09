import yaml

default_config = {
    "test_ratio": 0.2,
    "validation_ratio": 0.1,
    "delimiter": " ",
    "min_input_length": 10,
    "max_input_length": 14,
    "test_input_length": 10,
    "test_prediction_length": 5,
    "batch_size": 128,
    "device": "cpu",
    "max_epochs": 10,
    "block_size": 24,
    "learning_rate": 5.e-3,
    "weight_decay": 5.e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "decay_lr": True,
    "warmup_iters": 200,
    "lr_decay_iters": 40000,
    "min_lr": 5.e-7,
    "seed": 42,
    "data_dir": "./data",
    "dataset": "geolife7",
    "n_layer": 12,
    "n_head": 6,
    "n_embd": 512,
    "bias": False,
    "dropout": 0.1,
    "model_checkpoint_directory": "./models/",
    "train_from_checkpoint_if_exist": False,
    "custom_initialization": False,
    "patience": 3,
    "continuity": True,
    "beam_width": 5,
    "store_predictions": False,
}

def load_config(config_file: str) -> dict:
    """
    Load a YAML configuration file and apply default values for missing parameters.

    Parameters:
    - config_file (str): Path to the configuration YAML file.

    Returns:
    - config_list (dict): A dictionary with the final configuration, including defaults.
    """
    with open(config_file, 'r', encoding='utf-8') as stream:
        try:
            config_list = yaml.safe_load(stream)

            if config_list is None:
                config_list = {}

            for config_name, config_values in config_list.items():
                if config_values is None:
                    config_values = {}

                for key, value in default_config.items():
                    config_values[key] = config_values.get(key, value)

                config_list[config_name] = config_values

            return config_list

        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None
