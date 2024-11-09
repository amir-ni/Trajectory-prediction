import os
import glob
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from TrajLearn.TrajectoryBatchDataset import TrajectoryBatchDataset
from TrajLearn.model import ModelConfig, CausalLM
from TrajLearn.evaluator import evaluate_model
from TrajLearn.trainer import Trainer
from TrajLearn.logger import get_logger
from torch.utils.data import IterableDataset


def setup_environment(seed: int) -> None:
    """
    Set up the environment by configuring CUDA and setting random seeds.

    Args:
    - seed (int): The seed for random number generators.
    - device_id (str): The CUDA device ID to set for training.
    """
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_dataset(config: Dict[str, Any], test_mode: bool = False) -> TrajectoryBatchDataset:
    """
    Load the trajectory dataset based on configuration.

    Args:
    - config (Dict[str, Any]): Configuration dictionary.
    - test_mode (bool): Whether to load test or training data (default is False).

    Returns:
    - TrajectoryBatchDataset: The dataset object.
    """
    dataset_type = 'test' if test_mode else 'train'
    dataset_path = Path(config["data_dir"]) / config["dataset"]
    dataset = TrajectoryBatchDataset(
        dataset_path,
        dataset_type=dataset_type,
        delimiter=config["delimiter"],
        validation_ratio=config["validation_ratio"], 
        test_ratio=config["test_ratio"]
    )
    config["vocab_size"] = dataset.vocab_size
    return dataset


def load_model(config: Dict[str, Any], checkpoint_path: Optional[Path] = None, custom_init=None) -> torch.nn.Module:
    """
    Initialize and optionally load a model from a checkpoint.

    Args:
    - config (Dict[str, Any]): Configuration dictionary.
    - dataset (TrajectoryBatchDataset): Dataset to extract vocabulary size.
    - checkpoint_path (Optional[Path]): Path to the model checkpoint (default is None).

    Returns:
    - Module: The initialized model, possibly with loaded weights.
    """
    model_config = ModelConfig(
        block_size=config["block_size"],
        vocab_size=config["vocab_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        bias=config["bias"]
    )
    model = CausalLM(model_config, custom_init)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=config["device"])
        config_dict = checkpoint['config']
        optimizer_dict = checkpoint['optimizer']
        state_dict = checkpoint['model']
        # Remove unwanted prefixes in state_dict keys
        unwanted_prefix = '_orig_mod.'
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    return model


def train_model(
    name: str,
    dataset: TrajectoryBatchDataset,
    config: Dict[str, Any],
    model: Optional[torch.nn.Module] = None,
    custom_init: Optional[torch.Tensor] = None
) -> None:
    """
    Set up and execute the training process.

    Args:
    - name (str): Name for the current training session (used for saving logs/checkpoints).
    - dataset (TrajectoryBatchDataset): Dataset object for training.
    - config (Dict[str, Any]): Configuration dictionary.
    - model (Optional[torch.nn.Module]): The model to be trained (can be None before loading).
    """
    time_str = name + "-" + time.strftime("%Y%m%d-%H%M%S")
    Path(config["model_checkpoint_directory"]).mkdir(parents=True, exist_ok=True)
    model_checkpoint_directory = Path(config["model_checkpoint_directory"]) / time_str
    log_directory = model_checkpoint_directory / 'logs'

    if model is None:
        if config['train_from_checkpoint_if_exist']:
            model_checkpoints = sorted(glob.glob(str(Path(config["model_checkpoint_directory"]) / (name + "-*"))))
            if len(model_checkpoints) > 0:
                last_checkpoint = Path(model_checkpoints[-1]) / 'checkpoint.pt'
                model = load_model(config, checkpoint_path=last_checkpoint)

        if config['custom_initialization'] and model is None:
            custom_init_path = os.path.join(config["data_dir"], config["dataset"], 'embeddings.npy')
            embeddings_np = np.load(custom_init_path)
            custom_init = torch.from_numpy(embeddings_np).to(torch.float32)
            model = load_model(config, custom_init=custom_init)

        if model is None:
            model = load_model(config)

    logger = get_logger(log_directory, phase="train")
    model.to(config["device"])

    trainer = Trainer(model, dataset, config, logger, str(model_checkpoint_directory))
    trainer.train()


def test_model(name: str, dataset: TrajectoryBatchDataset, config: Dict[str, Any], model: Optional[torch.nn.Module] = None) -> list:
    """
    Set up and execute the testing process.

    Args:
    - name (str): Name of the configuration (used for loading the model checkpoint).
    - dataset (TrajectoryBatchDataset): Dataset object for testing.
    - config (Dict[str, Any]): Configuration dictionary.
    - model (Optional[torch.nn.Module]): The model to be tested (can be None before loading).
    """
    model_checkpoint_directory = sorted(glob.glob(str(Path(config["model_checkpoint_directory"]) / (name + "-*"))))[-1]
    log_directory = Path(model_checkpoint_directory) / 'logs'

    logger = get_logger(log_directory, phase="test")

    if model is None:
        checkpoint_path = Path(model_checkpoint_directory) / 'checkpoint.pt'
        model = load_model(config, checkpoint_path=checkpoint_path)
    model.to(config["device"])

    prediction_length = config["test_prediction_length"]
    dataset.create_batches(
        config["batch_size"], config["test_input_length"], prediction_length, False, False)

    return evaluate_model(model, dataset, config, logger)
