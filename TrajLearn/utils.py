import glob
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from TrajLearn.TrajectoryBatchDataset import TrajectoryBatchDataset
from TrajLearn.model import GPTConfig, GPT
from TrajLearn.evaluator import evaluate_model
from TrajLearn.trainer import Trainer
from TrajLearn.logger import get_logger


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
    data_type = 'test' if test_mode else 'train'
    dataset_path = Path(config["data_dir"]) / config["dataset"]
    dataset = TrajectoryBatchDataset(
        dataset_path,
        type=data_type,
        delimiter=config["delimiter"],
        validation_ratio=config["validation_ratio"], 
        test_ratio=config["test_ratio"]
    )
    return dataset


def load_model(config: Dict[str, Any], dataset: TrajectoryBatchDataset, checkpoint_path: Optional[Path] = None) -> GPT:
    """
    Initialize and optionally load a model from a checkpoint.

    Args:
    - config (Dict[str, Any]): Configuration dictionary.
    - dataset (TrajectoryBatchDataset): Dataset to extract vocabulary size.
    - checkpoint_path (Optional[Path]): Path to the model checkpoint (default is None).

    Returns:
    - GPT: The initialized GPT model, possibly with loaded weights.
    """
    gptconf = GPTConfig(
        block_size=config["block_size"],
        vocab_size=dataset.vocab_size,
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        bias=config["bias"]
    )
    model = GPT(gptconf)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=config["device"])
        state_dict = checkpoint['model']
        # Remove unwanted prefixes in state_dict keys
        unwanted_prefix = '_orig_mod.'
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    model.to(config["device"])
    return model


def train_model(config: Dict[str, Any], dataset: TrajectoryBatchDataset, model: GPT, name: str) -> None:
    """
    Set up and execute the training process.

    Args:
    - config (Dict[str, Any]): Configuration dictionary.
    - dataset (TrajectoryBatchDataset): Dataset object for training.
    - model (GPT): The GPT model to be trained.
    - name (str): Name for the current training session (used for saving logs/checkpoints).
    """
    time_str = name + "-" + time.strftime("%Y%m%d-%H%M%S")
    model_checkpoint_directory = Path(config["model_checkpoint_directory"]) / time_str
    log_directory = model_checkpoint_directory / 'logs'

    logger = get_logger(log_directory, phase="train")
    model.to(config["device"])

    trainer = Trainer(model, dataset, config, logger, str(model_checkpoint_directory))
    trainer.train()


def test_model(config: Dict[str, Any], dataset: TrajectoryBatchDataset, model: Optional[GPT], name: str) -> list:
    """
    Set up and execute the testing process.

    Args:
    - config (Dict[str, Any]): Configuration dictionary.
    - dataset (TrajectoryBatchDataset): Dataset object for testing.
    - model (Optional[GPT]): The GPT model to be tested (can be None before loading).
    - name (str): Name of the configuration (used for loading the model checkpoint).
    """
    model_checkpoint_directory = sorted(glob.glob(str(Path(config["model_checkpoint_directory"]) / (name + "-*"))))[-1]
    log_directory = Path(model_checkpoint_directory) / 'logs'

    logger = get_logger(log_directory, phase="test")
    checkpoint_path = Path(model_checkpoint_directory) / 'checkpoint.pt'

    model = load_model(config, dataset, checkpoint_path)

    return evaluate_model(model, dataset, config, logger)
