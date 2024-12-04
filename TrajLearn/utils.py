import os
import glob
import time
import pickle
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
from baselines import HigherOrderMarkovChain


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


def load_model(model: torch.nn.Module | HigherOrderMarkovChain, checkpoint_path: Optional[Path], device: str) -> torch.nn.Module:
    """
    Load a model from a checkpoint.

    Args:
    - config (Dict[str, Any]): Configuration dictionary.
    - dataset (TrajectoryBatchDataset): Dataset to extract vocabulary size.
    - checkpoint_path (Optional[Path]): Path to the model checkpoint (default is None).

    Returns:
    - Module: The initialized model, possibly with loaded weights.
    """
    if isinstance(model, HigherOrderMarkovChain):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        optimizer = None
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        optimizer = checkpoint['optimizer']
    config = checkpoint['config']
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    return model, config, optimizer

def initialize_model(config, custom_init=None):
    model_config = ModelConfig(
        block_size=config["block_size"],
        vocab_size=config["vocab_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        bias=config["bias"]
    )
    return CausalLM(model_config, custom_init)


def train_model(
    name: str,
    dataset: TrajectoryBatchDataset,
    config: Dict[str, Any],
    model: Optional[torch.nn.Module | HigherOrderMarkovChain] = None
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
    model_checkpoint_directory = Path(config["model_checkpoint_directory"]) / time_str
    Path(model_checkpoint_directory).mkdir(parents=True, exist_ok=True)
    log_directory = model_checkpoint_directory / 'logs'

    if model is None:
        if config['custom_initialization']:
            custom_init_path = os.path.join(config["data_dir"], config["dataset"], 'embeddings.npy')
            embeddings_np = np.load(custom_init_path)
            custom_init = torch.from_numpy(embeddings_np).to(torch.float32)
            model = initialize_model(config, custom_init=custom_init)
        else:
            model = initialize_model(config=config)

    if config['train_from_checkpoint_if_exist']:
        model_checkpoints = sorted(glob.glob(str(Path(config["model_checkpoint_directory"]) / (name + "-*"))))
        if len(model_checkpoints) > 0:
            last_checkpoint = Path(model_checkpoints[-1]) / 'checkpoint.pt'
            model, config, optimizer = load_model(model, last_checkpoint, config['device'])

    logger = get_logger(log_directory, name, phase="train")

    if isinstance(model, HigherOrderMarkovChain):
        model.train(dataset, logger, str(model_checkpoint_directory))
    else:
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
    model_checkpoint_directory = sorted(glob.glob(str(Path(config["model_checkpoint_directory"]) / (name + "-2024*"))))[-1]
    log_directory = Path(model_checkpoint_directory) / 'logs'

    logger = get_logger(log_directory, name, phase="test")

    if model is None:
        model = initialize_model(config)

    checkpoint_path = Path(model_checkpoint_directory) / 'checkpoint.pt'
    model, _, __ = load_model(model, checkpoint_path, config['device'])

    prediction_length = config["test_prediction_length"]
    dataset.create_batches(
        config["batch_size"], config["test_input_length"], prediction_length, False, False)

    if isinstance(model, HigherOrderMarkovChain):
        results = model.evaluate(dataset)
        logger.info(", ".join(results))
        return results
    else:
        model.to(config["device"])
        return evaluate_model(model, dataset, config, logger)
