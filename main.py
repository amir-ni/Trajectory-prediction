import glob
import os
import time
import random
import logging
import argparse
import numpy as np
import torch
import yaml
from TrajectoryBatchDataset import TrajectoryBatchDataset
from model import GPTConfig, GPT
from test import test
from trainer import Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.cuda.cudnn_enabled = False
torch.backends.cudnn.deterministic = True


def get_logger(log_directory, phase="train"):
    logger = logging.getLogger(phase)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logfile = os.path.join(
        log_directory, phase + ".log")
    file_handler = logging.FileHandler(logfile, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # file_handler2 = logging.FileHandler("config10.txt", mode='a')
    # file_handler2.setLevel(logging.INFO)
    # file_handler2.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    # logger.addHandler(file_handler2)
    return logger


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Trajectory Prediction Learning')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config_list = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for name, config in config_list.items():

        # set seed
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # get dataset
        if args.test:
            dataset = TrajectoryBatchDataset(os.path.join(
                config["data_dir"], config["dataset"]), type='test', delimiter=config["delimiter"], validation_ratio=config["validation_ratio"], test_ratio=config["test_ratio"])
        else:
            dataset = TrajectoryBatchDataset(os.path.join(
                config["data_dir"], config["dataset"]), type='train', delimiter=config["delimiter"], validation_ratio=config["validation_ratio"], test_ratio=config["test_ratio"])

        # get model
        gptconf = GPTConfig(
            block_size=config["block_size"],
            vocab_size=dataset.vocab_size,
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_embd=config["n_embd"],
            dropout=config["dropout"],
            bias=config["bias"])
        model = GPT(gptconf)

        if not args.test:
            # make and setup directories
            time_str = name + "-" + time.strftime("%Y%m%d-%H%M%S")
            model_checkpoint_directory = os.path.join(
                config["model_checkpoint_directory"], time_str)
            log_directory = os.path.join(model_checkpoint_directory, 'logs')
            os.makedirs(log_directory, exist_ok=True)
            logger = get_logger(log_directory, phase="train")
            model.to(config["device"])
            trainer = Trainer(model, dataset, config, logger,
                              model_checkpoint_directory)
            trainer.train()
        else:
            model_checkpoint_directory = glob.glob(os.path.join(
                config["model_checkpoint_directory"], name + "-*"))[-1]
            log_directory = os.path.join(model_checkpoint_directory, 'logs')
            logger = get_logger(log_directory, phase="test")
            checkpoint_path = os.path.join(
                model_checkpoint_directory, 'checkpoint.pt')
            checkpoint = torch.load(
                checkpoint_path, map_location=config["device"])
            state_dict = checkpoint['model']

            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            model.to(config["device"])
            accuracy, bleu_score = test(model, dataset, config, logger)
            print(accuracy)
