import os
import time
import math
import torch
from torch.optim import Optimizer
from tqdm import tqdm
from typing import Optional, Tuple

from TrajLearn.TrajectoryBatchDataset import TrajectoryBatchDataset


class Trainer:
    """
    Trainer class to handle the training and validation of a given model on trajectory datasets.
    """
    def __init__(self, model: torch.nn.Module, dataset: TrajectoryBatchDataset, config: dict,
                 logger, model_checkpoint_directory: str, always_save_checkpoint: bool = False,
                 optimizer: Optional[Optimizer] = None):
        """
        Initialize the Trainer class with model, dataset, configurations, and other options.

        Args:
            model (torch.nn.Module): The model to train.
            dataset (TrajectoryBatchDataset): The dataset to use for training.
            config (dict): Configuration dictionary with training parameters.
            logger: Logger for logging training progress.
            model_checkpoint_directory (str): Directory to save model checkpoints.
            always_save_checkpoint (bool): Whether to save a checkpoint every epoch. Defaults to False.
            optimizer (Optional[Optimizer]): Optimizer to use. If None, a default optimizer is configured.
        """
        self.model = model
        self.logger = logger
        self.train_dataset = dataset
        self.always_save_checkpoint = always_save_checkpoint
        self.device = config["device"]
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        self.config = config
        self.out_dir = model_checkpoint_directory
        self.max_epochs = config["max_epochs"]
        self.block_size = config["block_size"]
        self.batch_size = config["batch_size"]
        self.min_input_length = config["min_input_length"]
        self.max_input_length = config["max_input_length"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.beta1 = config["beta1"]
        self.beta2 = config["beta2"]
        self.grad_clip = config["grad_clip"]
        self.decay_lr = config["decay_lr"]
        self.warmup_iters = config["warmup_iters"]
        self.lr_decay_iters = config["lr_decay_iters"]
        self.min_lr = config["min_lr"]
        self.patience = config["patience"]
        self.early_stopping_counter = 0

        dtype = 'float32'
        ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[dtype]
        self.ctx = torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

        if optimizer is None:
            self.optimizer = model.configure_optimizers(
                self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device_type
            )
        else:
            self.optimizer = optimizer

        input_lengths = list(range(self.min_input_length, self.max_input_length + 1))
        self.train_dataset.create_batches(self.batch_size, input_lengths)

        self.validation_dataset = TrajectoryBatchDataset(
            os.path.join(config["data_dir"], config["dataset"]),
            dataset_type='val',
            delimiter=config["delimiter"],
            validation_ratio=config["validation_ratio"],
            test_ratio=config["test_ratio"]
        )
        self.validation_dataset.create_batches(self.batch_size, self.min_input_length)

    @torch.no_grad()
    def val_epoch(self) -> Tuple[float, float]:
        """
        Run a single validation epoch.

        Returns:
            Tuple[float, float]: Average validation loss and accuracy.
        """
        self.model.eval()
        total_val_loss = 0
        total_correct = 0
        total_samples = 0

        for X, Y in tqdm(self.validation_dataset, leave=False):
            x = X.to(self.device)
            y = Y.to(self.device)
            with self.ctx:
                output, loss = self.model(x, y)
            total_val_loss += loss.item()
            total_correct += (output.argmax(dim=2)[:, -1] == y[:, -1]).sum().item()
            total_samples += Y.shape[0]

        avg_val_loss = total_val_loss / total_samples
        val_accuracy = total_correct / total_samples
        return avg_val_loss, val_accuracy

    def train(self):
        """
        Train the model for a specified number of epochs, validate, and save checkpoints.
        """
        iter_num = 0
        best_val_loss = float('inf')
        self.logger.info("Starting training")

        for epoch in range(self.max_epochs):
            self.model.train()
            t_epoch_start = time.time()
            total_loss = 0
            total_samples = 0

            for X, Y in (pbar := tqdm(self.train_dataset, leave=False)):
                iter_num += 1
                lr = self.get_lr(iter_num) if self.decay_lr else self.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                with self.ctx:
                    _, loss = self.model(X.to(self.device), Y.to(self.device))
                total_loss += loss.item()

                total_samples += X.shape[0]

                self.scaler.scale(loss).backward()
                if self.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({'loss': total_loss / total_samples})

            dt = time.time() - t_epoch_start
            avg_loss = total_loss / total_samples
            self.logger.info(f"Training epoch {epoch + 1}/{self.max_epochs}, "
                             f"Training loss: {avg_loss:.3g}, Time: {dt:.1f}s")

            t_val_start = time.time()
            avg_val_loss, val_accuracy = self.val_epoch()
            dt = time.time() - t_val_start
            self.logger.info(f'Validation loss: {avg_val_loss:.3g}, '
                             f'Validation Accuracy: {val_accuracy * 100:.2f}%, Time: {dt:.1f}s')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint()
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

    def get_lr(self, it: int) -> float:
        """
        Calculate learning rate with optional warmup and cosine decay.

        Args:
            it (int): Current iteration number.

        Returns:
            float: Calculated learning rate for the current iteration.
        """
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        if it == self.warmup_iters:
            self.logger.info("Warm-up iterations ended, starting cosine decay")
        if it == self.lr_decay_iters:
            self.logger.info("Decay iterations ended, using minimum learning rate")
        if it >= self.lr_decay_iters:
            return self.min_lr

        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coefficient * (self.learning_rate - self.min_lr)

    def save_checkpoint(self):
        """
        Save model and optimizer state as a checkpoint.
        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }
        checkpoint_path = os.path.join(self.out_dir, 'checkpoint.pt')
        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info("Saved current best model to " + checkpoint_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
