import json
import os
import time
import math
import torch
from tqdm import tqdm

from TrajLearn.TrajectoryBatchDataset import TrajectoryBatchDataset


class Trainer(object):
    def __init__(self, model, dataset, config, logger, model_checkpoint_directory, always_save_checkpoint=False, optimizer=None):
        super(Trainer, self).__init__()
        self.model = model
        self.logger = logger
        self.train_dataset = dataset
        self.always_save_checkpoint = always_save_checkpoint
        self.device = config["device"]
        self.device_type = 'cuda' if 'cuda' in config["device"] else 'cpu'
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

        dtype = 'float32'
        ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16}[dtype]
        self.ctx = torch.amp.autocast(
            device_type=self.device_type, dtype=ptdtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        if optimizer is None:
            self.optimizer = model.configure_optimizers(
                self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device_type)
        else:
            self.optimizer = optimizer

        input_lengths = list(
            range(self.min_input_length, self.max_input_length+1))
        self.train_dataset.create_batches(
            self.config["batch_size"], input_lengths)

        self.validation_dataset = TrajectoryBatchDataset(os.path.join(
            config["data_dir"], config["dataset"]), dataset_type='val', delimiter=config["delimiter"], validation_ratio=config["validation_ratio"], test_ratio=config["test_ratio"])
        self.validation_dataset.create_batches(
            self.config["batch_size"], self.min_input_length)

    @torch.no_grad()
    def val_epoch(self):
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
            total_correct += (output.argmax(dim=2)
                              [:, -1] == y[:, -1]).sum().item()
            total_samples += Y.shape[0]

        avg_val_loss = total_val_loss / total_samples
        val_accuracy = total_correct / total_samples
        return avg_val_loss, val_accuracy

    def train(self):
        self.model.train()
        iter_num = 0
        best_val_loss = 1e91
        self.logger.info("Starting training")
        for epoch in range(self.max_epochs):
            # losses = []
            t_epoch_start = time.time()
            total_loss = 0
            total_samples = 0
            for X, Y in (pbar := tqdm(self.train_dataset, leave=False)):
                iter_num += 1
                lr = self.get_lr(
                    iter_num) if self.decay_lr else self.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    with self.ctx:
                        _, loss = self.model(
                            X.to(self.device), Y.to(self.device))
                    total_loss += loss.item()
                    total_samples += X.shape[0]
                    # losses.append(loss.item())
                    self.scaler.scale(loss).backward()
                    if self.grad_clip != 0.0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    pbar.set_postfix({'loss': total_loss/total_samples})
            dt = time.time() - t_epoch_start
            avg_loss = total_loss / total_samples
            self.logger.info(
                f"Training epoch {epoch+1}/{self.max_epochs}, Training loss: {avg_loss:.3g}, Time: {dt:.1f}s")
            t_val_start = time.time()
            avg_val_loss, val_accuracy = self.val_epoch()
            dt = time.time() - t_val_start
            self.logger.info(
                f'Validation loss: {avg_val_loss:.3g}, Validation Accuracy: {val_accuracy*100:.2f}, Time: {dt:.1f}s')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint()
            # json.dump(losses, open('losses'+str(epoch)+'.json', 'w'))

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        if it == self.warmup_iters:
            self.logger.info("warm up iters ended, starting cosine decay")
        # 2) if it > lr_decay_iters, return min learning rate
        if it == self.lr_decay_iters:
            self.logger.info(
                f"learning rate decay ended, using min learning rate {self.min_lr}")
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / \
            (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coefficient * (self.learning_rate - self.min_lr)

    def save_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, os.path.join(
            self.out_dir, 'checkpoint.pt'))
        self.logger.info("Saving current best model to " + self.out_dir)
