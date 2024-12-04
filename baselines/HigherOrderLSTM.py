import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class HigherOrderLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.lstm = nn.LSTM(config["n_embd"], config["n_embd"], num_layers=config["n_layer"], dropout=config["dropout"], batch_first=True)
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"])

    def forward(self, x, targets=None):
        x = self.embedding(x)
        x = torch.squeeze(x, dim=2)
        x, _ = self.lstm(x)
        logits = self.lm_head(x[:, [-1], :])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) if targets is not None else None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return optim.AdamW(self.parameters(), betas=betas, lr=learning_rate)