import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class HigherOrderAttnLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.lstm = nn.LSTM(config["n_embd"], config["n_embd"], num_layers=config["n_layer"], dropout=config["dropout"], batch_first=True)
        self.attention = nn.Linear(config["n_embd"], 1)
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"])

    def forward(self, x, targets=None):
        x = self.embedding(x)
        x = torch.squeeze(x, dim=2)
        lstm_outputs, _ = self.lstm(x)
        attention_scores = self.attention(lstm_outputs)
        attention_weights = torch.softmax(attention_scores.squeeze(dim=-1), dim=1)
        attended_outputs = torch.bmm(attention_weights.unsqueeze(dim=1), lstm_outputs)
        logits = self.lm_head(attended_outputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) if targets is not None else None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return optim.AdamW(self.parameters(), betas=betas, lr=learning_rate)