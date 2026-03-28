from __future__ import annotations

import torch
from torch import nn


class LSTMAnomalyClassifier(nn.Module):
    """
    LSTM classifier for sequence level anomaly detection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, (hidden_state, _) = self.lstm(inputs)
        final_hidden = hidden_state[-1]
        logits = self.classifier(final_hidden).squeeze(-1)
        return logits