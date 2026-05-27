from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM autoencoder for sequences.

    It learns to recreate the input sequence. Large error means "more unusual".
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

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B, L, F]
        returns: reconstruction [B, L, F]
        """
        _, (h_n, c_n) = self.encoder(inputs)
        # Use top-layer hidden state as a context vector.
        context = h_n[-1]  # [B, H]
        seq_len = inputs.size(1)

        # Repeat context for each timestep as decoder input.
        decoder_in = context.unsqueeze(1).repeat(1, seq_len, 1)  # [B, L, H]
        dec_out, _ = self.decoder(decoder_in, (h_n, c_n))
        recon = self.output(dec_out)
        return recon
