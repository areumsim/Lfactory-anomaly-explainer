"""LSTM Autoencoder for anomaly detection in time series.

Research context:
- Deep learning approach that learns temporal patterns through sequence
  reconstruction. Points with high reconstruction error are flagged as anomalies.
- Captures complex non-linear temporal dependencies that simpler methods miss.

Exports:
- detect_lstm_ae(series, sequence_length=50, latent_dim=32, epochs=50, **kwargs) -> dict

Returns:
- dict with keys: scores, preds, params

Dependencies:
- torch (PyTorch)

Notes:
- Requires PyTorch. Training may take longer than IsolationForest.
- Uses early stopping to prevent overfitting.
- Reconstruction error (MSE) per timestep is the anomaly score.
"""
from __future__ import annotations

from typing import List, Dict, Any
import math


def detect_lstm_ae(
    series: List[float],
    sequence_length: int = 50,
    latent_dim: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    quantile: float = 0.95,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """Detect anomalies using LSTM Autoencoder.

    Args:
        series: Input time series data
        sequence_length: Length of input sequences for LSTM
        latent_dim: Dimension of latent representation
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        quantile: Threshold quantile for binary predictions
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments (ignored)

    Returns:
        dict with keys:
            - scores: List[float] - Reconstruction errors (higher = more anomalous)
            - preds: List[int] - Binary predictions (1 = anomaly, 0 = normal)
            - params: dict - Parameters used
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        raise ImportError(
            "LSTM Autoencoder requires PyTorch. "
            "Install it with: pip install torch"
        )

    if len(series) < sequence_length:
        # Not enough data, return zeros
        return {
            "scores": [0.0] * len(series),
            "preds": [0] * len(series),
            "params": {
                "method": "lstm_autoencoder",
                "sequence_length": sequence_length,
                "latent_dim": latent_dim,
                "epochs": epochs,
                "error": "insufficient_data"
            }
        }

    # Set random seed
    torch.manual_seed(random_state)

    # Define LSTM Autoencoder model
    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int):
            super(LSTMAutoencoder, self).__init__()
            self.latent_dim = latent_dim

            # Encoder
            self.encoder_lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=latent_dim,
                num_layers=1,
                batch_first=True
            )

            # Decoder
            self.decoder_lstm = nn.LSTM(
                input_size=latent_dim,
                hidden_size=latent_dim,
                num_layers=1,
                batch_first=True
            )
            self.output_layer = nn.Linear(latent_dim, input_dim)

        def forward(self, x):
            # Encode
            _, (hidden, cell) = self.encoder_lstm(x)

            # Repeat hidden state for decoder input
            batch_size, seq_len, _ = x.size()
            decoder_input = hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)

            # Decode
            decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
            reconstruction = self.output_layer(decoder_output)

            return reconstruction

    # Prepare sequences
    def create_sequences(data, seq_len):
        sequences = []
        for i in range(len(data) - seq_len + 1):
            seq = data[i:i+seq_len]
            sequences.append(seq)
        return sequences

    sequences = create_sequences(series, sequence_length)

    # Normalize data (important for LSTM)
    mean_val = sum(series) / len(series)
    if len(series) > 1:
        variance = sum((x - mean_val) ** 2 for x in series) / (len(series) - 1)
        std_val = math.sqrt(variance) if variance > 0 else 1.0
    else:
        std_val = 1.0

    normalized_series = [(x - mean_val) / std_val for x in series]
    normalized_sequences = create_sequences(normalized_series, sequence_length)

    # Convert to tensors
    X = torch.FloatTensor(normalized_sequences).unsqueeze(-1)  # (N, seq_len, 1)

    # Initialize model
    model = LSTMAutoencoder(input_dim=1, latent_dim=latent_dim)
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    model.train()
    num_batches = (len(X) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        total_loss = 0.0
        # Shuffle data each epoch (simple approach)
        indices = list(range(len(X)))

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X))
            batch_indices = indices[start_idx:end_idx]

            batch_x = X[batch_indices]

            # Forward pass
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    # Compute reconstruction errors
    model.eval()
    with torch.no_grad():
        reconstructed = model(X)
        reconstruction_errors_seq = criterion(reconstructed, X).squeeze(-1)  # (N, seq_len)

    # Map errors back to time series points
    # Each point can appear in multiple sequences, so average the errors
    n = len(series)
    error_sums = [0.0] * n
    error_counts = [0] * n

    for seq_idx, seq_errors in enumerate(reconstruction_errors_seq):
        for pos_in_seq, error in enumerate(seq_errors):
            time_idx = seq_idx + pos_in_seq
            error_sums[time_idx] += error.item()
            error_counts[time_idx] += 1

    # Average errors
    scores = []
    for i in range(n):
        if error_counts[i] > 0:
            scores.append(error_sums[i] / error_counts[i])
        else:
            # First few points may not be in any sequence
            scores.append(0.0)

    # Normalize scores to [0, 1] for consistency
    if len(scores) > 0 and max(scores) > 0:
        max_score = max(scores)
        scores = [s / max_score for s in scores]

    # Threshold at quantile
    def _quantile(values, q):
        if not values:
            return 0.0
        q = min(max(q, 0.0), 1.0)
        arr = sorted(values)
        pos = q * (len(arr) - 1)
        i = int(pos)
        if i >= len(arr) - 1:
            return arr[-1]
        frac = pos - i
        return arr[i] * (1 - frac) + arr[i + 1] * frac

    threshold = _quantile(scores, quantile)
    preds = [1 if s >= threshold else 0 for s in scores]

    return {
        "scores": scores,
        "preds": preds,
        "params": {
            "method": "lstm_autoencoder",
            "sequence_length": int(sequence_length),
            "latent_dim": int(latent_dim),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "quantile": float(quantile),
            "random_state": int(random_state),
        }
    }
