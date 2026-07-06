import torch
import torch.nn as nn


class RawEEGConformer(nn.Module):
    """Small EEG Conformer: CNN window encoder plus temporal Transformer."""

    def __init__(
        self,
        eeg_channels=17,
        embedding_dim=64,
        attention_heads=4,
        window_transformer_layers=1,
        temporal_layers=1,
        num_classes=3,
        dropout=0.1,
        max_sequence_length=32,
    ):
        super().__init__()
        if embedding_dim % attention_heads != 0:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        if window_transformer_layers < 0 or temporal_layers < 0:
            raise ValueError("transformer layer counts must be non-negative")

        self.eeg_channels = int(eeg_channels)
        self.embedding_dim = int(embedding_dim)
        self.max_sequence_length = int(max_sequence_length)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(self.eeg_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout),
        )

        if window_transformer_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attention_heads,
                dim_feedforward=embedding_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.window_transformer = nn.TransformerEncoder(layer, window_transformer_layers)
        else:
            self.window_transformer = nn.Identity()

        self.position_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, embedding_dim))
        if temporal_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attention_heads,
                dim_feedforward=embedding_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.temporal_transformer = nn.TransformerEncoder(layer, temporal_layers)
        else:
            self.temporal_transformer = nn.Identity()

        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.perclos_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())

    def forward(self, eeg):
        if eeg.ndim != 4:
            raise ValueError("eeg must have shape [batch, time, channels, samples]")
        if eeg.shape[2] != self.eeg_channels:
            raise ValueError(f"expected {self.eeg_channels} EEG channels, got {eeg.shape[2]}")
        if eeg.shape[1] > self.max_sequence_length:
            raise ValueError("sequence length exceeds max_sequence_length")

        batch_size, sequence_length, channels, samples = eeg.shape
        windows = eeg.reshape(batch_size * sequence_length, 1, channels, samples)
        tokens = self.patch_encoder(windows).squeeze(2).transpose(1, 2)
        tokens = self.window_transformer(tokens)
        window_embeddings = tokens.mean(dim=1).reshape(batch_size, sequence_length, self.embedding_dim)

        sequence = window_embeddings + self.position_embedding[:, :sequence_length]
        sequence = self.temporal_transformer(sequence)
        pooled = sequence[:, -1]
        return {
            "class_logits": self.classifier(pooled),
            "perclos": self.perclos_head(pooled),
        }
