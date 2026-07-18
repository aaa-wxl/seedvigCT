import torch
import torch.nn as nn


CROSS_ATTENTION_DIRECTION_CHOICES = ("eeg_queries_eog", "eog_queries_eeg", "bidirectional")


class RawEEGConformer(nn.Module):
    """Small EEG Conformer: CNN window encoder plus temporal Transformer."""

    def __init__(
        self,
        eeg_channels=17,
        eog_channels=7,
        embedding_dim=64,
        attention_heads=4,
        window_transformer_layers=1,
        temporal_layers=1,
        num_classes=3,
        dropout=0.1,
        max_sequence_length=32,
        use_eog_cross_attention=False,
        use_temporal_delta=False,
        use_eog_gate=False,
        use_eog_anchor_residual=False,
        use_eog_residual_correction=False,
        eog_dropout=0.0,
        cross_attention_direction="eeg_queries_eog",
    ):
        super().__init__()
        if embedding_dim % attention_heads != 0:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        if window_transformer_layers < 0 or temporal_layers < 0:
            raise ValueError("transformer layer counts must be non-negative")
        if cross_attention_direction not in CROSS_ATTENTION_DIRECTION_CHOICES:
            raise ValueError(f"cross_attention_direction must be one of: {CROSS_ATTENTION_DIRECTION_CHOICES}")
        if use_eog_anchor_residual and use_eog_residual_correction:
            raise ValueError("choose either EOG-anchor residual or EOG residual correction")
        if use_eog_cross_attention and use_eog_residual_correction:
            raise ValueError("EOG residual correction cannot also use cross-attention")

        self.eeg_channels = int(eeg_channels)
        self.eog_channels = int(eog_channels)
        self.embedding_dim = int(embedding_dim)
        self.max_sequence_length = int(max_sequence_length)
        self.use_eog_cross_attention = bool(use_eog_cross_attention)
        self.use_temporal_delta = bool(use_temporal_delta)
        self.use_eog_gate = bool(use_eog_gate)
        self.use_eog_anchor_residual = bool(use_eog_anchor_residual)
        self.use_eog_residual_correction = bool(use_eog_residual_correction)
        self.eog_dropout = float(eog_dropout)
        self.cross_attention_direction = cross_attention_direction
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
        if self.use_eog_cross_attention or self.use_eog_anchor_residual or self.use_eog_residual_correction:
            self.eog_patch_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 32), padding=(0, 16), bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=(self.eog_channels, 1), groups=16, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, embedding_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embedding_dim),
                nn.GELU(),
                nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
                nn.Dropout(dropout),
            )
        if self.use_eog_cross_attention:
            self.eog_cross_attention = nn.MultiheadAttention(
                embedding_dim,
                attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.eog_fusion_norm = nn.LayerNorm(embedding_dim)
            if self.use_eog_gate:
                self.eog_gate = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, 1),
                    nn.Sigmoid(),
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

        head_dim = embedding_dim * 2 if self.use_temporal_delta else embedding_dim
        self.classifier = nn.Linear(head_dim, num_classes)
        self.perclos_head = nn.Sequential(nn.Linear(head_dim, 1), nn.Sigmoid())
        if self.use_eog_anchor_residual or self.use_eog_residual_correction:
            self.eog_perclos_head = nn.Sequential(nn.Linear(head_dim, 1), nn.Sigmoid())
            self.eog_residual_gate = nn.Sequential(
                nn.Linear(head_dim * 2, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 1),
                nn.Sigmoid(),
            )
            if self.use_eog_anchor_residual:
                self.eeg_perclos_head = nn.Sequential(nn.Linear(head_dim, 1), nn.Sigmoid())
            if self.use_eog_residual_correction:
                self.eog_residual_delta = nn.Linear(head_dim * 2, 1)

    def _encode_eog_tokens(self, eog, batch_size, sequence_length):
        if eog.ndim != 4:
            raise ValueError("eog must have shape [batch, time, channels, samples]")
        if eog.shape[2] != self.eog_channels:
            raise ValueError(f"expected {self.eog_channels} EOG channels, got {eog.shape[2]}")
        eog_windows = eog.reshape(batch_size * sequence_length, 1, self.eog_channels, eog.shape[-1])
        return self.eog_patch_encoder(eog_windows).squeeze(2).transpose(1, 2)

    def _pool_sequence(self, window_embeddings):
        sequence_length = window_embeddings.shape[1]
        sequence = window_embeddings + self.position_embedding[:, :sequence_length]
        sequence = self.temporal_transformer(sequence)
        pooled = sequence[:, -1]
        if self.use_temporal_delta:
            if sequence_length > 1:
                delta = sequence[:, -1] - sequence[:, -2]
            else:
                delta = torch.zeros_like(pooled)
            pooled = torch.cat([pooled, delta], dim=-1)
        return pooled, sequence

    def _cross_attend(self, base_embeddings, source_tokens, batch_size, sequence_length):
        query = base_embeddings.reshape(batch_size * sequence_length, 1, self.embedding_dim)
        attended, weights = self.eog_cross_attention(query, source_tokens, source_tokens)
        attended = attended.reshape(batch_size, sequence_length, self.embedding_dim)
        gate = None
        if self.use_eog_gate:
            gate = self.eog_gate(torch.cat([base_embeddings, attended], dim=-1))
        if self.training and self.eog_dropout > 0:
            keep_probability = max(0.0, 1.0 - self.eog_dropout)
            mask = attended.new_empty(batch_size, sequence_length, 1).bernoulli_(keep_probability)
            if keep_probability > 0:
                mask = mask / keep_probability
            if gate is None:
                attended = attended * mask
            else:
                gate = gate * mask
        if gate is not None:
            attended = gate * attended
        return attended, weights.reshape(batch_size, sequence_length, *weights.shape[1:]), gate

    def forward(self, eeg, eog=None):
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
        outputs = {}
        if self.use_eog_anchor_residual or self.use_eog_residual_correction:
            if eog is None:
                raise ValueError("EOG residual fusion requires EOG input")
            eog_tokens = self._encode_eog_tokens(eog, batch_size, sequence_length)
            eog_window_embeddings = eog_tokens.mean(dim=1).reshape(batch_size, sequence_length, self.embedding_dim)
            eeg_pooled, _ = self._pool_sequence(window_embeddings)
            eog_pooled, _ = self._pool_sequence(eog_window_embeddings)
            perclos_eog = self.eog_perclos_head(eog_pooled)
            residual_gate = self.eog_residual_gate(torch.cat([eeg_pooled, eog_pooled], dim=-1))
            pooled = eog_pooled + residual_gate * (eeg_pooled - eog_pooled)
            if self.use_eog_residual_correction:
                residual_delta = self.eog_residual_delta(torch.cat([eeg_pooled, eog_pooled], dim=-1))
                perclos = torch.sigmoid(
                    torch.logit(perclos_eog.clamp(1e-4, 1 - 1e-4)) + residual_gate * residual_delta
                )
                outputs.update(
                    {
                        "class_logits": self.classifier(pooled),
                        "perclos": perclos,
                        "perclos_eog": perclos_eog,
                        "eog_residual_delta": residual_delta,
                        "eog_residual_gate": residual_gate,
                    }
                )
                return outputs

            perclos_eeg = self.eeg_perclos_head(eeg_pooled)
            outputs.update(
                {
                    "class_logits": self.classifier(pooled),
                    "perclos": perclos_eog + residual_gate * (perclos_eeg - perclos_eog),
                    "perclos_eeg": perclos_eeg,
                    "perclos_eog": perclos_eog,
                    "eog_residual_gate": residual_gate,
                }
            )
            return outputs

        if self.use_eog_cross_attention and eog is not None:
            eog_tokens = self._encode_eog_tokens(eog, batch_size, sequence_length)
            if self.cross_attention_direction == "eeg_queries_eog":
                attended, weights, gate = self._cross_attend(window_embeddings, eog_tokens, batch_size, sequence_length)
                window_embeddings = self.eog_fusion_norm(window_embeddings + attended)
                if gate is not None:
                    outputs["eog_gate"] = gate
                outputs["eog_attention_weights"] = weights
            elif self.cross_attention_direction == "eog_queries_eeg":
                eog_window_embeddings = eog_tokens.mean(dim=1).reshape(batch_size, sequence_length, self.embedding_dim)
                attended, weights, gate = self._cross_attend(eog_window_embeddings, tokens, batch_size, sequence_length)
                window_embeddings = self.eog_fusion_norm(eog_window_embeddings + attended)
                if gate is not None:
                    outputs["eog_gate"] = gate
                outputs["eog_attention_weights"] = weights
            else:
                eog_window_embeddings = eog_tokens.mean(dim=1).reshape(batch_size, sequence_length, self.embedding_dim)
                eeg_attended, eeg_weights, eeg_gate = self._cross_attend(
                    window_embeddings,
                    eog_tokens,
                    batch_size,
                    sequence_length,
                )
                eog_attended, eog_weights, eog_gate = self._cross_attend(
                    eog_window_embeddings,
                    tokens,
                    batch_size,
                    sequence_length,
                )
                window_embeddings = self.eog_fusion_norm(
                    0.5 * (window_embeddings + eeg_attended + eog_window_embeddings + eog_attended)
                )
                if eeg_gate is not None:
                    outputs["eog_gate"] = eeg_gate
                    outputs["reverse_eog_gate"] = eog_gate
                outputs["eog_attention_weights"] = eeg_weights
                outputs["reverse_attention_weights"] = eog_weights

        pooled, _ = self._pool_sequence(window_embeddings)
        outputs.update(
            {
                "class_logits": self.classifier(pooled),
                "perclos": self.perclos_head(pooled),
            }
        )
        return outputs
