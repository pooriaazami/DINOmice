"""
Behavior Classification Model using DINOv3 features.
Supports two architectures:
1. GRU-based (hierarchical): Patch GRU -> Window GRU -> Classifier
2. Transformer-based: Flatten all patches in window -> Transformer Encoder -> Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import yaml
import math


class PatchGRU(nn.Module):
    """
    GRU to process patch embeddings within a single frame.
    Patches are processed in spatial order (row-major).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Output projection
        direction_factor = 2 if bidirectional else 1
        self.output_proj = nn.Linear(
            hidden_size * direction_factor,
            hidden_size
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (batch, num_patches, input_dim)
        Returns:
            frame_embedding: (batch, hidden_size)
        """
        batch_size, num_patches, _ = patch_embeddings.shape

        # Process through GRU
        gru_out, _ = self.gru(patch_embeddings)  # (batch, num_patches, hidden_size * direction_factor)

        # Take the last hidden state (or mean pooling)
        # Using mean pooling over all patches for robustness
        frame_embedding = gru_out.mean(dim=1)  # (batch, hidden_size * direction_factor)

        # Project and normalize
        frame_embedding = self.output_proj(frame_embedding)
        frame_embedding = self.dropout(frame_embedding)
        frame_embedding = self.layer_norm(frame_embedding)

        return frame_embedding


class WindowGRU(nn.Module):
    """
    GRU to process frame embeddings across a temporal window.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        direction_factor = 2 if bidirectional else 1
        self.output_proj = nn.Linear(
            hidden_size * direction_factor,
            hidden_size
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_embeddings: (batch, window_size, input_dim)
        Returns:
            window_embedding: (batch, hidden_size)
        """
        batch_size, window_size, _ = frame_embeddings.shape

        # Process through GRU
        gru_out, _ = self.gru(frame_embeddings)  # (batch, window_size, hidden_size * direction_factor)

        # Take the last frame's output (for predicting behavior at last frame)
        # Or use mean pooling - using last frame as per requirement
        window_embedding = gru_out[:, -1, :]  # (batch, hidden_size * direction_factor)

        # Project and normalize
        window_embedding = self.output_proj(window_embedding)
        window_embedding = self.dropout(window_embedding)
        window_embedding = self.layer_norm(window_embedding)

        return window_embedding


class BehaviorClassifier(nn.Module):
    """
    Complete behavior classification pipeline.
    """

    def __init__(
        self,
        num_behaviors: int,
        dinov3_dim: int = 768,  # DINOv3 ViT-B/16 dimension
        patch_gru_hidden: int = 256,
        patch_gru_layers: int = 2,
        window_gru_hidden: int = 256,
        window_gru_layers: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        self.num_behaviors = num_behaviors
        self.dinov3_dim = dinov3_dim

        # Patch-level GRU
        self.patch_gru = PatchGRU(
            input_dim=dinov3_dim,
            hidden_size=patch_gru_hidden,
            num_layers=patch_gru_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # Window-level GRU
        self.window_gru = WindowGRU(
            input_dim=patch_gru_hidden,
            hidden_size=window_gru_hidden,
            num_layers=window_gru_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(window_gru_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, num_behaviors)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (batch, window_size, num_patches, dinov3_dim)
            patch_mask: (batch, window_size, num_patches) - boolean mask for valid patches
        Returns:
            logits: (batch, num_behaviors)
        """
        batch_size, window_size, num_patches, _ = patch_embeddings.shape

        # Process each frame's patches through Patch GRU
        frame_embeddings = []
        for t in range(window_size):
            frame_patches = patch_embeddings[:, t, :, :]  # (batch, num_patches, dinov3_dim)

            # Apply mask if provided (set masked patches to zero)
            if patch_mask is not None:
                mask = patch_mask[:, t, :].unsqueeze(-1)  # (batch, num_patches, 1)
                frame_patches = frame_patches * mask.float()

            frame_emb = self.patch_gru(frame_patches)  # (batch, patch_gru_hidden)
            frame_embeddings.append(frame_emb)

        # Stack frame embeddings: (batch, window_size, patch_gru_hidden)
        frame_embeddings = torch.stack(frame_embeddings, dim=1)

        # Process through Window GRU
        window_embedding = self.window_gru(frame_embeddings)  # (batch, window_gru_hidden)

        # Classification
        logits = self.classifier(window_embedding)  # (batch, num_behaviors)

        return logits


def create_model_from_config(config_path: str, num_behaviors: int) -> BehaviorClassifier:
    """Create model from YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})

    return BehaviorClassifier(
        num_behaviors=num_behaviors,
        dinov3_dim=768,  # Fixed for DINOv3 ViT-B/16
        patch_gru_hidden=model_config.get('gru_hidden_size', 256),
        patch_gru_layers=model_config.get('gru_num_layers', 2),
        window_gru_hidden=model_config.get('gru_hidden_size', 256),
        window_gru_layers=model_config.get('gru_num_layers', 2),
        fc_hidden=model_config.get('fc_hidden_size', 128),
        dropout=model_config.get('dropout', 0.2)
    )


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BehaviorTransformer(nn.Module):
    """
    Transformer-based behavior classifier.

    Takes all patch embeddings from all frames in the window,
    concatenates them into a single sequence, and processes
    through a Transformer encoder with a [CLS] token for classification.
    """

    def __init__(
        self,
        num_behaviors: int,
        dinov3_dim: int = 768,
        transformer_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_patches_per_frame: int = 50,
        window_size: int = 16,
        use_cls_token: bool = True,
        pool_type: str = 'cls'  # 'cls' or 'mean'
    ):
        super().__init__()
        self.num_behaviors = num_behaviors
        self.dinov3_dim = dinov3_dim
        self.transformer_dim = transformer_dim
        self.max_patches_per_frame = max_patches_per_frame
        self.window_size = window_size
        self.use_cls_token = use_cls_token
        self.pool_type = pool_type

        # Max sequence length = window_size * max_patches_per_frame + 1 (for CLS token)
        self.max_seq_len = window_size * max_patches_per_frame + (1 if use_cls_token else 0)

        # Input projection from DINOv3 dim to transformer dim
        self.input_proj = nn.Linear(dinov3_dim, transformer_dim)

        # Learnable CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(transformer_dim, max_len=self.max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_behaviors)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (batch, window_size, num_patches, dinov3_dim)
            patch_mask: (batch, window_size, num_patches) - boolean mask for valid patches (True = valid)
        Returns:
            logits: (batch, num_behaviors)
        """
        batch_size, window_size, num_patches, _ = patch_embeddings.shape

        # Flatten window and patch dimensions: (batch, window_size * num_patches, dinov3_dim)
        seq_len = window_size * num_patches
        x = patch_embeddings.view(batch_size, seq_len, self.dinov3_dim)

        # Project to transformer dimension
        x = self.input_proj(x)  # (batch, seq_len, transformer_dim)

        # Prepare mask: (batch, seq_len) - True for valid positions
        if patch_mask is not None:
            mask = patch_mask.view(batch_size, seq_len)  # (batch, seq_len)
        else:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)

        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, transformer_dim)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len + 1, transformer_dim)

            # Extend mask for CLS token (always valid)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # (batch, seq_len + 1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create attention mask for transformer (True = attend, False = masked)
        # Transformer expects src_key_padding_mask: (batch, seq_len) where True = masked
        src_key_padding_mask = ~mask  # Invert: True = padded/masked positions

        # Process through transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (batch, seq_len, transformer_dim)

        # Pool the output
        if self.pool_type == 'cls' and self.use_cls_token:
            # Use CLS token output
            pooled = x[:, 0]  # (batch, transformer_dim)
        elif self.pool_type == 'mean':
            # Mean pooling over valid tokens (excluding CLS if present)
            if self.use_cls_token:
                token_embeddings = x[:, 1:]  # Exclude CLS
                valid_mask = mask[:, 1:].unsqueeze(-1).float()  # (batch, seq_len, 1)
            else:
                token_embeddings = x
                valid_mask = mask.unsqueeze(-1).float()
            pooled = (token_embeddings * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            # Default: use last token (if no CLS)
            pooled = x[:, -1]

        # Classification
        logits = self.classifier(pooled)  # (batch, num_behaviors)

        return logits


def create_model_from_config(config_path: str, num_behaviors: int) -> nn.Module:
    """Create model from YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})

    # Get architecture type (default: 'gru')
    architecture = model_config.get('architecture', 'gru').lower()

    if architecture == 'transformer':
        return BehaviorTransformer(
            num_behaviors=num_behaviors,
            dinov3_dim=768,  # Fixed for DINOv3 ViT-B/16
            transformer_dim=model_config.get('transformer_dim', 512),
            num_layers=model_config.get('transformer_num_layers', 4),
            num_heads=model_config.get('transformer_num_heads', 8),
            ff_dim=model_config.get('transformer_ff_dim', 2048),
            dropout=model_config.get('transformer_dropout', 0.1),
            max_patches_per_frame=model_config.get('max_patches_per_frame', 50),
            window_size=model_config.get('window_size', 16),
            use_cls_token=model_config.get('transformer_use_cls_token', True),
            pool_type=model_config.get('transformer_pool_type', 'cls')
        )
    else:
        # Default: GRU architecture
        return BehaviorClassifier(
            num_behaviors=num_behaviors,
            dinov3_dim=768,  # Fixed for DINOv3 ViT-B/16
            patch_gru_hidden=model_config.get('gru_hidden_size', 256),
            patch_gru_layers=model_config.get('gru_num_layers', 2),
            window_gru_hidden=model_config.get('gru_hidden_size', 256),
            window_gru_layers=model_config.get('gru_num_layers', 2),
            fc_hidden=model_config.get('fc_hidden_size', 128),
            dropout=model_config.get('dropout', 0.2)
        )


if __name__ == "__main__":
    # Test GRU model
    print("Testing GRU model...")
    model = BehaviorClassifier(num_behaviors=10)
    print(f"GRU Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 2
    window_size = 16
    num_patches = 20
    dinov3_dim = 768

    x = torch.randn(batch_size, window_size, num_patches, dinov3_dim)
    mask = torch.ones(batch_size, window_size, num_patches, dtype=torch.bool)

    logits = model(x, mask)
    print(f"GRU Output shape: {logits.shape}")

    # Test Transformer model
    print("\nTesting Transformer model...")
    model_t = BehaviorTransformer(
        num_behaviors=10,
        window_size=window_size,
        max_patches_per_frame=num_patches
    )
    print(f"Transformer Model parameters: {sum(p.numel() for p in model_t.parameters()):,}")

    logits_t = model_t(x, mask)
    print(f"Transformer Output shape: {logits_t.shape}")