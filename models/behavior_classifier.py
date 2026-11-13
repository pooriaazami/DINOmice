"""
Behavior Classification Model using DINOv3 features and GRU architecture.

Pipeline:
1. DINOv3 extracts patch features for each frame
2. For each frame, select patches similar to reference patch (similarity > threshold)
3. Add border around selected patches
4. Feed patch embeddings (in spatial order) to Patch GRU
5. Process window of W frames
6. Feed window embeddings to Window GRU
7. Final embedding -> Fully Connected layer for multi-label classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import yaml


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


if __name__ == "__main__":
    # Test model creation
    model = BehaviorClassifier(num_behaviors=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 2
    window_size = 16
    num_patches = 20  # Variable number of patches per frame
    dinov3_dim = 768

    x = torch.randn(batch_size, window_size, num_patches, dinov3_dim)
    mask = torch.ones(batch_size, window_size, num_patches, dtype=torch.bool)

    logits = model(x, mask)
    print(f"Output shape: {logits.shape}")  # Should be (2, 10)