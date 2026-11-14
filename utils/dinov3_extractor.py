"""
DINOv3 Feature Extraction Utilities.

Based on the reference notebook 2_DINO_sim.ipynb
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import Tuple, List, Optional, Union
import yaml


class DINOv3Extractor:
    """Extract patch-level features from video frames using DINOv3."""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Model configuration
        self.patch_size = self.model.config.patch_size
        self.num_registers = getattr(self.model.config, 'num_register_tokens', 0)
        self.embed_dim = self.model.config.hidden_size

        print(f"DINOv3 loaded on {self.device}")
        print(f"Patch size: {self.patch_size}, Registers: {self.num_registers}, Embed dim: {self.embed_dim}")

    def extract_frame_features(self, frame: np.ndarray, return_cls: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract normalized patch features from a single frame.

        Args:
            frame: H x W x 3 (RGB) numpy array (already converted to 3-channel if needed)
            return_cls: If True, also return the CLS token embedding
        Returns:
            If return_cls=False: patch_features: H_p x W_p x D tensor (normalized)
            If return_cls=True: (patch_features, cls_token) where cls_token is (D,)
        """
        # Handle grayscale input (convert to 3-channel)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Convert to PIL
        image = Image.fromarray(frame.astype(np.uint8))

        # Process - processor handles resizing and normalization
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        # Extract CLS token (first token, index 0)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (1, D)
        cls_token = F.normalize(cls_token, p=2, dim=-1)[0]  # (D,)

        # Remove CLS and register tokens
        patch_features = outputs.last_hidden_state[:, 1 + self.num_registers:, :]

        # Reshape to spatial grid
        H_p = pixel_values.shape[-2] // self.patch_size
        W_p = pixel_values.shape[-1] // self.patch_size

        patch_features = patch_features.reshape(1, H_p, W_p, -1)[0]

        # Normalize features
        patch_features = F.normalize(patch_features, p=2, dim=-1)

        if return_cls:
            return patch_features.cpu(), cls_token.cpu()
        return patch_features.cpu()

    def extract_video_features(
        self,
        video_path: str,
        frame_indices: Optional[List[int]] = None,
        sample_rate: int = 100,
        return_cls: bool = False
    ) -> Union[Tuple[torch.Tensor, List[int]], Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Extract features from video frames.
        Matches notebook: sample every N frames, convert to grayscale then 3-channel RGB.

        Args:
            video_path: Path to video file
            frame_indices: Specific frame indices to extract (optional)
            sample_rate: Sample every N frames (if frame_indices not provided). Default 100 as in notebook.
            return_cls: If True, also return CLS tokens for each frame
        Returns:
            If return_cls=False: (features, extracted_indices)
            If return_cls=True: (features, cls_tokens, extracted_indices)
                features: T x H_p x W_p x D tensor
                cls_tokens: T x D tensor
                extracted_indices: List of frame indices that were extracted
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_indices is None:
            frame_indices = list(range(0, total_frames, sample_rate))

        features_list = []
        cls_tokens_list = []
        extracted_indices = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert BGR to RGB then to Grayscale (as in notebook)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Grayscale -> 3-channel RGB by repeating (as in notebook)
            frame = np.repeat(frame[..., None], 3, axis=-1)

            if return_cls:
                features, cls_token = self.extract_frame_features(frame, return_cls=True)
                features_list.append(features)
                cls_tokens_list.append(cls_token)
            else:
                features = self.extract_frame_features(frame)
                features_list.append(features)
            extracted_indices.append(idx)

        cap.release()

        if features_list:
            features = torch.stack(features_list)
            if return_cls:
                cls_tokens = torch.stack(cls_tokens_list)
                return features, cls_tokens, extracted_indices
            return features, extracted_indices
        else:
            if return_cls:
                return torch.empty(0), torch.empty(0), []
            return torch.empty(0), []


def compute_similarity(
    features: torch.Tensor,
    reference_patch: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between reference patch and all patches in all frames.

    Args:
        features: T x H_p x W_p x D
        reference_patch: D (single patch embedding)
    Returns:
        similarities: T x H_p x W_p
    """
    # features: (T, H_p, W_p, D), reference: (D,)
    # Output: (T, H_p, W_p)
    similarities = torch.einsum('thwd,d->thw', features, reference_patch)
    return similarities


def select_similar_patches(
    features: torch.Tensor,
    reference_patch: torch.Tensor,
    threshold: float = 0.80,
    border: int = 1
) -> Tuple[List[torch.Tensor], List[List[Tuple[int, int]]]]:
    """
    Select patches similar to reference patch with border.

    Args:
        features: T x H_p x W_p x D
        reference_patch: D (reference patch embedding from first frame)
        threshold: Similarity threshold (0-1)
        border: Border size around selected patches
    Returns:
        selected_embeddings: List of (num_selected_patches, D) tensors per frame
        selected_coords: List of [(row, col), ...] per frame
    """
    T, H_p, W_p, D = features.shape

    # Compute similarities for all frames
    similarities = compute_similarity(features, reference_patch)  # T x H_p x W_p

    selected_embeddings = []
    selected_coords = []

    for t in range(T):
        sim = similarities[t]  # H_p x W_p

        # Find patches above threshold
        mask = sim > threshold
        rows, cols = torch.where(mask)

        if len(rows) == 0:
            # No patches found - return empty
            selected_embeddings.append(torch.empty(0, D))
            selected_coords.append([])
            continue

        # Expand selection by border
        expanded_mask = torch.zeros_like(mask, dtype=torch.bool)
        for r, c in zip(rows.tolist(), cols.tolist()):
            r0 = max(0, r - border)
            r1 = min(H_p, r + border + 1)
            c0 = max(0, c - border)
            c1 = min(W_p, c + border + 1)
            expanded_mask[r0:r1, c0:c1] = True

        # Get coordinates of expanded selection
        exp_rows, exp_cols = torch.where(expanded_mask)

        # Sort by spatial order (row-major)
        sort_idx = torch.argsort(exp_rows * W_p + exp_cols)
        exp_rows = exp_rows[sort_idx]
        exp_cols = exp_cols[sort_idx]

        # Extract embeddings in spatial order
        frame_embeddings = features[t, exp_rows, exp_cols, :]  # (num_patches, D)

        coords = list(zip(exp_rows.tolist(), exp_cols.tolist()))

        selected_embeddings.append(frame_embeddings)
        selected_coords.append(coords)

    return selected_embeddings, selected_coords


def pad_patch_sequence(
    patch_embeddings: List[torch.Tensor],
    max_patches: int,
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length patch sequences to fixed length.

    Args:
        patch_embeddings: List of (num_patches, D) tensors
        max_patches: Maximum number of patches
        pad_value: Padding value
    Returns:
        padded: (T, max_patches, D)
        mask: (T, max_patches) boolean mask (True = valid)
    """
    T = len(patch_embeddings)
    D = patch_embeddings[0].shape[-1] if T > 0 and patch_embeddings[0].numel() > 0 else 768

    padded = torch.full((T, max_patches, D), pad_value)
    mask = torch.zeros(T, max_patches, dtype=torch.bool)

    for t, emb in enumerate(patch_embeddings):
        n = min(emb.shape[0], max_patches)
        if n > 0:
            padded[t, :n] = emb[:n]
            mask[t, :n] = True

    return padded, mask


def create_sliding_windows(
    patch_embeddings: torch.Tensor,
    patch_mask: torch.Tensor,
    window_size: int,
    stride: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sliding windows over temporal dimension.

    Args:
        patch_embeddings: (T, max_patches, D)
        patch_mask: (T, max_patches)
        window_size: Number of frames per window
        stride: Stride between windows
    Returns:
        windows: (num_windows, window_size, max_patches, D)
        window_masks: (num_windows, window_size, max_patches)
    """
    T, max_patches, D = patch_embeddings.shape

    windows = []
    window_masks = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows.append(patch_embeddings[start:end])
        window_masks.append(patch_mask[start:end])

    if windows:
        windows = torch.stack(windows)
        window_masks = torch.stack(window_masks)
    else:
        windows = torch.empty(0, window_size, max_patches, D)
        window_masks = torch.empty(0, window_size, max_patches, dtype=torch.bool)

    return windows, window_masks


if __name__ == "__main__":
    # Test feature extraction
    extractor = DINOv3Extractor()

    # Create dummy frame (grayscale -> 3-channel as in notebook)
    dummy_gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    dummy_frame = np.repeat(dummy_gray[..., None], 3, axis=-1)
    features = extractor.extract_frame_features(dummy_frame)
    print(f"Frame features shape: {features.shape}")  # Should be (14, 14, 768)

    # Test similarity computation
    ref_patch = features[7, 7]  # Center patch
    sim = compute_similarity(features.unsqueeze(0), ref_patch)
    print(f"Similarity shape: {sim.shape}")  # Should be (1, 14, 14)