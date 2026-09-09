"""Multi-layer, position-aware DINOv2 anomaly scoring."""

from collections.abc import Sequence

import torch
from anomalib.data import InferenceBatch
from anomalib.models.components.feature_extractors import TimmFeatureExtractor
from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel
from torch.nn import functional as F  # noqa: N812


class EnhancedAnomalyDINOModel(AnomalyDINOModel):
    """DINOv2 patch bank with multi-layer and spatially local density-aware kNN."""

    def __init__(
        self,
        num_neighbours: int = 5,
        encoder_name: str = "vit_small_patch14_dinov2",
        masking: bool = False,
        feature_layers: Sequence[int] = (8, 10, 11),
        position_radius: int = 1,
        spatial_weight: float = 0.05,
        density_neighbours: int = 5,
    ) -> None:
        """Configure the frozen layers and spatial-density kNN scorer.

        Args:
            num_neighbours: Neighbours averaged for each query patch.
            encoder_name: Pretrained timm DINOv2 encoder name.
            masking: Whether to apply AnomalyDINO's PCA foreground mask.
            feature_layers: Zero-based transformer block indices to concatenate.
            position_radius: Maximum row/column offset for candidate patches.
            spatial_weight: Additive penalty per squared patch-grid offset.
            density_neighbours: Neighbours used for normal-density estimation.
        """
        super().__init__(
            num_neighbours=num_neighbours,
            encoder_name=encoder_name,
            masking=masking,
            coreset_subsampling=False,
        )
        if not feature_layers:
            raise ValueError("feature_layers must not be empty")
        if position_radius < 0:
            raise ValueError("position_radius must be non-negative")
        if spatial_weight < 0:
            raise ValueError("spatial_weight must be non-negative")
        if density_neighbours < 1:
            raise ValueError("density_neighbours must be at least 1")

        self.feature_layers = tuple(int(layer) for layer in feature_layers)
        self.layer_names = tuple(f"blocks.{layer}" for layer in self.feature_layers)
        self.position_radius = position_radius
        self.spatial_weight = spatial_weight
        self.density_neighbours = density_neighbours
        self.feature_encoder = TimmFeatureExtractor(
            backbone=encoder_name,
            layers=list(self.layer_names),
            pre_trained=True,
            requires_grad=False,
            output_fmt="NLC",
            return_class_token=False,
            norm=True,
            dynamic_img_size=True,
        )
        self.patch_size = self.feature_encoder.patch_size
        self.embedding_store: list[torch.Tensor] = []
        self.mask_store: list[torch.Tensor] = []
        self.register_buffer("memory_valid", torch.empty(0, dtype=torch.bool))
        self.register_buffer("memory_density", torch.empty(0))
        self.register_buffer("density_reference", torch.tensor(1.0))

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Concatenate raw patch tokens from the selected transformer blocks."""
        outputs = self.feature_encoder(image_tensor)
        return torch.cat([outputs[name] for name in self.layer_names], dim=-1)

    def fit(self) -> None:
        """Finalize the structured bank and estimate normal density at each position."""
        if not self.embedding_store:
            raise ValueError("No embeddings collected. Run model in training mode first.")
        self.memory_bank = torch.cat(self.embedding_store, dim=0)
        self.memory_valid = torch.cat(self.mask_store, dim=0)
        self.embedding_store.clear()
        self.mask_store.clear()

        samples, patches, _ = self.memory_bank.shape
        densities = torch.ones((samples, patches), device=self.memory_bank.device, dtype=self.memory_bank.dtype)
        for patch_index in range(patches):
            valid_indices = torch.nonzero(self.memory_valid[:, patch_index], as_tuple=False).squeeze(1)
            if valid_indices.numel() < 2:
                continue
            features = self.memory_bank[valid_indices, patch_index]
            distances = (1 - features @ features.T).clamp_(0, 2)
            distances.fill_diagonal_(float("inf"))
            k = min(self.density_neighbours, valid_indices.numel() - 1)
            local_density = distances.topk(k=k, largest=False, dim=1).values.mean(dim=1)
            densities[valid_indices, patch_index] = local_density.clamp_min(1e-3)
        self.memory_density = densities
        valid_densities = densities[self.memory_valid]
        self.density_reference = valid_densities.median().clamp_min(1e-3)

    @staticmethod
    def patchcore_image_score(neighbour_scores: torch.Tensor) -> torch.Tensor:
        """Apply PatchCore-style neighborhood confidence to the worst patch."""
        patch_scores = neighbour_scores.mean(dim=-1)
        if neighbour_scores.shape[-1] == 1:
            return patch_scores.amax(dim=1, keepdim=True)
        worst_patch = patch_scores.argmax(dim=1)
        batch_indices = torch.arange(patch_scores.shape[0], device=patch_scores.device)
        worst_neighbours = neighbour_scores[batch_indices, worst_patch]
        confidence = 1 - F.softmax(worst_neighbours, dim=1)[:, 0]
        image_score: torch.Tensor = (confidence * patch_scores[batch_indices, worst_patch]).unsqueeze(1)
        return image_score

    def _score_features(
        self,
        features: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return density-normalized kNN scores for every spatial patch."""
        batch_size, num_patches, _ = features.shape
        height, width = grid_size
        k = min(self.num_neighbours, self.memory_bank.shape[0] * (2 * self.position_radius + 1) ** 2)
        neighbour_scores = torch.zeros((batch_size, num_patches, k), device=features.device, dtype=features.dtype)

        for patch_index in range(num_patches):
            row, column = divmod(patch_index, width)
            nearby = [
                y * width + x
                for y in range(max(0, row - self.position_radius), min(height, row + self.position_radius + 1))
                for x in range(max(0, column - self.position_radius), min(width, column + self.position_radius + 1))
            ]
            candidates = self.memory_bank[:, nearby].reshape(-1, self.memory_bank.shape[-1])
            candidate_valid = self.memory_valid[:, nearby].reshape(-1)
            candidate_density = self.memory_density[:, nearby].reshape(-1)
            candidate_positions = torch.tensor(nearby, device=features.device).repeat(self.memory_bank.shape[0])
            candidates = candidates[candidate_valid]
            candidate_density = candidate_density[candidate_valid]
            candidate_positions = candidate_positions[candidate_valid]
            if candidates.shape[0] < k:
                # Foreground masking can remove every training token in a local
                # window. Fall back to the global valid bank while retaining
                # the spatial penalty, rather than failing or inventing zeros.
                candidates = self.memory_bank.reshape(-1, self.memory_bank.shape[-1])
                candidate_valid = self.memory_valid.reshape(-1)
                candidate_density = self.memory_density.reshape(-1)
                candidate_positions = torch.arange(num_patches, device=features.device).repeat(
                    self.memory_bank.shape[0]
                )
                candidates = candidates[candidate_valid]
                candidate_density = candidate_density[candidate_valid]
                candidate_positions = candidate_positions[candidate_valid]
            if candidates.shape[0] < k:
                raise RuntimeError("Too few valid DINOv2 neighbours after global fallback")

            distances = (1 - features[:, patch_index] @ candidates.T).clamp_(0, 2)
            candidate_rows = torch.div(candidate_positions, width, rounding_mode="floor")
            candidate_cols = candidate_positions.remainder(width)
            spatial_distance = (candidate_rows - row).square() + (candidate_cols - column).square()
            distances = distances + self.spatial_weight * spatial_distance.to(distances.dtype)
            values, indices = distances.topk(k=k, largest=False, dim=1)
            # A bounded quarter-power correction retains raw cosine-distance
            # ordering far better than an unstable direct density ratio.
            density_scale = (self.density_reference / candidate_density[indices].clamp_min(1e-3)).pow(0.25)
            neighbour_scores[:, patch_index] = values * density_scale.clamp(0.5, 2.0)

        return neighbour_scores.mean(dim=-1), neighbour_scores

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Collect structured normal tokens or score test tokens."""
        input_tensor = input_tensor.type(self.memory_bank.dtype)
        batch_size, _, input_height, input_width = input_tensor.shape
        crop_height = input_height % self.patch_size
        crop_width = input_width % self.patch_size
        top, left = crop_height // 2, crop_width // 2
        bottom, right = crop_height - top, crop_width - left
        cropped_height, cropped_width = input_height - crop_height, input_width - crop_width
        if crop_height or crop_width:
            input_tensor = input_tensor[:, :, top : input_height - bottom, left : input_width - right]
        grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)

        features = self.extract_features(input_tensor)
        if self.masking:
            masks_np = self.compute_background_masks(features.detach().cpu().numpy(), grid_size)
            masks = torch.from_numpy(masks_np).to(features.device)
        else:
            masks = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        features = F.normalize(features, p=2, dim=-1)

        if self.training:
            self.embedding_store.append(features)
            self.mask_store.append(masks)
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        if self.memory_bank.numel() == 0:
            raise RuntimeError("Memory bank is empty. Run fit before inference.")

        patch_scores, neighbour_scores = self._score_features(features, grid_size)
        patch_scores = patch_scores.masked_fill(~masks, 0)
        neighbour_scores = neighbour_scores.masked_fill(~masks.unsqueeze(-1), 0)
        image_score = self.patchcore_image_score(neighbour_scores)
        anomaly_map = patch_scores.view(batch_size, 1, *grid_size)
        anomaly_map = self.anomaly_map_generator(anomaly_map, (cropped_height, cropped_width))
        if crop_height or crop_width:
            anomaly_map = F.pad(anomaly_map, (left, right, top, bottom), mode="replicate")
        return InferenceBatch(pred_score=image_score, anomaly_map=anomaly_map)
