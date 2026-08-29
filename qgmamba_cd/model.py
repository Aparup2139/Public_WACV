from __future__ import annotations

import copy
import math
from contextlib import nullcontext

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def no_amp(tensor: torch.Tensor):
    if tensor.device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        hidden = max(channels // ratio, 8)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(
            self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)), inplace=True))
        )
        return x * weight


class LightFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pre = ConvBNAct(in_channels * 2, out_channels, kernel_size=1, padding=0)
        self.conv = ConvBNAct(out_channels, out_channels)
        self.se = SEBlock(out_channels)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([torch.abs(feature_a - feature_b), feature_a * feature_b], dim=1)
        x = self.se(self.conv(self.pre(x)))
        spatial = self.spatial_attention(
            torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        )
        return x * spatial


class SafeMamba(nn.Module):
    """Notebook's bidirectional PyTorch selective state-space scan."""

    def __init__(self, d_model: int, d_state: int = 16, scan_chunk_size: int = 16):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.scan_chunk_size = scan_chunk_size

    def _scan_one_direction(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        transition = -torch.exp(self.A)
        hidden = torch.zeros(batch, channels, self.A.shape[1], device=x.device, dtype=x.dtype)
        outputs = []
        projected_b = self.B(x)
        transition = transition.unsqueeze(0).unsqueeze(0)
        input_term = delta.unsqueeze(-1) * projected_b.unsqueeze(2)
        for start in range(0, length, self.scan_chunk_size):
            end = min(start + self.scan_chunk_size, length)
            delta_chunk = delta[:, start:end].unsqueeze(-1)
            input_chunk = input_term[:, start:end]
            log_prefix = torch.cumsum(delta_chunk * transition, dim=1).clamp(min=-60.0)
            prefix = torch.exp(log_prefix)
            cumulative = torch.cumsum(input_chunk / prefix, dim=1)
            states = prefix * (hidden.unsqueeze(1) + cumulative)
            outputs.append((states * self.C.weight.unsqueeze(0).unsqueeze(0)).sum(dim=-1))
            hidden = states[:, -1]
        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        with no_amp(x):
            x = x.float()
            x_input, gate = self.in_proj(x).chunk(2, dim=-1)
            x_input = F.silu(x_input)
            delta = F.softplus(self.dt_proj(x_input)).clamp(1.0e-4, 1.0)
            forward = self._scan_one_direction(x_input, delta)
            backward = torch.flip(
                self._scan_one_direction(torch.flip(x_input, [1]), torch.flip(delta, [1])),
                [1],
            )
            output = self.out_proj((forward + backward) * F.silu(gate))
            output = self.norm(x + output)
        return output.to(original_dtype)


class GraphMambaInteractionModule(nn.Module):
    """Swap-invariant dual-order temporal interaction over pooled region tokens."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int = 64,
        scan_chunk_size: int = 16,
        symmetric: bool = True,
    ):
        super().__init__()
        grid = int(math.sqrt(num_nodes))
        if grid * grid != num_nodes:
            raise ValueError("num_nodes must be a perfect square")
        self.num_nodes = num_nodes
        self.grid_size = grid
        self.symmetric = symmetric
        self.proj_a = nn.Conv2d(in_channels, out_channels, 1)
        self.proj_b = nn.Conv2d(in_channels, out_channels, 1)
        self.mamba = SafeMamba(out_channels, scan_chunk_size=scan_chunk_size)
        self.refine = ConvBNAct(out_channels, out_channels)
        self.se = SEBlock(out_channels)
        self.gate = nn.Parameter(torch.zeros(1))
        if symmetric:
            self.interaction_fuse = ConvBNAct(
                out_channels * 4, out_channels, kernel_size=1, padding=0
            )
            self.symmetric_gate = nn.Parameter(torch.tensor(0.1))

    def _pool(self, feature: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            feature, (self.grid_size, self.grid_size)
        ).flatten(2).transpose(1, 2)

    def _ordered_interaction(
        self, nodes_left: torch.Tensor, nodes_right: torch.Tensor
    ) -> torch.Tensor:
        """Process one ordering, recover both partitions, then fuse symmetrically."""
        processed = self.mamba(torch.cat([nodes_left, nodes_right], dim=1))
        processed_left, processed_right = processed.split(self.num_nodes, dim=1)
        interaction = torch.cat(
            [
                torch.abs(processed_left - processed_right),
                processed_left * processed_right,
                0.5 * (processed_left + processed_right),
                torch.abs(nodes_left - nodes_right),
            ],
            dim=-1,
        )
        batch, _, _ = interaction.shape
        interaction = interaction.transpose(1, 2).reshape(
            batch, -1, self.grid_size, self.grid_size
        )
        return self.interaction_fuse(interaction)

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        if not self.symmetric:
            query_a = self.proj_a(feature_a)
            query_b = self.proj_b(feature_b)
            batch, channels, height, width = query_a.shape
            processed = self.mamba(torch.cat([self._pool(query_a), self._pool(query_b)], dim=1))
            output_b = processed[:, self.num_nodes :, :].transpose(1, 2).reshape(
                batch, channels, self.grid_size, self.grid_size
            )
            output_b = F.interpolate(
                output_b, size=(height, width), mode="bilinear", align_corners=False
            )
            baseline = torch.abs(query_a - query_b)
            return baseline + self.gate * (self.se(self.refine(output_b)) - baseline)
        # Evaluate both temporal assignments with shared Mamba weights. Swapping
        # inputs only swaps these two terms, so their average is invariant.
        query_aa = self.proj_a(feature_a)
        query_bb = self.proj_b(feature_b)
        query_ab = self.proj_a(feature_b)
        query_ba = self.proj_b(feature_a)
        _, _, height, width = query_aa.shape
        interaction_ab = self._ordered_interaction(self._pool(query_aa), self._pool(query_bb))
        interaction_ba = self._ordered_interaction(self._pool(query_ab), self._pool(query_ba))
        interaction = 0.5 * (interaction_ab + interaction_ba)
        interaction = F.interpolate(
            interaction, size=(height, width), mode="bilinear", align_corners=False
        )
        baseline = 0.5 * (
            torch.abs(query_aa - query_bb) + torch.abs(query_ab - query_ba)
        )
        residual = self.se(self.refine(interaction))
        return baseline + torch.tanh(self.symmetric_gate) * residual + 0.0 * self.gate


class OrthogonalFeatureDisentanglement(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        if channels % 2:
            raise ValueError("OrthogonalFeatureDisentanglement requires an even channel count")
        self.pairs = channels // 2
        self.theta = nn.Parameter(torch.randn(1, self.pairs, 1, 1))
        self.shared_conv = ConvBNAct(channels, channels)
        self.change_conv = ConvBNAct(channels, channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared_conv(x)
        change = self.change_conv(x)
        batch, channels, height, width = shared.shape
        shared = shared.reshape(batch, self.pairs, 2, height, width)
        change = change.reshape(batch, self.pairs, 2, height, width)
        cosine = torch.cos(self.theta)
        sine = torch.sin(self.theta)
        shared_0 = shared[:, :, 0] * cosine - change[:, :, 0] * sine
        change_0 = shared[:, :, 0] * sine + change[:, :, 0] * cosine
        shared_1 = shared[:, :, 1] * cosine - change[:, :, 1] * sine
        change_1 = shared[:, :, 1] * sine + change[:, :, 1] * cosine
        output_shared = torch.stack([shared_0, shared_1], dim=2).reshape(batch, channels, height, width)
        output_change = torch.stack([change_0, change_1], dim=2).reshape(batch, channels, height, width)
        theta_flat = self.theta.reshape(-1)
        orthogonal_loss = 1.0 / (theta_flat.var() + 0.1)

        # Paper-style OFD regularizers (BMD-CD Eq. 4), computed on the
        # disentangled outputs with spatial positions playing the token role.
        # gram_loss = ||S_c^T S_i||_F^2 after channel-wise l2 normalization
        # (mean-reduced over the C x C Gram entries so the weight stays
        # resolution- and width-independent). change_std is the per-channel
        # standard deviation of S_c over the batch and token axes; the hinge
        # against the margin tau is applied in FullLoss so tau remains a loss
        # hyperparameter. Both are weighted by LossConfig.orthogonal_gram and
        # LossConfig.variance, which default to 0.0, so every existing config
        # and checkpoint behaves exactly as before.
        flat_change = output_change.flatten(2)
        flat_shared = output_shared.flatten(2)
        norm_change = F.normalize(flat_change.float(), dim=2, eps=1.0e-6)
        norm_shared = F.normalize(flat_shared.float(), dim=2, eps=1.0e-6)
        gram = torch.bmm(norm_change, norm_shared.transpose(1, 2))
        gram_loss = gram.pow(2).mean()
        change_std = torch.sqrt(
            flat_change.float().transpose(0, 1).reshape(channels, -1).var(dim=1, unbiased=False)
            + 1.0e-6
        )
        return output_shared, output_change, orthogonal_loss, gram_loss, change_std


class TemporalRefinementGate(nn.Module):
    def __init__(self, in_channels: int, fusion_channels: int):
        super().__init__()
        self.diff_proj = nn.Conv2d(in_channels, fusion_channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, 1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        difference = torch.abs(self.diff_proj(feature_a) - self.diff_proj(feature_b))
        gate = self.gate(torch.cat([fused, difference], dim=1))
        return fused * gate + difference * (1.0 - gate)


class SpatialRefinementHead(nn.Module):
    """Small gated multi-dilation residual for boundary and roof detail."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                    nn.Conv2d(channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.fuse(torch.cat([branch(feature) for branch in self.branches], dim=1))
        return torch.tanh(self.gate) * residual


class MultiScaleFusionDecoder(nn.Module):
    def __init__(
        self,
        input_channels: list[int],
        decoder_channels: int = 128,
        refinement_enabled: bool = False,
        refinement_channels: int = 64,
        edge_fusion_enabled: bool = True,
    ):
        super().__init__()
        self.edge_fusion_enabled = edge_fusion_enabled
        self.laterals = nn.ModuleList([nn.Conv2d(channels, decoder_channels, 1) for channels in input_channels])
        self.up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(decoder_channels * 2, decoder_channels),
                    ConvBNAct(decoder_channels, decoder_channels),
                )
                for _ in range(len(input_channels) - 1)
            ]
        )
        self.scale_mlp = nn.Sequential(
            nn.Linear(decoder_channels * len(input_channels), decoder_channels),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_channels, len(input_channels)),
        )
        self.head_features = ConvBNAct(decoder_channels, 64)
        self.head_logit = nn.Conv2d(64, 1, 1)
        self.condition_projection = nn.Conv2d(64, 8, 1)
        self.deep_head_1 = nn.Conv2d(decoder_channels, 1, 1)
        self.deep_head_2 = nn.Conv2d(decoder_channels, 1, 1)
        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1)
        )
        self.refinement = SpatialRefinementHead(refinement_channels) if refinement_enabled else None

    def forward(self, features: list[torch.Tensor], target_size: tuple[int, int]):
        laterals = [layer(feature) for layer, feature in zip(self.laterals, features)]
        reference_size = laterals[0].shape[-2:]
        aligned = [
            F.interpolate(feature, size=reference_size, mode="bilinear", align_corners=False)
            if feature.shape[-2:] != reference_size
            else feature
            for feature in laterals
        ]
        pooled = torch.cat([F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in aligned], dim=1)
        weights = torch.softmax(self.scale_mlp(pooled), dim=1)
        weighted = [feature * weights[:, index : index + 1, None, None] for index, feature in enumerate(laterals)]
        x = weighted[3]
        deep_1 = self.deep_head_1(x)
        x = self.up_blocks[0](
            torch.cat([F.interpolate(x, size=weighted[2].shape[-2:], mode="bilinear", align_corners=False), weighted[2]], dim=1)
        )
        deep_2 = self.deep_head_2(x)
        x = self.up_blocks[1](
            torch.cat([F.interpolate(x, size=weighted[1].shape[-2:], mode="bilinear", align_corners=False), weighted[1]], dim=1)
        )
        x = self.up_blocks[2](
            torch.cat([F.interpolate(x, size=weighted[0].shape[-2:], mode="bilinear", align_corners=False), weighted[0]], dim=1)
        )
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        feature = self.head_features(x)
        edge_logits = self.edge_head(feature)
        edge_modulated = (
            feature * (1.0 + torch.sigmoid(edge_logits))
            if self.edge_fusion_enabled
            else feature
        )
        logits = self.head_logit(edge_modulated)
        if self.refinement is not None:
            logits = logits + self.refinement(edge_modulated)
        condition = self.condition_projection(edge_modulated)
        deep_1 = F.interpolate(deep_1, size=target_size, mode="bilinear", align_corners=False)
        deep_2 = F.interpolate(deep_2, size=target_size, mode="bilinear", align_corners=False)
        return logits, condition, (deep_1, deep_2), edge_logits


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    cumulative = torch.cos(((x / timesteps) + offset) / (1 + offset) * math.pi / 2) ** 2
    cumulative = cumulative / cumulative[0]
    return torch.clamp(1 - cumulative[1:] / cumulative[:-1], 0, 0.999)


class TimestepEmbedding(nn.Module):
    def __init__(self, dimensions: int = 128):
        super().__init__()
        self.dimensions = dimensions
        self.mlp = nn.Sequential(
            nn.Linear(dimensions, dimensions * 4), nn.SiLU(), nn.Linear(dimensions * 4, dimensions)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimensions // 2
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device) * -(math.log(10000) / (half - 1))
        )
        embedding = timesteps.float()[:, None] * frequencies[None, :]
        return self.mlp(torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1))


class DiffusionResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dimensions: int = 128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.time_projection = nn.Linear(time_dimensions, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x) + self.time_projection(time_embedding)[:, :, None, None]
        return self.conv2(hidden) + self.skip(x)


class DiffusionUNet(nn.Module):
    def __init__(self, input_channels: int, condition_channels: int, channels: list[int], output_channels: int):
        super().__init__()
        c0, c1, c2 = channels
        self.time_embedding = TimestepEmbedding(128)
        self.input_block = DiffusionResidualBlock(input_channels + condition_channels, c0)
        self.down_1 = nn.Sequential(nn.Conv2d(c0, c0, 3, stride=2, padding=1), nn.SiLU())
        self.block_1 = DiffusionResidualBlock(c0, c1)
        self.down_2 = nn.Sequential(nn.Conv2d(c1, c1, 3, stride=2, padding=1), nn.SiLU())
        self.block_2 = DiffusionResidualBlock(c1, c2)
        self.middle = DiffusionResidualBlock(c2, c2)
        self.up_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_block_2 = DiffusionResidualBlock(c2 + c1, c1)
        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_block_1 = DiffusionResidualBlock(c1 + c0, c0)
        self.output = nn.Conv2d(c0, output_channels, 1)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        level_0 = self.input_block(torch.cat([noisy, condition], dim=1), time_embedding)
        level_1 = self.block_1(self.down_1(level_0), time_embedding)
        level_2 = self.block_2(self.down_2(level_1), time_embedding)
        hidden = self.middle(level_2, time_embedding)
        hidden = self.up_block_2(torch.cat([self.up_2(hidden), level_1], dim=1), time_embedding)
        hidden = self.up_block_1(torch.cat([self.up_1(hidden), level_0], dim=1), time_embedding)
        return self.output(hidden)


class LegacyDiffusionDecoder(nn.Module):
    """Original notebook-compatible auxiliary diffusion path."""

    def __init__(self, feature_channels: int, timesteps: int, inference_steps: int, channels: list[int], use_ddim: bool):
        super().__init__()
        self.timesteps = timesteps
        self.inference_steps = inference_steps
        self.use_ddim = use_ddim
        self.unet = DiffusionUNet(feature_channels, 1, channels, feature_channels)
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.refine_projection = nn.Conv2d(feature_channels, 1, 1)

    def train_step(self, coarse_logits, condition_features, ground_truth):
        with no_amp(condition_features):
            condition_features = condition_features.float()
            coarse_logits = coarse_logits.float()
            unchanged_mask = F.interpolate(
                (1.0 - ground_truth.float()).detach(),
                size=condition_features.shape[-2:], mode="nearest",
            )
            masked_features = condition_features * unchanged_mask + condition_features.detach() * (1.0 - unchanged_mask)
            timesteps = torch.randint(
                0, self.timesteps, (ground_truth.shape[0],), device=ground_truth.device
            )
            noise = torch.randn_like(masked_features)
            sqrt_alpha = self.sqrt_alpha_bar[timesteps][:, None, None, None]
            sqrt_one_minus = self.sqrt_one_minus_alpha_bar[timesteps][:, None, None, None]
            noisy = sqrt_alpha * masked_features + sqrt_one_minus * noise
            coarse_condition = F.interpolate(
                torch.sigmoid(coarse_logits).detach(), size=noisy.shape[-2:], mode="bilinear", align_corners=False
            )
            predicted_noise = self.unet(noisy, timesteps, coarse_condition)
            diffusion_loss = F.mse_loss(predicted_noise * unchanged_mask, noise * unchanged_mask)
            projected = F.interpolate(
                self.refine_projection(masked_features), size=coarse_logits.shape[-2:], mode="bilinear", align_corners=False
            )
            diffusion_loss = diffusion_loss + 0.1 * F.mse_loss(projected, coarse_logits.detach())
        return coarse_logits, diffusion_loss

    @torch.no_grad()
    def sample(self, coarse_logits, condition_features):
        return coarse_logits


class StochasticDiffusionDecoder(nn.Module):
    """Conditional logit-space diffusion with gated residual prediction."""

    def __init__(
        self,
        feature_channels: int,
        timesteps: int,
        inference_steps: int,
        channels: list[int],
        use_ddim: bool,
        start_timestep: int,
        noise_scale: float,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.inference_steps = inference_steps
        self.use_ddim = use_ddim
        self.start_timestep = min(max(int(start_timestep), 1), timesteps - 1)
        self.noise_scale = float(noise_scale)
        # Noisy logits are the signal; decoder features and coarse logits are
        # the conditioning channels.
        self.unet = DiffusionUNet(1, feature_channels + 1, channels, 1)
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.refinement_gate = nn.Parameter(torch.tensor(-1.5))

    @staticmethod
    def _target_logits(ground_truth: torch.Tensor) -> torch.Tensor:
        probability = ground_truth.float() * 0.98 + 0.01
        return torch.logit(probability)

    @staticmethod
    def _resize_like(feature: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if feature.shape[-2:] == reference.shape[-2:]:
            return feature
        return F.interpolate(feature, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def _condition(self, coarse_logits: torch.Tensor, condition_features: torch.Tensor) -> torch.Tensor:
        condition_features = self._resize_like(condition_features.float(), coarse_logits)
        return torch.cat([condition_features, coarse_logits.float()], dim=1)

    def train_step(self, coarse_logits: torch.Tensor, condition_features: torch.Tensor, ground_truth: torch.Tensor):
        with no_amp(condition_features):
            condition_features = condition_features.float()
            coarse_logits = coarse_logits.float()
            ground_truth = ground_truth.float()
            target_logits = self._target_logits(ground_truth)
            timesteps = torch.randint(
                0, self.start_timestep + 1,
                (ground_truth.shape[0],), device=ground_truth.device,
            )
            noise = torch.randn_like(target_logits)
            sqrt_alpha = self.sqrt_alpha_bar[timesteps][:, None, None, None]
            sqrt_one_minus = self.sqrt_one_minus_alpha_bar[timesteps][:, None, None, None]
            noisy_logits = sqrt_alpha * target_logits + sqrt_one_minus * noise
            predicted_noise = self.unet(
                noisy_logits, timesteps, self._condition(coarse_logits, condition_features)
            )
            predicted_clean = (
                noisy_logits - sqrt_one_minus * predicted_noise
            ) / sqrt_alpha.clamp_min(1.0e-6)
            predicted_clean = predicted_clean.clamp(-8.0, 8.0)
            gate = torch.sigmoid(self.refinement_gate)
            refined_logits = coarse_logits + gate * (predicted_clean - coarse_logits)
            diffusion_loss = F.mse_loss(predicted_noise, noise)
        return refined_logits, diffusion_loss

    @torch.no_grad()
    def sample(self, coarse_logits: torch.Tensor, condition_features: torch.Tensor) -> torch.Tensor:
        if not self.use_ddim:
            return coarse_logits
        with no_amp(condition_features):
            coarse_logits = coarse_logits.float()
            condition = self._condition(coarse_logits, condition_features.float())
            start_alpha = self.alpha_bar[self.start_timestep]
            sample = (
                torch.sqrt(start_alpha) * coarse_logits
                + self.noise_scale * torch.sqrt(1.0 - start_alpha) * torch.randn_like(coarse_logits)
            )
            timesteps = torch.linspace(
                self.start_timestep, 0, self.inference_steps, device=sample.device
            ).round().long().unique_consecutive().tolist()
            for index, timestep in enumerate(timesteps):
                time_tensor = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
                predicted_noise = self.unet(sample, time_tensor, condition)
                alpha_bar = self.alpha_bar[timestep]
                predicted_clean = (sample - torch.sqrt(1 - alpha_bar) * predicted_noise) / (torch.sqrt(alpha_bar) + 1.0e-8)
                predicted_clean = predicted_clean.clamp(-8.0, 8.0)
                if index < len(timesteps) - 1:
                    next_alpha = self.alpha_bar[timesteps[index + 1]]
                    sample = torch.sqrt(next_alpha) * predicted_clean + torch.sqrt(1 - next_alpha) * predicted_noise
                else:
                    sample = predicted_clean
            gate = torch.sigmoid(self.refinement_gate)
            return coarse_logits + gate * (sample - coarse_logits)


class QGMambaDiffCD(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_channels_last = False
        candidates = [config.encoder_name]
        if config.allow_encoder_fallback:
            candidates = list(
                dict.fromkeys(
                    candidates
                    + [
                        config.encoder_name + ".ms_in1k",
                        "swin_tiny_patch4_window7_224.ms_in1k",
                        "swin_tiny_patch4_window7_224",
                        "convnext_tiny.fb_in1k",
                        "convnext_tiny",
                    ]
                )
            )
        errors = []
        for name in candidates:
            try:
                self.encoder = timm.create_model(
                    name,
                    pretrained=config.pretrained,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                    in_chans=3,
                    img_size=config.input_size,
                )
                self.encoder_name = name
                break
            except Exception as exc:  # preserve fallback while retaining diagnostics
                errors.append(f"{name}: {exc}")
        else:
            raise RuntimeError("No encoder candidate could be created:\n" + "\n".join(errors))
        c1, c2, c3, c4 = self.encoder.feature_info.channels()
        f1, f2, f3, f4 = 64, 128, 192, 256
        self.fuse1 = LightFusion(c1, f1)
        self.fuse2 = LightFusion(c2, f2)
        self.fuse3 = GraphMambaInteractionModule(
            c3, f3, scan_chunk_size=config.scan_chunk_size,
            symmetric=config.symmetric_temporal_enabled,
        )
        self.fuse4 = GraphMambaInteractionModule(
            c4, f4, scan_chunk_size=config.scan_chunk_size,
            symmetric=config.symmetric_temporal_enabled,
        )
        self.temporal_gate3 = TemporalRefinementGate(c3, f3)
        self.temporal_gate4 = TemporalRefinementGate(c4, f4)
        self.unitary3 = OrthogonalFeatureDisentanglement(f3)
        self.unitary4 = OrthogonalFeatureDisentanglement(f4)
        self.decoder = MultiScaleFusionDecoder(
            [f1, f2, f3, f4],
            decoder_channels=128,
            refinement_enabled=config.refinement_enabled,
            refinement_channels=config.refinement_channels,
            edge_fusion_enabled=config.edge_fusion_enabled,
        )
        self.auxiliary_head = nn.Conv2d(f4, 1, 1)
        if config.corrected_diffusion_enabled:
            self.diffusion = StochasticDiffusionDecoder(
                feature_channels=8,
                timesteps=config.diffusion_timesteps,
                inference_steps=config.diffusion_infer_steps,
                channels=config.diffusion_channels,
                use_ddim=config.use_ddim_at_inference,
                start_timestep=config.diffusion_start_timestep,
                noise_scale=config.diffusion_noise_scale,
            )
        else:
            self.diffusion = LegacyDiffusionDecoder(
                feature_channels=8,
                timesteps=config.diffusion_timesteps,
                inference_steps=config.diffusion_infer_steps,
                channels=config.diffusion_channels,
                use_ddim=config.use_ddim_at_inference,
            )
        self.diffusion_enabled = config.diffusion_enabled
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
            persistent=False,
        )

    def _fix_format(self, feature: torch.Tensor, channels: int) -> torch.Tensor:
        if feature.shape[-1] == channels and feature.shape[1] != channels:
            feature = feature.permute(0, 3, 1, 2)
        if self.use_channels_last and feature.device.type == "cuda":
            return feature.contiguous(memory_format=torch.channels_last)
        return feature.contiguous()

    def _encode_pair(self, image_a: torch.Tensor, image_b: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        channels = self.encoder.feature_info.channels()
        combined = torch.cat([image_a, image_b], dim=0)
        features_a = []
        features_b = []
        batch_size = image_a.shape[0]
        for feature, channel in zip(self.encoder(combined), channels):
            feature = self._fix_format(feature, channel)
            feature_a, feature_b = feature.split(batch_size, dim=0)
            features_a.append(feature_a)
            features_b.append(feature_b)
        return features_a, features_b

    def _sobel_edges(self, mask: torch.Tensor) -> torch.Tensor:
        return (
            F.conv2d(mask.float(), self.sobel_x, padding=1).abs()
            + F.conv2d(mask.float(), self.sobel_y, padding=1).abs()
        ).clamp(0, 1)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor, ground_truth: torch.Tensor | None = None):
        (f1a, f2a, f3a, f4a), (f1b, f2b, f3b, f4b) = self._encode_pair(image_a, image_b)
        p1 = self.fuse1(f1a, f1b)
        p2 = self.fuse2(f2a, f2b)
        p3 = self.temporal_gate3(self.fuse3(f3a, f3b), f3a, f3b)
        p4 = self.temporal_gate4(self.fuse4(f4a, f4b), f4a, f4b)
        _, change3, orthogonal3, gram3, std3 = self.unitary3(p3)
        _, change4, orthogonal4, gram4, std4 = self.unitary4(p4)
        orthogonal_loss = orthogonal3 + orthogonal4
        gram_loss = gram3 + gram4
        change_std = torch.cat([std3, std4], dim=0)
        if self.training and ground_truth is not None:
            unchanged = F.interpolate(1.0 - ground_truth, size=f1a.shape[-2:], mode="nearest")
            orthogonal_loss = orthogonal_loss + 0.1 * F.l1_loss(f1a * unchanged, f1b * unchanged)
        target_size = image_a.shape[-2:]
        coarse, condition, deep_auxiliary, edge_logits = self.decoder(
            [p1, p2, change3, change4], target_size
        )
        p4_auxiliary = F.interpolate(
            self.auxiliary_head(change4), size=target_size, mode="bilinear", align_corners=False
        )
        auxiliary = (p4_auxiliary, deep_auxiliary[0], deep_auxiliary[1])
        if self.training and ground_truth is not None:
            if self.diffusion_enabled:
                refined, diffusion_loss = self.diffusion.train_step(coarse, condition, ground_truth)
            else:
                refined, diffusion_loss = coarse, coarse.sum() * 0.0
            return (
                refined,
                coarse,
                auxiliary,
                edge_logits,
                diffusion_loss,
                orthogonal_loss,
                gram_loss,
                change_std,
            )
        refined = self.diffusion.sample(coarse, condition) if self.diffusion_enabled else coarse
        return (refined,)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.ema.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module, decay: float | None = None) -> None:
        decay = self.decay if decay is None else decay
        source = model.state_dict()
        for name, value in self.ema.state_dict().items():
            current = source[name].detach()
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(current, alpha=1.0 - decay)
            else:
                value.copy_(current)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.ema.load_state_dict(state_dict)
