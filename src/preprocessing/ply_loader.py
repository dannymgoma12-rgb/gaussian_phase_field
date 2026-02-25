"""
Pre-trained 3DGS .ply loader for the MPM fracture simulation pipeline.

Loads a trained Gaussian Splatting .ply file and maps its photorealistic
appearance (SH colors, opacity, scale, rotation) onto MPM surface particles.
"""

import math
import numpy as np
import torch
from torch import nn


class PretrainedPlyLoader:
    """Load pre-trained 3DGS .ply and map appearance to MPM surface particles."""

    def __init__(self, ply_path: str, sh_degree: int = 3):
        self.ply_path = ply_path
        self.sh_degree = sh_degree
        self._normalization_scale = None
        self._normalization_center = None

    def load_raw_ply(self) -> dict:
        """Parse .ply file into raw numpy arrays.

        Returns:
            dict with keys: xyz, features_dc, features_rest, opacity,
                           scaling, rotation, sh_degree
        """
        from plyfile import PlyData

        print(f"[PLY Loader] Reading {self.ply_path}")
        plydata = PlyData.read(self.ply_path)
        vtx = plydata.elements[0]

        xyz = np.stack([
            np.asarray(vtx["x"]),
            np.asarray(vtx["y"]),
            np.asarray(vtx["z"]),
        ], axis=1)
        M = xyz.shape[0]
        print(f"  - {M} Gaussians loaded")

        # DC SH coefficients
        features_dc = np.zeros((M, 3, 1))
        features_dc[:, 0, 0] = np.asarray(vtx["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(vtx["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(vtx["f_dc_2"])

        # Higher-order SH — auto-detect degree from field count
        extra_f_names = sorted(
            [p.name for p in vtx.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split('_')[-1])
        )
        if len(extra_f_names) > 0:
            ply_sh_degree = int(math.sqrt(len(extra_f_names) // 3 + 1)) - 1
            features_rest = np.zeros((M, len(extra_f_names)))
            for idx, name in enumerate(extra_f_names):
                features_rest[:, idx] = np.asarray(vtx[name])
            features_rest = features_rest.reshape(
                (M, 3, (ply_sh_degree + 1) ** 2 - 1)
            )
        else:
            ply_sh_degree = 0
            features_rest = np.zeros((M, 3, 0))
        print(f"  - SH degree: {ply_sh_degree}")

        # Opacity
        opacities = np.asarray(vtx["opacity"])[..., np.newaxis]

        # Scales (log-space)
        scale_names = sorted(
            [p.name for p in vtx.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split('_')[-1])
        )
        scales = np.stack([np.asarray(vtx[n]) for n in scale_names], axis=1)

        # Rotations (quaternions)
        rot_names = sorted(
            [p.name for p in vtx.properties if p.name.startswith("rot")],
            key=lambda x: int(x.split('_')[-1])
        )
        rots = np.stack([np.asarray(vtx[n]) for n in rot_names], axis=1)

        print(f"  - Position range: [{xyz.min(axis=0)}, {xyz.max(axis=0)}]")

        return {
            'xyz': xyz,
            'features_dc': features_dc,
            'features_rest': features_rest,
            'opacity': opacities,
            'scaling': scales,
            'rotation': rots,
            'sh_degree': ply_sh_degree,
        }

    def normalize_positions(self, ply_xyz: np.ndarray):
        """Normalize .ply positions to [0.025, 0.975]^3 (matching mesh normalization).

        Args:
            ply_xyz: (M, 3) raw positions from .ply

        Returns:
            normalized: (M, 3) positions in [0,1]^3
            scale_factor: float, uniform scale applied (for splat scale adjustment)
        """
        bbox_min = ply_xyz.min(axis=0)
        bbox_max = ply_xyz.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        extent = (bbox_max - bbox_min).max()

        # Match mesh_converter.py normalization: 95% fill in [0,1]^3
        normalized = (ply_xyz - center) / extent * 0.95 + 0.5
        scale_factor = 0.95 / extent

        self._normalization_scale = scale_factor
        self._normalization_center = center

        print(f"  - PLY bbox: [{bbox_min}] to [{bbox_max}]")
        print(f"  - Center: {center}, extent: {extent:.4f}")
        print(f"  - Scale factor: {scale_factor:.6f}")
        print(f"  - Normalized range: [{normalized.min(axis=0)}] to [{normalized.max(axis=0)}]")

        return normalized, scale_factor

    def match_to_surface_particles(
        self,
        ply_xyz_normalized: np.ndarray,
        surface_xyz: np.ndarray,
    ):
        """Match each surface particle to its nearest .ply splat.

        Args:
            ply_xyz_normalized: (M, 3) normalized .ply positions
            surface_xyz: (N, 3) surface particle positions in [0,1]^3

        Returns:
            indices: (N,) indices into ply arrays
            distances: (N,) matching distances
        """
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn.fit(ply_xyz_normalized)
        distances, indices = nn.kneighbors(surface_xyz)
        distances = distances.flatten()
        indices = indices.flatten()

        print(f"  [PLY Match] {len(surface_xyz)} surface particles → "
              f"{len(ply_xyz_normalized)} PLY splats")
        print(f"  - Distance: min={distances.min():.6f}, "
              f"max={distances.max():.6f}, mean={distances.mean():.6f}, "
              f"median={np.median(distances):.6f}")

        far_threshold = 0.05
        n_far = (distances > far_threshold).sum()
        if n_far > 0:
            pct = 100.0 * n_far / len(distances)
            print(f"  - WARNING: {n_far} particles ({pct:.1f}%) "
                  f"have distance > {far_threshold} (poor alignment)")

        return indices, distances

    def create_matched_gaussians(
        self,
        gaussians,
        surface_pcd,
        ply_data: dict,
        match_indices: np.ndarray,
        scale_factor: float,
        scale_multiplier: float = 1.0,
    ):
        """Populate GaussianModel with surface positions + matched .ply appearance.

        Args:
            gaussians: GaussianModel instance (empty, will be populated)
            surface_pcd: BasicPointCloud with surface particle positions
            ply_data: dict from load_raw_ply()
            match_indices: (N,) indices from match_to_surface_particles()
            scale_factor: from normalize_positions()
            scale_multiplier: optional manual scale adjustment
        """
        N = surface_pcd.points.shape[0]

        # Positions: from surface particles (MPM will drive these)
        xyz = torch.tensor(
            np.asarray(surface_pcd.points), dtype=torch.float, device="cuda"
        )

        # SH DC features: from matched .ply splats
        features_dc = torch.tensor(
            ply_data['features_dc'][match_indices],
            dtype=torch.float, device="cuda"
        ).transpose(1, 2).contiguous()  # (N, 3, 1) → (N, 1, 3)

        # SH rest: handle degree mismatch
        ply_sh = ply_data['sh_degree']
        target_sh = gaussians.max_sh_degree
        target_rest_count = (target_sh + 1) ** 2 - 1
        ply_rest_count = (ply_sh + 1) ** 2 - 1

        rest_matched = ply_data['features_rest'][match_indices]  # (N, 3, ply_rest)

        if ply_rest_count == 0 or target_rest_count == 0:
            rest = np.zeros((N, 3, max(target_rest_count, 1)))
            if ply_rest_count > 0 and target_rest_count > 0:
                copy_count = min(ply_rest_count, target_rest_count)
                rest[:, :, :copy_count] = rest_matched[:, :, :copy_count]
        elif ply_rest_count >= target_rest_count:
            rest = rest_matched[:, :, :target_rest_count]
        else:
            rest = np.zeros((N, 3, target_rest_count))
            rest[:, :, :ply_rest_count] = rest_matched

        if ply_sh != target_sh:
            print(f"  [SH Degree] PLY={ply_sh}, config={target_sh} "
                  f"→ {'truncated' if ply_sh > target_sh else 'padded with zeros'}")

        features_rest = torch.tensor(
            rest, dtype=torch.float, device="cuda"
        ).transpose(1, 2).contiguous()  # (N, 3, rest) → (N, rest, 3)

        # Opacity: from .ply (already in inverse-sigmoid / logit space)
        opacity = torch.tensor(
            ply_data['opacity'][match_indices],
            dtype=torch.float, device="cuda"
        )

        # Scale: from .ply, adjusted for coordinate normalization
        scales_raw = ply_data['scaling'][match_indices]  # (N, 3), log-space
        scale_adjustment = math.log(scale_factor)
        if scale_multiplier != 1.0:
            scale_adjustment += math.log(scale_multiplier)
        scales_adjusted = scales_raw + scale_adjustment

        scaling = torch.tensor(
            scales_adjusted, dtype=torch.float, device="cuda"
        )

        # Rotation: from .ply (quaternions, preserved under uniform scaling)
        rotation = torch.tensor(
            ply_data['rotation'][match_indices],
            dtype=torch.float, device="cuda"
        )

        # Assign as nn.Parameters
        gaussians._xyz = nn.Parameter(xyz.requires_grad_(True))
        gaussians._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        gaussians._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        gaussians._opacity = nn.Parameter(opacity.requires_grad_(True))
        gaussians._scaling = nn.Parameter(scaling.requires_grad_(True))
        gaussians._rotation = nn.Parameter(rotation.requires_grad_(True))
        gaussians.active_sh_degree = gaussians.max_sh_degree

        # Normals (needed for impact front-face detection)
        normals = np.asarray(surface_pcd.normals)
        if normals is not None and len(normals) > 0:
            gaussians._normal = torch.tensor(
                normals, dtype=torch.float, device="cuda"
            )
        else:
            gaussians._normal = torch.zeros((N, 3), device="cuda")

        # Safe defaults for exposure (prevents AttributeError)
        gaussians.exposure_mapping = {}
        gaussians.pretrained_exposures = None
        gaussians.max_radii2D = torch.zeros(N, device="cuda")

        print(f"  [Gaussians] Populated {N} splats with pretrained appearance")
        print(f"  - Scale adjustment: {scale_adjustment:.4f} "
              f"(factor={scale_factor:.6f}, multiplier={scale_multiplier:.2f})")
