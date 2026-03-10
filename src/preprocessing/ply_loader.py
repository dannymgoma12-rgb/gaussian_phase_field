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

    def normalize_positions(self, ply_xyz: np.ndarray,
                            mesh_center: np.ndarray = None,
                            mesh_scale: float = None):
        """Normalize .ply positions using mesh coordinate system.

        If mesh_center/mesh_scale are provided, PLY is normalized using the
        SAME transform as the mesh, ensuring spatial alignment. Otherwise
        falls back to independent normalization (legacy, causes misalignment
        for scenes where the PLY extent differs from the mesh extent).

        Args:
            ply_xyz: (M, 3) raw positions from .ply
            mesh_center: (3,) original mesh bbox center (from mesh_converter)
            mesh_scale: float, original mesh bbox max extent

        Returns:
            normalized: (M, 3) positions in [0,1]^3
            scale_factor: float, uniform scale applied (for splat scale adjustment)
        """
        if mesh_center is not None and mesh_scale is not None:
            # Align to mesh: use the SAME center and scale as mesh_converter
            normalized = (ply_xyz - mesh_center) / mesh_scale * 0.95 + 0.5
            scale_factor = 0.95 / mesh_scale
            print(f"  - Aligning PLY to mesh coordinate system")
            print(f"  - Mesh center: {mesh_center}, mesh scale: {mesh_scale:.4f}")
        else:
            # Legacy: independent normalization (PLY's own bbox)
            bbox_min = ply_xyz.min(axis=0)
            bbox_max = ply_xyz.max(axis=0)
            center = (bbox_min + bbox_max) / 2.0
            extent = (bbox_max - bbox_min).max()
            normalized = (ply_xyz - center) / extent * 0.95 + 0.5
            scale_factor = 0.95 / extent
            print(f"  - WARNING: Independent normalization (no mesh alignment)")

        self._normalization_scale = scale_factor
        self._normalization_center = mesh_center

        print(f"  - Scale factor: {scale_factor:.6f}")
        print(f"  - Normalized range: [{normalized.min(axis=0)}] to "
              f"[{normalized.max(axis=0)}]")

        return normalized, scale_factor

    def filter_foreground(self, ply_xyz_norm: np.ndarray,
                          surface_xyz: np.ndarray,
                          max_distance: float = 0.08):
        """Filter PLY Gaussians to keep only those near the mesh surface.

        Removes background Gaussians (sky, ground, distant objects) that
        are not part of the target object.

        Args:
            ply_xyz_norm: (M, 3) normalized PLY positions
            surface_xyz: (N, 3) mesh surface particle positions in [0,1]^3
            max_distance: maximum distance to nearest surface particle

        Returns:
            fg_mask: (M,) boolean mask — True for foreground Gaussians
            dists: (M,) distance to nearest surface particle
        """
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn_model.fit(surface_xyz)
        dists, _ = nn_model.kneighbors(ply_xyz_norm)
        dists = dists.flatten()

        fg_mask = dists <= max_distance
        n_fg = fg_mask.sum()
        n_bg = (~fg_mask).sum()
        print(f"  [Foreground Filter] {n_fg} kept, {n_bg} removed "
              f"(threshold={max_distance:.3f})")
        print(f"  - Foreground distances: min={dists[fg_mask].min():.4f}, "
              f"max={dists[fg_mask].max():.4f}, "
              f"median={np.median(dists[fg_mask]):.4f}")

        return fg_mask, dists

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

        print(f"  [PLY Match] {len(surface_xyz)} surface particles -> "
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

    def _build_sh_rest(self, ply_data, indices, gaussians, count):
        """Build SH rest features with degree matching."""
        ply_sh = ply_data['sh_degree']
        target_sh = gaussians.max_sh_degree
        target_rest = (target_sh + 1) ** 2 - 1
        ply_rest = (ply_sh + 1) ** 2 - 1
        rest_raw = ply_data['features_rest'][indices]

        if ply_rest == 0 or target_rest == 0:
            rest = np.zeros((count, 3, max(target_rest, 1)))
            if ply_rest > 0 and target_rest > 0:
                c = min(ply_rest, target_rest)
                rest[:, :, :c] = rest_raw[:, :, :c]
        elif ply_rest >= target_rest:
            rest = rest_raw[:, :, :target_rest]
        else:
            rest = np.zeros((count, 3, target_rest))
            rest[:, :, :ply_rest] = rest_raw

        return torch.tensor(rest, dtype=torch.float, device="cuda"
                            ).transpose(1, 2).contiguous()

    def _compute_scale_adjustment(self, scales_raw, scale_factor,
                                  scale_multiplier):
        """Compute scale adjustment with auto scene-size compensation.

        Targets median ~0.001 linear in [0,1]^3 space (~4px at standard camera).
        """
        scale_adj = math.log(scale_factor)
        if scale_multiplier != 1.0:
            scale_adj += math.log(scale_multiplier)
        scales = scales_raw + scale_adj

        # Auto-compensate: target median matches lego reference (~4px)
        TARGET_MEDIAN_LOG = -6.83  # log(0.00108)
        median_s = np.median(scales)
        boost = TARGET_MEDIAN_LOG - median_s
        if abs(boost) > 0.5:
            scales = scales + boost
            print(f"  - Auto scale boost: {boost:+.2f} "
                  f"(median {median_s:.2f} -> {TARGET_MEDIAN_LOG:.2f})")

        # Cap outliers at p97 to prevent background blobs
        p97 = np.percentile(scales, 97)
        cap = min(p97, TARGET_MEDIAN_LOG + 4.0)
        n_capped = (scales > cap).sum()
        if n_capped > 0:
            scales = np.clip(scales, None, cap)
            print(f"  - Scale cap at {cap:.2f}: {n_capped} capped "
                  f"(linear {np.exp(cap):.4f})")

        print(f"  - Scale (log): median={np.median(scales):.2f}, "
              f"max={scales.max():.2f}")

        return scales

    def create_matched_gaussians(
        self,
        gaussians,
        surface_pcd,
        ply_data: dict,
        match_indices: np.ndarray,
        scale_factor: float,
        scale_multiplier: float = 1.0,
    ):
        """Populate GaussianModel with surface positions + matched .ply appearance."""
        N = surface_pcd.points.shape[0]

        xyz = torch.tensor(
            np.asarray(surface_pcd.points), dtype=torch.float, device="cuda")
        features_dc = torch.tensor(
            ply_data['features_dc'][match_indices],
            dtype=torch.float, device="cuda"
        ).transpose(1, 2).contiguous()
        features_rest = self._build_sh_rest(
            ply_data, match_indices, gaussians, N)
        opacity = torch.tensor(
            ply_data['opacity'][match_indices],
            dtype=torch.float, device="cuda")

        scales_adjusted = self._compute_scale_adjustment(
            ply_data['scaling'][match_indices], scale_factor, scale_multiplier)
        scaling = torch.tensor(scales_adjusted, dtype=torch.float, device="cuda")

        rotation = torch.tensor(
            ply_data['rotation'][match_indices],
            dtype=torch.float, device="cuda")

        gaussians._xyz = nn.Parameter(xyz.requires_grad_(True))
        gaussians._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        gaussians._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        gaussians._opacity = nn.Parameter(opacity.requires_grad_(True))
        gaussians._scaling = nn.Parameter(scaling.requires_grad_(True))
        gaussians._rotation = nn.Parameter(rotation.requires_grad_(True))
        gaussians.active_sh_degree = gaussians.max_sh_degree

        normals = np.asarray(surface_pcd.normals)
        if normals is not None and len(normals) > 0:
            gaussians._normal = torch.tensor(
                normals, dtype=torch.float, device="cuda")
        else:
            gaussians._normal = torch.zeros((N, 3), device="cuda")

        gaussians.exposure_mapping = {}
        gaussians.pretrained_exposures = None
        gaussians.max_radii2D = torch.zeros(N, device="cuda")

        print(f"  [Gaussians] Populated {N} matched splats")

    def create_direct_gaussians(
        self,
        gaussians,
        ply_data: dict,
        ply_xyz_normalized: np.ndarray,
        surface_xyz: np.ndarray,
        scale_factor: float,
        scale_multiplier: float = 1.0,
        fg_mask: np.ndarray = None,
    ):
        """Populate GaussianModel with foreground PLY Gaussians (no duplication).

        Uses only foreground Gaussians (filtered by fg_mask). Each PLY Gaussian
        maps to its nearest surface particle for deformation tracking.

        Args:
            gaussians: GaussianModel instance
            ply_data: dict from load_raw_ply()
            ply_xyz_normalized: (M, 3) normalized PLY positions
            surface_xyz: (N, 3) surface particle positions
            scale_factor: from normalize_positions()
            scale_multiplier: optional manual scale adjustment
            fg_mask: (M,) boolean mask from filter_foreground()

        Returns:
            ply_to_surface: (K,) index of nearest surface particle per kept Gaussian
        """
        from sklearn.neighbors import NearestNeighbors

        # Apply foreground filter
        if fg_mask is not None:
            indices = np.where(fg_mask)[0]
            xyz_fg = ply_xyz_normalized[indices]
            print(f"  [Direct PLY] Using {len(indices)} foreground Gaussians")
        else:
            indices = np.arange(len(ply_xyz_normalized))
            xyz_fg = ply_xyz_normalized

        K = len(xyz_fg)

        # Map each PLY Gaussian to nearest surface particle
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn_model.fit(surface_xyz)
        dists, surf_idx = nn_model.kneighbors(xyz_fg)
        ply_to_surface = surf_idx.flatten()

        # Positions
        xyz = torch.tensor(xyz_fg, dtype=torch.float, device="cuda")

        # SH
        features_dc = torch.tensor(
            ply_data['features_dc'][indices],
            dtype=torch.float, device="cuda"
        ).transpose(1, 2).contiguous()
        features_rest = self._build_sh_rest(
            ply_data, indices, gaussians, K)

        # Opacity
        opacity = torch.tensor(
            ply_data['opacity'][indices], dtype=torch.float, device="cuda")

        # Scales with auto-compensation
        scales_adjusted = self._compute_scale_adjustment(
            ply_data['scaling'][indices], scale_factor, scale_multiplier)
        scaling = torch.tensor(
            scales_adjusted, dtype=torch.float, device="cuda")

        # Rotation
        rotation = torch.tensor(
            ply_data['rotation'][indices], dtype=torch.float, device="cuda")

        # Assign
        gaussians._xyz = nn.Parameter(xyz.requires_grad_(True))
        gaussians._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        gaussians._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        gaussians._opacity = nn.Parameter(opacity.requires_grad_(True))
        gaussians._scaling = nn.Parameter(scaling.requires_grad_(True))
        gaussians._rotation = nn.Parameter(rotation.requires_grad_(True))
        gaussians.active_sh_degree = gaussians.max_sh_degree
        gaussians._normal = torch.zeros((K, 3), device="cuda")
        gaussians.exposure_mapping = {}
        gaussians.pretrained_exposures = None
        gaussians.max_radii2D = torch.zeros(K, device="cuda")

        print(f"  [Gaussians] Direct PLY: {K} foreground splats")

        return ply_to_surface
