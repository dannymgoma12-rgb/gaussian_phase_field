"""Gaussian crack visualization via AT2 phase-field damage."""

import torch
from torch import Tensor


class GaussianCrackVisualizer:
    """
    Crack visualization on Gaussian Splats using AT2 phase-field damage.

    Maps c_surface ∈ [0,1] (from AT2 PDE) to Gaussian scale and opacity.
    Polylines drive crack propagation physics; c_surface drives rendering.
    """

    def __init__(
        self,
        damage_threshold: float = 0.3,
        device: str = "cuda",
    ):
        self.damage_threshold = damage_threshold
        self.device = device

        self._original_dc = None
        self._original_rest = None
        self._original_opacity = None
        self._original_scaling = None

        print(f"[GaussianCrackVisualizer] Initialized (AT2 damage mode)")
        print(f"  - Damage threshold: {damage_threshold}")
        print(f"  - Device: {device}")

    @torch.no_grad()
    def _apply_damage_visualization(self, gaussians, c_surface: Tensor):
        """Visualize AT2 damage field on Gaussian Splats.

        Maps c_surface → scale shrinkage + opacity reduction.
        Smooth cubic falloff avoids hard seams at damage boundaries.
        """
        thresh = self.damage_threshold

        # Scale shrinkage: gradual onset from thresh*0.5
        low_thresh = thresh * 0.5
        damaged = c_surface > low_thresh
        if damaged.any():
            t = ((c_surface[damaged] - low_thresh)
                 / (1.0 - low_thresh)).clamp(0.0, 1.0)
            scale_mult = 1.0 - 0.9 * (t ** 2) * (3.0 - 2.0 * t)  # cubic
            gaussians._scaling.data[damaged] += torch.log(
                scale_mult.unsqueeze(1).clamp(min=0.01))

        # Opacity reduction: aggressive fade at high damage
        high_damage = c_surface > thresh
        if high_damage.any():
            t_o = ((c_surface[high_damage] - thresh)
                   / (1.0 - thresh)).clamp(0.0, 1.0)
            opacity_mult = 1.0 - 0.95 * (t_o ** 2)
            cur_prob = torch.sigmoid(gaussians._opacity.data[high_damage])
            new_prob = (cur_prob * opacity_mult.unsqueeze(1)).clamp(1e-6, 1 - 1e-6)
            gaussians._opacity.data[high_damage] = torch.log(
                new_prob / (1.0 - new_prob))

    @torch.no_grad()
    def _apply_deformation_gradient(self, gaussians, F_per_gaussian: Tensor):
        """Apply deformation gradient to Gaussian scale and rotation.

        Polar decomposition: F = R · S
        - R → rotate Gaussian orientation (quaternion multiplication)
        - S → stretch Gaussian scales (log-space addition)

        Args:
            gaussians: 3DGS Gaussian model
            F_per_gaussian: (K, 3, 3) deformation gradient per Gaussian
        """
        K = F_per_gaussian.shape[0]
        device = F_per_gaussian.device

        # Polar decomposition via SVD: F = U * diag(sigma) * Vt
        # R = U * Vt, S = V * diag(sigma) * Vt
        U, sigma, Vt = torch.linalg.svd(F_per_gaussian)

        # Ensure proper rotation (det > 0)
        det_sign = torch.det(U @ Vt).sign().unsqueeze(-1).unsqueeze(-1)
        U_fixed = U.clone()
        U_fixed[:, :, -1] *= det_sign.squeeze(-1)
        sigma_fixed = sigma.clone()
        sigma_fixed[:, -1] *= det_sign.squeeze(-1).squeeze(-1)

        R_mat = U_fixed @ Vt  # (K, 3, 3) rotation matrices

        # Apply stretch to log-scales
        log_stretch = torch.log(sigma_fixed.clamp(min=0.1, max=3.0))  # clamp for stability
        gaussians._scaling.data += log_stretch

        # Convert R to quaternion and multiply with existing rotation
        # Batch rotation matrix to quaternion
        q_rot = self._rotmat_to_quat_batch(R_mat)  # (K, 4) wxyz

        # Hamilton product: q_rot * q_old
        q_old = gaussians._rotation.data  # (K, 4) wxyz
        q_new = self._quat_multiply(q_rot, q_old)
        gaussians._rotation.data = q_new

    @staticmethod
    def _rotmat_to_quat_batch(R: Tensor) -> Tensor:
        """Convert batch of rotation matrices to quaternions (wxyz).

        Args:
            R: (K, 3, 3) rotation matrices
        Returns:
            q: (K, 4) quaternions in wxyz order
        """
        K = R.shape[0]
        q = torch.zeros(K, 4, device=R.device, dtype=R.dtype)

        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        # Case 1: tr > 0
        mask1 = tr > 0
        if mask1.any():
            s = torch.sqrt(tr[mask1] + 1.0) * 2.0
            q[mask1, 0] = 0.25 * s
            q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
            q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
            q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

        # Case 2: R[0,0] largest
        mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        if mask2.any():
            s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0
            q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
            q[mask2, 1] = 0.25 * s
            q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
            q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

        # Case 3: R[1,1] largest
        mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
        if mask3.any():
            s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0
            q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
            q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
            q[mask3, 2] = 0.25 * s
            q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

        # Case 4: R[2,2] largest
        mask4 = ~mask1 & ~mask2 & ~mask3
        if mask4.any():
            s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2.0
            q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
            q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
            q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
            q[mask4, 3] = 0.25 * s

        # Normalize
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
        return q

    @staticmethod
    def _quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
        """Hamilton product q1 * q2 (both in wxyz format)."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    def update_gaussians(
        self,
        gaussians,
        c_surface: Tensor,
        x_world: Tensor,
        preserve_original: bool = True,
        debris_mask: Tensor = None,
        F_per_gaussian: Tensor = None,
    ):
        """Update Gaussian properties each frame.

        Args:
            gaussians:        3DGS Gaussian model
            c_surface:        (N_surf,) AT2 damage values ∈ [0,1]
            x_world:          (N_surf, 3) current surface positions in world space
            preserve_original: cache original Gaussian properties on first call
            debris_mask:      (N_surf,) bool — small fragment Gaussians to hide
            F_per_gaussian:   (N_surf, 3, 3) deformation gradient per Gaussian
        """
        # Cache original properties once
        if preserve_original and self._original_dc is None:
            self._original_dc = gaussians._features_dc.data.clone()
            self._original_rest = gaussians._features_rest.data.clone()
            self._original_opacity = gaussians._opacity.data.clone()
            self._original_scaling = gaussians._scaling.data.clone()
            self._original_rotation = gaussians._rotation.data.clone()

        # Update positions from MPM physics
        gaussians._xyz.data = x_world

        # Restore originals before applying effects
        gaussians._features_dc.data.copy_(self._original_dc)
        gaussians._features_rest.data.copy_(self._original_rest)
        gaussians._opacity.data.copy_(self._original_opacity)
        gaussians._scaling.data.copy_(self._original_scaling)
        gaussians._rotation.data.copy_(self._original_rotation)

        # Apply deformation gradient to scale and rotation
        if F_per_gaussian is not None:
            self._apply_deformation_gradient(gaussians, F_per_gaussian)

        # Debris: mild shrinkage + darkening (keep visible)
        if debris_mask is not None and debris_mask.any():
            gaussians._scaling.data[debris_mask] -= 0.35
            gaussians._features_dc.data[debris_mask] *= 0.7

        # Apply damage visualization
        has_damage = (c_surface is not None
                      and c_surface.max() > self.damage_threshold)
        if has_damage:
            self._apply_damage_visualization(gaussians, c_surface)
