import torch
import torch.nn as nn

# This file originates from the public repository `EndlessSora/focal-frequency-loss`.
# It implements the FocalFrequencyLoss class described in the ICCV 2021 paper
# “Focal Frequency Loss for Image Reconstruction and Synthesis” by Liming Jiang,
# Bo Dai, Wayne Wu and Chen Change Loy【194955270957107†L0-L21】.  The original
# repository is licensed under the MIT License, which permits reuse and
# modification provided that the copyright notice is preserved【194955270957107†L0-L21】.

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split("+")[0].split("."))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class FocalFrequencyLoss(nn.Module):
    """Compute the Focal Frequency Loss between a reconstructed image and its
    ground truth target.

    This loss operates in the frequency domain by transforming each input into
    its 2‑2 discrete Fourier transform (DFT).  A dynamic spectrum weight
    matrix—computed from the squared Euclidean distance between the predicted
    and target spectra—emphasizes frequency components that are hard to
    reconstruct while down‑weighting components that are easy【7698168858157†L9-L23】.

    The implementation here is adapted from the `EndlessSora/focal-frequency-loss`
    repository and includes several optional arguments controlling the
    computation:

    Args:
        loss_weight (float): Weight applied to the final loss value.  Default: 1.0.
        alpha (float): Scaling factor of the spectrum weight matrix.  Default: 1.0.
        patch_factor (int): Number of patches per dimension to split the input
            image into before taking its Fourier transform.  A value of 1
            computes the loss over the entire image; larger values compute the
            loss over image patches【7698168858157†L26-L40】.
        ave_spectrum (bool): If True, compute the average spectrum across the
            batch before computing the loss.  Default: False【7698168858157†L107-L110】.
        log_matrix (bool): If True, apply a logarithm to the spectrum weight
            matrix.  Default: False【7698168858157†L69-L71】.
        batch_matrix (bool): If True, normalize the spectrum weight matrix using
            batch statistics rather than per‑sample statistics.  Default: False【7698168858157†L73-L79】.

    Returns:
        A scalar tensor representing the focal frequency loss between the
        prediction and target.【7698168858157†L95-L113】
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.alpha = float(alpha)
        self.patch_factor = int(patch_factor)
        self.ave_spectrum = bool(ave_spectrum)
        self.log_matrix = bool(log_matrix)
        self.batch_matrix = bool(batch_matrix)

    def tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an image tensor to its frequency representation.

        The tensor is first split into patches along the spatial dimensions if
        ``patch_factor`` > 1.  Then a 2‑2 DFT is performed on each patch.  For
        PyTorch ≥ 1.8, the new :mod:`torch.fft` API is used; otherwise, the
        legacy :func:`torch.rfft` function is employed【7698168858157†L35-L57】.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Frequency representation of shape
            (N, patch_factor², C, H/patch_factor, W/patch_factor, 2), where
            the last dimension holds real and imaginary parts.
        """
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        # Ensure the image size can be evenly divided into patches
        assert (
            h % patch_factor == 0 and w % patch_factor == 0
        ), "Patch factor should be divisible by image height and width"

        # Split into patches
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch = x[:, :, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w]
                patch_list.append(patch)
        # Stack into (N, P, C, H_patch, W_patch)
        y = torch.stack(patch_list, dim=1)

        # Perform 2‑2 DFT
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm="ortho")
            # Separate real and imaginary parts
            freq = torch.stack([freq.real, freq.imag], dim=-1)
        else:
            # Fallback for legacy PyTorch versions
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(
        self,
        recon_freq: torch.Tensor,
        real_freq: torch.Tensor,
        matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the focal frequency loss given frequency domain tensors.

        Args:
            recon_freq (torch.Tensor): Frequency representation of the
                reconstructed/predicted image.
            real_freq (torch.Tensor): Frequency representation of the ground
                truth image.
            matrix (torch.Tensor or None): Optional precomputed spectrum
                weight matrix.  If ``None``, the matrix is computed dynamically
                from the Euclidean distance between ``recon_freq`` and ``real_freq``【7698168858157†L59-L81】.

        Returns:
            torch.Tensor: Scalar tensor containing the loss.
        """
        # Spectrum weight matrix
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            # Compute dynamic spectrum weight matrix based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            # Each frequency component has a real and imaginary part
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha
            # Optionally apply logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)
            # Normalize the matrix either across the batch or per sample
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                # Normalize by maximum over spatial dimensions and patch dimension
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
            # Clamp values to [0,1] and handle NaNs
            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()
        # Check range
        assert (
            weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1
        ), (
            f"Spectrum weight matrix values should be in [0,1], got Min: {weight_matrix.min().item()} "
            f"Max: {weight_matrix.max().item()}"
        )
        # Compute squared Euclidean distance in frequency domain
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        # Weighted sum
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        matrix: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the forward pass of the focal frequency loss.

        Args:
            pred (torch.Tensor): Predicted tensor of shape (N, C, H, W).
            target (torch.Tensor): Ground truth tensor of shape (N, C, H, W).
            matrix (torch.Tensor, optional): Precomputed spectrum weight matrix.
                If ``None``, the matrix is computed on‑the‑fly.【7698168858157†L95-L113】

        Returns:
            torch.Tensor: Scalar focal frequency loss value multiplied by
            ``loss_weight``.
        """
        # Transform to frequency domain
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        # Optionally use batch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, dim=0, keepdim=True)
            target_freq = torch.mean(target_freq, dim=0, keepdim=True)
        # Compute the core loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight
