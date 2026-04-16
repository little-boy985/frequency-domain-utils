"""
fsdr_randomization.py
=====================

This module implements a simplified version of the **Frequency Space Domain
Randomization (FSDR)** augmentation proposed by Huang et al. for domain
generalization.  FSDR operates in the frequency domain: images are
decomposed into frequency components, then *domain-invariant* frequencies
(generally low‑frequency signals) are preserved while *domain‑variant*
frequencies (high‑frequency details) are randomly perturbed.  Randomizing
the domain‑variant components encourages a model to rely on robust, global
features and helps it generalize to unseen domains【209874257443478†L292-L304】.

The original FSDR implementation uses discrete cosine transforms (DCT) and
learns to identify domain‑invariant vs. vvariant components.  For
simplicity, this module uses a Fourier transform (FFT) and identifies
domain‑variant components by ranking frequency magnitudes: low‑magnitude
coefficients typically correspond to high‑frequency details, which are
randomized while high‑magnitude (low‑frequency) components are kept.  This
approximation preserves the spirit of FSDR described in the authors' paper
and README【209874257443478†L292-L304】.

The function defined here can be used as a data augmentation step for
images represented as PyTorch tensors.

Note: This code is provided under the MIT license of the original
repository. Please retain the license text if you reuse or modify it.
"""

from __future__ import annotations

import torch


def fsdr_randomize(
    image: torch.Tensor,
    ratio: float = 0.1,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Randomize high‑frequency components of an image in the frequency domain.

    This function performs a simplistic version of Frequency Space Domain
    Randomization (FSDR) as described by Huang et al【209874257443478†L292-L304】.  The input
    image is transformed into the frequency domain using a 2‑D FFT,
    frequency coefficients are ranked by magnitude, and the smallest
    ``ratio`` fraction of coefficients (considered domain‑variant) are
    replaced by random noise while the rest (domain‑invariant) are kept.

    Parameters
    ----------
    image:
        A ``torch.Tensor`` of shape ``(..., H, W)`` or ``(C, H, W)``
        representing a grayscale or multi‑channel image.  If a batch
        dimension is present, the transform is applied per image in the
        batch.
    ratio:
        Fraction of coefficients with the smallest magnitude that will be
        randomized.  A small value (e.g., 0.1) randomizes high‑frequency
        details while preserving global structure.
    epsilon:
        Small constant added to magnitudes to avoid numerical issues when
        computing quantiles.

    Returns
    -------
    torch.Tensor
        The randomized image with the same shape as the input.  Real
        components of the inverse FFT are returned; imaginary parts are
        discarded.

    Examples
    --------
    >>> import torch
    >>> img = torch.rand(3, 128, 128)  # random RGB image
    >>> aug = fsdr_randomize(img, ratio=0.05)
    >>> aug.shape
    torch.Size([3, 128, 128])

    Notes
    -----
    - This implementation is a heuristic adaptation of FSDR that does not
      require learning domain‑invariant vs. variant components.  For a more
      faithful reproduction, please refer to the original paper and code
      repository【209874257443478†L292-L304】.
    - Complex tensors returned by the FFT are handled via magnitude
      (`abs`) and phase (`angle`) decomposition; after randomization,
      coefficients are reconstructed and the inverse FFT is applied.
    """

    # Ensure image has at least 2 spatial dimensions.
    if image.dim() < 2:
        raise ValueError("Input must have at least two spatial dimensions")

    # Flatten non‑spatial dimensions for unified processing.
    orig_shape = image.shape
    # Reshape to (N, H, W) where N is product of other dimensions.
    n = int(torch.prod(torch.tensor(orig_shape[:-2])))
    img_flat = image.reshape(n, orig_shape[-2], orig_shape[-1])

    # Perform FFT on each image.
    freq = torch.fft.fft2(img_flat)
    amplitude = torch.abs(freq)
    phase = torch.angle(freq)

    # Compute threshold for domain‑variant (high‑freq) coefficients.
    # Add epsilon to avoid issues with zeros.
    amplitude_flat = amplitude.view(n, -1) + epsilon
    # Determine the quantile across all coefficients per image.
    kth_values = torch.quantile(amplitude_flat, ratio, dim=1, keepdim=True)
    # Broadcast threshold to original shape.
    threshold = kth_values.view(n, 1, 1)
    mask = amplitude < threshold  # True for high‑freq coefficients

    # Randomize magnitudes of domain‑variant coefficients.  We use
    # uniform noise scaled by the mean magnitude of the masked region.
    # Compute mean per image; avoid division by zero.
    masked_amp = amplitude[mask]
    # If all coefficients are kept (rare when ratio is small), fallback to
    # zeros to avoid error.
    mean_amp = masked_amp.mean() if masked_amp.numel() > 0 else 0.0
    noise = torch.rand_like(amplitude[mask]) * mean_amp
    amplitude = amplitude.clone()  # avoid modifying in‑place
    amplitude[mask] = noise

    # Reconstruct frequency domain tensor.
    freq_new = amplitude * torch.exp(1j * phase)
    # Inverse FFT and take real part.
    img_randomized = torch.real(torch.fft.ifft2(freq_new))
    # Restore original shape.
    return img_randomized.reshape(orig_shape)


__all__ = ["fsdr_randomize"]
