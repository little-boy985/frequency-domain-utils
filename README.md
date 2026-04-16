# Frequency‑Domain Utility Modules

This repository collects a few small but useful components for working in the
frequency domain.  The included modules are inspired by recent research on
frequency‑aware training, domain generalization and image restoration.  They
are provided here to complement your existing project, which already
implements cross‑attention between spatial and frequency maps.  Each module
is self‑contained and documented; you can import them directly into your
codebase.

## Contents

| Module | Description |
|-------|------------|
| `focal_frequency_loss.py` | Implementation of the **Focal Frequency Loss** (FFL). This loss function measures the difference between an output and a target image in the Fourier domain and applies a dynamic spectrum weight to emphasize hard‑to‑synthesize frequency components. It was introduced in the ICCV 2020 paper by Jiang et al. and released under the MIT license【7698168858157†L9-L23】. The implementation here faithfully follows the original code and exposes a PyTorch class `FocalFrequencyLoss` with configurable loss weight, log scaling and patch size【7698168858157†L95-L113】. |
| `fsdr_randomization.py` | A simplified **Frequency Space Domain Randomization** (FSDR) augmentation. Inspired by FSDR for domain generalization, this function converts an image to the frequency domain, identifies a fraction of low‑magnitude (high‑frequency) coefficients as domain‑variant components and randomizes them while preserving high‑magnitude (low‑frequency) components. This approach is motivated by Huang et al.'s observation that keeping domain‑invariant frequency components and perturbing domain‑variant ones leads to controllable randomization and preserves semantic structures【209874257443478†L292-L304】. |

## How to use

1. **Focal Frequency Loss** – import `FocalFrequencyLoss` from
   `focal_frequency_loss.py` and use it like a standard PyTorch loss.  During
   training, it computes the L1 difference in the Fourier domain and
   multiplies it by a dynamic weight matrix to focus on difficult frequencies【7698168858157†L9-L23】.  You can adjust the `alpha` and `patch_factor` parameters to trade off between global and local frequency structures.  The code is derived from the official implementation licensed under MIT【194955270957107†L0-L21】.

   ```python
   from useful_github_code.focal_frequency_loss import FocalFrequencyLoss

   criterion = FocalFrequencyLoss(alpha=1.0, patch_factor=1, log_matrix=True)
   loss = criterion(output, target)
   loss.backward()
   ```

2. **FSDR Randomization** – import `fsdr_randomize` from
   `fsdr_randomization.py`.  Given a PyTorch tensor image of shape `(C, H, W)`
   (or batched), it returns a new image where a fraction of high‑frequency
   components are replaced by random noise.  The `ratio` argument controls
   how many coefficients are randomized (e.g., `ratio=0.1` randomizes 10% of
   the smallest‑magnitude frequencies).  This can be used as a data
   augmentation step to improve domain generalization as demonstrated by the
   original FSDR work【209874257443478†L292-L304】.

   ```python
   from useful_github_code.fsdr_randomization import fsdr_randomize

   aug_image = fsdr_randomize(image, ratio=0.05)
   ```

## Related research and references

Several recent papers and repositories have explored the synergy between
spatial and frequency domains.  Although their full implementations are
outside the scope of this utility package, you may find their ideas and
designs helpful:

- **AnyIR** – An **all‑in‑one image restoration** framework that uses a
  **spatial–frequency parallel fusion** strategy to unify denoising,
  deblurring and super‑resolution tasks.  Its authors emphasise that
  parallel fusion of spatial and frequency information leads to robust
  degradation‑aware representations and improved restoration fidelity【427242165965047†L14-L22】【427242165965047†L59-L61】.

- **SFCFusion** – A **Spatial–Frequency Collaborative Fusion** network
  for infrared and visible image fusion.  The repository provides
  implementation details and training instructions for reproducing the
  method, which was published in IEEE TIM 2024【604927806711359†L274-L327】.

- **SFDFusion** – An **Efficient Spatial‑Frequency Domain Fusion** network
  for infrared/visible image fusion (ECAI 2024).  Its code base includes
  configuration files and scripts for training and inference【431780947496524†L285-L321】.

- **FSDR** – The **Frequency Space Domain Randomization** method for
  domain generalization.  The authors point out that randomizing
  domain‑variant frequency components while keeping domain‑invariant ones
  produces controllable randomization and preserves semantic structures
  【209874257443478†L292-L304】.  Our `fsdr_randomization.py` is a simplified
  adaptation of this idea.

- **D3** – A **Dual‑Domain Downsampling** (D3) module described as
  *attention‑modulated frequency‑aware pooling via spatial guidance*.
  Although we do not include its code here, the official repository notes
  that the implementation is built on top of MMPretrain and uses
  frequency‑aware pooling to improve classification and detection【275513211516926†L274-L346】.

These references are included to provide context and further reading.  We
recommend consulting their respective papers and repositories if you wish to
adopt more advanced spatial–frequency fusion strategies.

## License

All files in this directory are provided under the terms of the MIT license
from the original Focal Frequency Loss and FSDR repositories【194955270957107†L0-L21】.  Please
retain the license and attribution if you reuse or modify the code.
