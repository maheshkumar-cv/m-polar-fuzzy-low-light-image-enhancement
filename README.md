# M-Polar Based Fuzzy Image Enhancement Using Exposure Fusion

Python implementation and interactive demo accompanying the article
**M-polar based fuzzy image enhancement using exposure fusion**.

The method enhances low-light images by:

1. Building an **m-polar fuzzy image**: m fuzzy versions (poles) of the input,
   each generated with a different parameter γₖ (radians):
   `F_k = 1 − (1 − A) / (1 + tan(γₖ)·A)`.
   Default poles: γ = [−0.75, −0.6, −0.3, 0, 1.2, 1.4, 1.5] (m = 7).
2. Computing three quality measures per pole — **contrast** (absolute
   Laplacian), **saturation** (std. dev. of the colour channels) and
   **well-exposedness** (Gaussian around a target exposure, σ = 0.2) — and
   multiplying them (each raised to its exponent `m = [1, 1, 1]`) into a weight map,
   normalised across poles at every pixel.
3. Blending the poles with **Laplacian-pyramid exposure fusion**, then clipping to [0, 1].

## Repository contents

| File | Description |
|---|---|
| `mfe.py` | Self-contained pipeline: m-polar image generation, quality measures, Gaussian/Laplacian pyramids, fusion. Port of the MATLAB `expFusion` class. Also runnable as a script. |
| `app.py` | Streamlit demo: poles, weight maps, fused result with histograms, PNG download. |
| `requirements.txt` | Python dependencies. |
| `sample.png` | *(optional)* Your own low-light test image; used as the default input. |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python mfe.py path/to/image.jpg out.png   # command line
streamlit run app.py                      # interactive demo
```

```python
import mfe
from skimage import io

A = io.imread("your_image.jpg")[..., :3]
out = mfe.fuzzy_exposure_fusion(A)  # float RGB in [0, 1]
out = mfe.fuzzy_exposure_fusion(A, m=(1, 1, 1), gammas=(-0.6, 0, 1.4), exp_value=0.5)
out, poles, weights = mfe.fuzzy_exposure_fusion(A, return_details=True)
```

MATLAB name-value options map to Python as: `'m'` → `m`, `'gamma'` → `gammas`,
`'exposed'` → `exp_value`.

## Notes on the port

- The fuzzy image is clipped to [0, 1] and the denominator is floored at machine epsilon,
  to stay safe if you choose a very negative γ.
- Pyramid filtering uses symmetric borders for downsampling and zero borders for upsampling, as in
  the MATLAB code.

## Citation

*(Add the article's BibTeX entry here once published.)*
