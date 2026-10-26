# M-polar based fuzzy image enhancement using exposure fusion

Python implementation and interactive demo.
The method enhances low-light images by:

1. Building an **m-polar fuzzy image** from the input: m fuzzy versions
   (poles) of the image, each generated with its own parameter γₖ (radians)
   as `F_k = 1 − (1 − A) / (1 + tan(γₖ)·A)`. The default poles are
   γ = [−0.75, −0.6, −0.3, 0, 1.2, 1.4, 1.5] (m = 7).
2. Computing three quality measures for every pole — **contrast** (absolute
   Laplacian), **saturation** (standard deviation of the colour channels) and
   **well-exposedness** (Gaussian around a target exposure, σ = 0.2) — and
   multiplying them, each raised to its exponent (default `m = [1, 1, 1]`),
   into a weight map that is normalised across the poles at every pixel.
3. **Fusing** the poles with Laplacian-pyramid exposure fusion, weighting
   each pole's pyramid by the Gaussian pyramid of its weight map.
4. Clipping the reconstructed image to [0, 1].

The input image is processed directly in RGB; no colour-space conversion or
search loop is involved.

**Live demo:** https://mpolar.streamlit.app

## Paper

> **M-polar based fuzzy image enhancement using exposure fusion**
> *(add authors, venue, volume/pages and year here)*

See [Citation](#citation) below for the BibTeX entry.

## Repository contents

| File | Description |
|---|---|
| `mfe.py` | Self-contained pipeline: m-polar fuzzy image generation, the contrast/saturation/well-exposedness measures, Gaussian and Laplacian pyramids, and exposure fusion (`fuzzy_exposure_fusion`).|
| `app.py` | Interactive Streamlit demo that visualizes every stage of the pipeline — the m poles, the per-pole weight maps, the fused result with histograms, and a PNG download. |
| `requirements.txt` | Python dependencies. |
| `sample.png` | *(optional)* Bundle your own low-light test image under this name and both scripts will use it as their default input. |

## Installation

```bash
git clone https://github.com/maheshkumar-cv/m-polar-fuzzy-low-light-image-enhancement
cd m-polar-fuzzy-low-light-image-enhancement
pip install -r requirements.txt
```

## Usage

### Command line

```bash
# Uses sample.png in this folder, default poles and weights
python mfe.py

# Or specify input and output explicitly
python mfe.py path/to/your_image.jpg path/to/output.png
```

Prints the number of poles, the entropy before and after enhancement, and
saves the enhanced image.

> **Note on runtime:** cost grows with image size and with the number of
> poles, since a Laplacian pyramid is built for every pole. The command line
> works at full resolution; the demo downsizes to the "Max image side"
> slider value (default 384 px).

### Interactive demo

```bash
streamlit run app.py
```

Opens a browser tab where you can:
- Upload your own image (or use the bundled `sample.png` / a stock
  fallback image if none is present).
- Choose a γ preset (3, 5 or 7 poles, brightening only) or type your own
  list of γ values.
- Set the contrast, saturation and well-exposedness exponents and the
  target exposure.
- Inspect each of the m fuzzy poles.
- View the normalised weight map of every pole.
- Compare the original and enhanced images with intensity histograms and
  mean / entropy / contrast metrics.
- Download the final enhanced RGB image.

A hosted version of this demo is live at **https://mpolar.streamlit.app** — no
installation needed.

## Using the library in your own code

```python
import mfe
from skimage import io

A = io.imread("your_image.jpg")[..., :3]

# Float RGB in [0, 1], default poles and weights
enhanced = mfe.fuzzy_exposure_fusion(A)

# Custom poles, weight exponents (contrast, saturation, well-exposedness)
# and target exposure
enhanced = mfe.fuzzy_exposure_fusion(
    A, m=(1, 1, 1), gammas=(-0.6, 0.0, 1.4), exp_value=0.5
)

# Also return the poles (m, H, W, 3) and weight maps (m, H, W)
enhanced, poles, weights = mfe.fuzzy_exposure_fusion(A, return_details=True)
```


## Citation

If you use this code, please cite the paper:

```bibtex
@ARTICLE{2026Mpolar,
	author = {Mahesh Kumar C V and Saraswathi D},
	title = {M-polar based fuzzy image enhancement using exposure fusion},
	journal = {Frontiers in Artificial Intelligence},
	year = {2026},
	volume = {},
	number = {},
	pages = {}}
```

## Contact

For questions about the code or the paper, feel free to email
maheshkumarcv961@gmail.com.
