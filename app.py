import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage import data as skdata
from skimage import img_as_ubyte
from skimage import io as skio
from skimage import transform

import mfe

st.set_page_config(page_title="M-polar fuzzy exposure fusion", layout="wide")

DEFAULT_IMAGE = Path(__file__).with_name("sample.png")
PRESETS = {
    "Paper default (7 poles)": mfe.DEFAULT_GAMMAS,
    "3 poles": (-0.6, 0.0, 1.4),
    "5 poles": (-0.75, -0.3, 0.0, 1.2, 1.5),
    "Brightening only": (0.0, 0.6, 1.0, 1.2, 1.4, 1.5),
    "Custom": None,
}


def to_u8(x):
    return np.clip(np.rint(np.nan_to_num(np.asarray(x, float)) * 255), 0, 255).astype(np.uint8)


def show(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except (TypeError, ValueError):
        st.image(img, caption=caption, use_container_width=True)


def load_image(file_bytes, max_side):
    if file_bytes is not None:
        A = skio.imread(io.BytesIO(file_bytes))
    elif DEFAULT_IMAGE.exists():
        A = skio.imread(DEFAULT_IMAGE)
    else:
        A = to_u8((skdata.astronaut() / 255.0) ** 2.2)
    if A.ndim == 2:
        A = np.dstack([A] * 3)
    A = img_as_ubyte(A[..., :3])
    h, w = A.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        A = img_as_ubyte(transform.resize(A, (int(h * s), int(w * s)), anti_aliasing=True))
    return A


@st.cache_data(show_spinner=False)
def run(A, gammas, m, exp_value):
    out, I, W = mfe.fuzzy_exposure_fusion(A, m=m, gammas=gammas, exp_value=exp_value, return_details=True)
    return dict(out=out, I=I, W=W)


st.title("M-polar based fuzzy image enhancement using exposure fusion")
st.caption("RGB → m fuzzy images (poles) → quality weight maps → Laplacian-pyramid exposure fusion")

with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    max_side = st.slider("Max image side (px)", 64, 1024, 384, 32)

    st.header("Poles (γ, radians)")
    preset = st.selectbox("Preset", list(PRESETS))
    if PRESETS[preset] is None:
        txt = st.text_input("Comma-separated γ values", "-0.75, -0.6, -0.3, 0, 1.2, 1.4, 1.5")
        try:
            gammas = tuple(float(t) for t in txt.split(",") if t.strip())
        except ValueError:
            st.error("Could not parse γ values.")
            st.stop()
    else:
        gammas = PRESETS[preset]
    if len(gammas) < 2:
        st.error("Use at least 2 poles.")
        st.stop()

    st.header("Fusion weights (exponents)")
    mc = st.slider("Contrast", 0.0, 3.0, mfe.DEFAULT_M[0], 0.1)
    ms = st.slider("Saturation", 0.0, 3.0, mfe.DEFAULT_M[1], 0.1)
    mw = st.slider("Well-exposedness", 0.0, 3.0, mfe.DEFAULT_M[2], 0.1)
    exp_value = st.slider("Target exposure", 0.0, 1.0, mfe.DEFAULT_EXPOSED, 0.05)
    st.caption("An exponent of 0 switches that measure off.")

A = load_image(up.getvalue() if up else None, max_side)
if up is None:
    st.info("No file uploaded – showing " + ("the bundled sample image." if DEFAULT_IMAGE.exists()
                                             else "a darkened stock image.") + " Upload your own in the sidebar.")

R = run(A, tuple(gammas), (mc, ms, mw), exp_value)
Af = mfe.to_float(A)
g_in, g_out = mfe.rgb2gray(Af), mfe.rgb2gray(R["out"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Poles (m)", len(gammas))
c2.metric("Mean intensity", f"{g_out.mean():.3f}", f"{g_out.mean() - g_in.mean():+.3f}")
c3.metric("Entropy", f"{mfe.entropy(g_out):.3f}", f"{mfe.entropy(g_out) - mfe.entropy(g_in):+.3f}")
c4.metric("Std (contrast)", f"{g_out.std():.3f}", f"{g_out.std() - g_in.std():+.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["1 · M-polar fuzzy images", "2 · Weight maps",
                                  "3 · Fused result", "4 · Download"])

with tab1:
    show(A, "Original")
    cols = st.columns(min(4, len(gammas)))
    for k, g in enumerate(gammas):
        with cols[k % len(cols)]:
            show(to_u8(R["I"][k]), f"Pole {k + 1}: γ = {g:g}  (tan γ = {np.tan(g):.2f})")

with tab2:
    st.write("Normalised per-pixel weight of each pole (bright = this pole dominates there).")
    cols = st.columns(min(4, len(gammas)))
    for k, g in enumerate(gammas):
        with cols[k % len(cols)]:
            show(to_u8(R["W"][k]), f"Pole {k + 1}: γ = {g:g}")
    share = pd.DataFrame({"pole": range(1, len(gammas) + 1), "γ": gammas,
                          "mean weight": R["W"].mean(axis=(1, 2))})
    st.dataframe(share, hide_index=True)

with tab3:
    a, b = st.columns(2)
    with a:
        show(A, "Original")
    with b:
        show(to_u8(R["out"]), "Enhanced (exposure fusion)")
    fig, ax = plt.subplots(figsize=(8, 2.8))
    x = np.linspace(0, 1, 256)
    for arr, lab in ((g_in, "original"), (g_out, "enhanced")):
        h, _ = np.histogram(arr, bins=256, range=(0, 1))
        ax.plot(x, h / h.sum(), label=lab, lw=1.2)
    ax.set_xlabel("intensity")
    ax.set_ylabel("fraction of pixels")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    buf = io.BytesIO()
    Image.fromarray(to_u8(R["out"])).save(buf, format="PNG")
    show(to_u8(R["out"]))
    st.download_button("Download enhanced PNG", buf.getvalue(), "enhanced_mpolar.png", "image/png")
