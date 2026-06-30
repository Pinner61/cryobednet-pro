from pathlib import Path
import numpy as np
import streamlit as st

st.set_page_config(page_title="CryoBedNet", layout="wide")
st.title("CryoBedNet: Antarctic Bed Topography Super-Resolution")
st.caption("Prediction, baseline, error, and transect explorer")

path = st.text_input("Prediction file", "outputs/mock_cpu/predictions_holdout.npz")
if not Path(path).exists():
    st.warning("Run training and evaluation first, then load the predictions file.")
    st.stop()

z = np.load(path, allow_pickle=True)
pred, target, bicubic = z["pred"], z["target"], z["bicubic"]
tile_id = z["tile_id"] if "tile_id" in z else np.arange(len(pred)).astype(str)
idx = st.slider("Tile", 0, len(pred) - 1, 0)
st.subheader(str(tile_id[idx]))

cols = st.columns(4)
for col, name, arr in zip(cols, ["Target", "Prediction", "Bicubic", "Error"], [target, pred, bicubic, pred - target]):
    with col:
        st.write(name)
        st.image(arr[idx, 0], clamp=True, use_container_width=True)

row = st.slider("Transect row", 0, pred.shape[-2] - 1, pred.shape[-2] // 2)
st.line_chart({"target": target[idx, 0, row], "prediction": pred[idx, 0, row], "bicubic": bicubic[idx, 0, row]})

err = pred - target
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{np.mean(np.abs(err)):.4f}")
c2.metric("RMSE", f"{np.sqrt(np.mean(err**2)):.4f}")
c3.metric("Mean bias", f"{np.mean(err):.4f}")
