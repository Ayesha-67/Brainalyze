import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import plotly.graph_objects as go
import plotly.express as px

from dipy.io.image import load_nifti
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines

# Streamlit UI setup
st.set_page_config(page_title="dMRI Tractography Viewer", layout="wide", page_icon="🧬")
st.title("🧠 Diffusion MRI Tractography Viewer")
st.markdown("Upload dMRI data (.nii.gz), b-vectors, and b-values files to visualize brain tracts and explore FA statistics.")

# Uploaders
dmri_file = st.file_uploader("Upload dMRI Data (.nii.gz)", type=["nii.gz"])
bvecs_file = st.file_uploader("Upload b-vectors file (.txt)", type=["txt"])
bvals_file = st.file_uploader("Upload b-values file (.txt)", type=["txt"])

class ThresholdTissueClassifier:
    def __init__(self, fa, threshold):
        self.fa = fa
        self.threshold = threshold
    def check_point(self, point):
        i, j, k = np.round(point).astype(int)
        try:
            return self.fa[i, j, k] > self.threshold
        except IndexError:
            return False

@st.cache_data
def load_and_prepare_data(dmri_file, bvecs_file, bvals_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_dmri:
        tmp_dmri.write(dmri_file.read())
        tmp_dmri.flush()
        data, affine = load_nifti(tmp_dmri.name)

    bvecs = np.loadtxt(bvecs_file)
    bvals = np.loadtxt(bvals_file)

    if bvecs.shape[0] != 3:
        bvecs = bvecs.T
    if bvecs.shape[1] != data.shape[-1] or bvals.shape[0] != data.shape[-1]:
        st.error("Mismatch between DWI volumes and gradient table!")
        return None, None, None, None

    gtab = gradient_table(bvals, bvecs)
    return data, affine, gtab, True

if dmri_file and bvecs_file and bvals_file:
    data, affine, gtab, success = load_and_prepare_data(dmri_file, bvecs_file, bvals_file)

    if success:
        st.subheader("Processing Data...")

        try:
            b0_mask = gtab.b0s_mask
            mean_b0 = np.mean(data[..., b0_mask], axis=-1)
            masked_data, mask = median_otsu(mean_b0, median_radius=2, numpass=1)

            ten_model = TensorModel(gtab)
            ten_fit = ten_model.fit(data, mask=mask)
            fa = ten_fit.fa
            fa = np.clip(fa, 0, 1)

            seeds = utils.seeds_from_mask(fa > 0.3, density=1, affine=affine)
            sphere = get_sphere(name='repulsion724')
            peaks = peaks_from_model(model=ten_model, data=data, sphere=sphere,
                                     relative_peak_threshold=0.8, min_separation_angle=15,
                                     mask=mask, return_sh=False)

            stopping_criterion = ThresholdStoppingCriterion(fa, 0.2)
            streamline_generator = LocalTracking(peaks, stopping_criterion, seeds, affine=affine, step_size=0.5)
            streamlines = Streamlines(streamline_generator)

            if len(streamlines) == 0:
                st.warning("No streamlines were generated. Try adjusting FA threshold or mask.")
            else:
                fig = go.Figure()
                for s in streamlines[::len(streamlines)//100 or 1]:
                    fig.add_trace(go.Scatter3d(
                        x=s[:, 0], y=s[:, 1], z=s[:, 2],
                        mode='lines',
                        line=dict(width=1),
                        opacity=0.5
                    ))

                fig.update_layout(
                    title="3D Tractography Visualization",
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    height=700
                )
                st.plotly_chart(fig)

                # Simulated FA vs Age plot
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Tract FA vs Age (Simulated)")
                    simulated_ages = np.linspace(5, 60, 100)
                    simulated_FA = 0.53 + 0.02 * np.sin(simulated_ages / 10)
                    fig_age = px.line(x=simulated_ages, y=simulated_FA, labels={"x": "Age (years)", "y": "Fractional Anisotropy"}, title="FA Changes with Age")
                    st.plotly_chart(fig_age)

                with col2:
                    st.subheader("📈 FA Comparison: Left vs Right Hemisphere (Simulated)")
                    angles = np.linspace(-45, 45, 100)
                    fa_left = 0.3 + 0.2 * np.sin(np.radians(angles)) + 0.02 * np.random.rand(100)
                    fa_right = 0.28 + 0.18 * np.cos(np.radians(angles)) + 0.02 * np.random.rand(100)

                    fig_hemi = go.Figure()
                    fig_hemi.add_trace(go.Scatter(x=angles, y=fa_left, mode='lines+markers', name='Left Hemisphere'))
                    fig_hemi.add_trace(go.Scatter(x=angles, y=fa_right, mode='lines+markers', name='Right Hemisphere'))
                    fig_hemi.update_layout(title="FA Across Corpus Callosum", xaxis_title="Angle (°)", yaxis_title="FA")
                    st.plotly_chart(fig_hemi)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
else:
    st.info("Please upload all required files to start visualization.")

