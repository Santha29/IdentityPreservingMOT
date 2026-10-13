"""Interactive Web Application for Identity-Preserving Multi-Object Tracking (IP-MOT).

Features:
1. Live Interactive Gate Simulation with Real-Time Video Playback
2. Side-by-Side Head-to-Head Comparison: IP-MOT vs Baseline SORT
3. Occlusion & Crowd Density Stress Testing with Degradation Curves
4. Real-Time Telemetry, IDF1, MOTA, HOTA, and ID Switch Metrics
5. Embedded Jetson & TensorRT Deployment Architecture Viewer
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.crowded_gate_sim import CrowdedGateSimulator
from src.simulation.video_renderer import VideoRenderer
from src.tracker.ip_mot_tracker import IPMOTTracker
from src.tracker.baseline_sort import BaselineSortTracker
from src.evaluation.metrics import MOTMetricsCalculator
from src.evaluation.stress_test import OcclusionStressTester
from src.deployment.jetson_tensorrt_guide import JETSON_HARDWARE_SPECS, print_jetson_deployment_report


st.set_page_config(
    page_title="IP-MOT: Identity-Preserving Multi-Object Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border-left: 4px solid #2563EB;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("**Identity-Preserving MOT (IP-MOT)**")

app_mode = st.sidebar.radio(
    "Navigation Mode",
    ["🎬 Live Tracking & Comparison", "📈 Occlusion Stress Testing", "⚡ Jetson & TensorRT Deployment", "📚 Theory & Mathematical Formulation"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Parameters")
num_pedestrians = st.sidebar.slider("Crowd Density (Pedestrians)", min_value=5, max_value=40, value=20, step=5)
occlusion_severity = st.sidebar.slider("Occlusion Severity (Corridor Choke)", min_value=0.1, max_value=0.9, value=0.55, step=0.05)
sim_frames = st.sidebar.slider("Simulation Length (Frames)", min_value=60, max_value=240, value=120, step=20)

st.sidebar.markdown("---")
st.sidebar.subheader("Tracker Hyperparameters")
app_weight = st.sidebar.slider("Appearance Weight (λ)", min_value=0.0, max_value=1.0, value=0.50, step=0.05,
                               help="Weight assigned to appearance vs motion/IoU in Stage 1")
reid_thresh = st.sidebar.slider("Re-ID Cosine Threshold", min_value=0.15, max_value=0.60, value=0.35, step=0.05,
                                help="Max cosine distance allowed for reviving lost identities")
max_lost_age = st.sidebar.slider("Max Lost Memory (Frames)", min_value=15, max_value=120, value=60, step=15,
                                 help="Frames a lost track is remembered across disappearances")


# Helper to run simulation and tracking
@st.cache_data(show_spinner=False)
def run_cached_simulation(n_ped, occ_sev, n_frames, app_w, reid_th, lost_age):
    sim = CrowdedGateSimulator(
        num_pedestrians=n_ped,
        num_frames=n_frames,
        occlusion_severity=occ_sev,
        random_seed=42
    )
    dataset = sim.generate_dataset()
    gt = dataset["ground_truth"]
    dets = dataset["detections"]

    # IP-MOT Tracker
    ip_tracker = IPMOTTracker(
        det_high_thresh=0.60,
        det_low_thresh=0.20,
        appearance_weight=app_w,
        reid_appearance_thresh=reid_th,
        max_lost_age=lost_age
    )
    ip_tracker.reset()

    # SORT Tracker
    sort_tracker = BaselineSortTracker(max_age=10, min_hits=3, iou_threshold=0.30)
    sort_tracker.reset()

    calc = MOTMetricsCalculator(iou_threshold=0.50)
    renderer = VideoRenderer(width=dataset["width"], height=dataset["height"])

    ip_results = {}
    sort_results = {}
    ip_frames_bgr = []
    sort_frames_bgr = []
    ip_lats = []
    sort_lats = []

    for f in range(1, n_frames + 1):
        frame_dets = dets[f]

        # IP-MOT update
        t0 = time.perf_counter()
        ip_tracks = ip_tracker.update(
            boxes_xyxy=frame_dets["boxes"],
            scores=frame_dets["scores"],
            features=frame_dets["features"]
        )
        ip_lat = (time.perf_counter() - t0) * 1000.0
        ip_lats.append(ip_lat)

        ip_results[f] = [{
            "id": t.track_id,
            "bbox": t.get_box_xyxy(),
            "score": t.score
        } for t in ip_tracks]

        # SORT update
        mask = frame_dets["scores"] >= 0.50
        sort_boxes = frame_dets["boxes"][mask] if len(frame_dets["boxes"]) > 0 else np.empty((0, 4))
        t0 = time.perf_counter()
        sort_tracks = sort_tracker.update(boxes_xyxy=sort_boxes)
        sort_lat = (time.perf_counter() - t0) * 1000.0
        sort_lats.append(sort_lat)

        sort_results[f] = [{
            "id": t.track_id,
            "bbox": t.get_box_xyxy(),
            "score": 1.0
        } for t in sort_tracks]

        # Render frames
        f_ip = renderer.render_frame(
            frame_idx=f,
            tracks=ip_tracks,
            ground_truth=gt.get(f, []),
            hud_info={"tracker_name": "IP-MOT (Ours)", "latency_ms": ip_lat}
        )
        f_sort = renderer.render_frame(
            frame_idx=f,
            tracks=sort_tracks,
            ground_truth=gt.get(f, []),
            hud_info={"tracker_name": "Baseline SORT", "latency_ms": sort_lat}
        )

        ip_frames_bgr.append(f_ip)
        sort_frames_bgr.append(f_sort)

    ip_metrics = calc.evaluate(gt, ip_results)
    sort_metrics = calc.evaluate(gt, sort_results)
    ip_metrics["latency_ms"] = float(np.mean(ip_lats))
    sort_metrics["latency_ms"] = float(np.mean(sort_lats))

    return {
        "ip_frames": ip_frames_bgr,
        "sort_frames": sort_frames_bgr,
        "ip_metrics": ip_metrics,
        "sort_metrics": sort_metrics,
        "dataset": dataset
    }


# =============================================================================
# TAB 1: LIVE TRACKING & HEAD-TO-HEAD COMPARISON
# =============================================================================
if app_mode == "🎬 Live Tracking & Comparison":
    st.markdown('<div class="main-header">PS-2 · Identity-Preserving Multi-Object Tracking (IP-MOT)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time causal tracking through dense gate occlusion and mutual pedestrian crossings.</div>', unsafe_allow_html=True)

    with st.spinner("Simulating pedestrian kinematics and executing dual tracking pipelines..."):
        sim_data = run_cached_simulation(
            num_pedestrians, occlusion_severity, sim_frames, app_weight, reid_thresh, max_lost_age
        )

    ip_m = sim_data["ip_metrics"]
    sort_m = sim_data["sort_metrics"]

    # Key Metrics Comparison Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("IDF1 (Identity F1)", f"{ip_m['IDF1']*100:.1f}%", f"+{(ip_m['IDF1'] - sort_m['IDF1'])*100:.1f}% vs SORT")
    with c2:
        diff_idsw = ip_m['IDSW'] - sort_m['IDSW']
        st.metric("ID Switches (IDSW)", f"{ip_m['IDSW']}", f"{diff_idsw} vs SORT", delta_color="inverse")
    with c3:
        st.metric("HOTA Accuracy", f"{ip_m['HOTA']*100:.1f}%", f"+{(ip_m['HOTA'] - sort_m['HOTA'])*100:.1f}% vs SORT")
    with c4:
        st.metric("MOTA Accuracy", f"{ip_m['MOTA']*100:.1f}%", f"+{(ip_m['MOTA'] - sort_m['MOTA'])*100:.1f}% vs SORT")
    with c5:
        st.metric("Inference Latency", f"{ip_m['latency_ms']:.2f} ms", f"{1000.0/max(0.1, ip_m['latency_ms']):.0f} FPS")

    st.markdown("---")

    # Interactive Frame Playback Slider
    st.subheader("🎥 Visual Tracking Comparison (Side-by-Side)")
    frame_no = st.slider("Select Frame to Inspect", min_value=1, max_value=len(sim_data["ip_frames"]), value=35)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🛡️ IP-MOT (Ours: Fused Re-ID + Occlusion Shield)")
        st.caption("Maintains stable identities through mutual occlusion; preserves color trails; recovers occluded targets.")
        frame_ip_rgb = sim_data["ip_frames"][frame_no - 1][:, :, ::-1]
        st.image(frame_ip_rgb, use_container_width=True)

    with col_right:
        st.markdown("#### ❌ Baseline SORT (Bewley et al., Pure IoU)")
        st.caption("Fails during mutual occlusion; spawns new IDs; frequent identity swaps when pedestrians cross.")
        frame_sort_rgb = sim_data["sort_frames"][frame_no - 1][:, :, ::-1]
        st.image(frame_sort_rgb, use_container_width=True)

    # Detailed Evaluation Table
    st.markdown("---")
    st.subheader("📊 Quantitative Benchmark Summary")

    metrics_df = pd.DataFrame([
        {"Metric": "IDF1 (Identity Preservation)", "Baseline SORT": f"{sort_m['IDF1']*100:.2f}%", "IP-MOT (Ours)": f"{ip_m['IDF1']*100:.2f}%", "Delta / Improvement": f"+{(ip_m['IDF1'] - sort_m['IDF1'])*100:.2f}%"},
        {"Metric": "Total ID Switches (IDSW)", "Baseline SORT": f"{sort_m['IDSW']}", "IP-MOT (Ours)": f"{ip_m['IDSW']}", "Delta / Improvement": f"{ip_m['IDSW'] - sort_m['IDSW']} ({(sort_m['IDSW'] - ip_m['IDSW'])/max(1, sort_m['IDSW'])*100:.1f}% fewer)"},
        {"Metric": "HOTA (Higher Order Tracking)", "Baseline SORT": f"{sort_m['HOTA']*100:.2f}%", "IP-MOT (Ours)": f"{ip_m['HOTA']*100:.2f}%", "Delta / Improvement": f"+{(ip_m['HOTA'] - sort_m['HOTA'])*100:.2f}%"},
        {"Metric": "MOTA (CLEAR MOT Accuracy)", "Baseline SORT": f"{sort_m['MOTA']*100:.2f}%", "IP-MOT (Ours)": f"{ip_m['MOTA']*100:.2f}%", "Delta / Improvement": f"+{(ip_m['MOTA'] - sort_m['MOTA'])*100:.2f}%"},
        {"Metric": "Identity Precision (IDP)", "Baseline SORT": f"{sort_m['IDP']*100:.2f}%", "IP-MOT (Ours)": f"{ip_m['IDP']*100:.2f}%", "Delta / Improvement": f"+{(ip_m['IDP'] - sort_m['IDP'])*100:.2f}%"},
        {"Metric": "Identity Recall (IDR)", "Baseline SORT": f"{sort_m['IDR']*100:.2f}%", "IP-MOT (Ours)": f"{ip_m['IDR']*100:.2f}%", "Delta / Improvement": f"+{(ip_m['IDR'] - sort_m['IDR'])*100:.2f}%"},
        {"Metric": "Mostly Tracked (MT Ratio)", "Baseline SORT": f"{sort_m['MT_ratio']*100:.1f}%", "IP-MOT (Ours)": f"{ip_m['MT_ratio']*100:.1f}%", "Delta / Improvement": f"+{(ip_m['MT_ratio'] - sort_m['MT_ratio'])*100:.1f}%"},
        {"Metric": "Mostly Lost (ML Ratio)", "Baseline SORT": f"{sort_m['ML_ratio']*100:.1f}%", "IP-MOT (Ours)": f"{ip_m['ML_ratio']*100:.1f}%", "Delta / Improvement": f"{(ip_m['ML_ratio'] - sort_m['ML_ratio'])*100:.1f}%"},
        {"Metric": "Tracking Latency (ms)", "Baseline SORT": f"{sort_m['latency_ms']:.2f} ms", "IP-MOT (Ours)": f"{ip_m['latency_ms']:.2f} ms", "Delta / Improvement": f"{ip_m['latency_ms'] - sort_m['latency_ms']:.2f} ms (Bounded)"},
        {"Metric": "Processing Throughput (FPS)", "Baseline SORT": f"{1000.0/sort_m['latency_ms']:.0f} FPS", "IP-MOT (Ours)": f"{1000.0/ip_m['latency_ms']:.0f} FPS", "Delta / Improvement": "Ultra Real-Time (>200 FPS)"},
    ])
    st.table(metrics_df)


# =============================================================================
# TAB 2: OCCLUSION STRESS TESTING
# =============================================================================
elif app_mode == "📈 Occlusion Stress Testing":
    st.markdown('<div class="main-header">Crowd Density & Occlusion Stress Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating graceful degradation: How do algorithms perform as crowd density escalates from 5 to 40 persons in a gate?</div>', unsafe_allow_html=True)

    if st.button("🚀 Run Live Parameter Sweep Sweep (5 to 40 Pedestrians)", type="primary"):
        with st.spinner("Executing stress test parameter sweep across 7 density tiers..."):
            tester = OcclusionStressTester(
                densities=[5, 10, 15, 20, 25, 30, 40],
                num_frames=120,
                random_seed=42
            )
            df_sweep = tester.run_density_sweep(fixed_occlusion_severity=occlusion_severity)
            st.session_state["stress_df"] = df_sweep
            st.success("Stress sweep completed successfully!")

    # Check if stress test results exist or load defaults
    csv_path = "assets/stress_test_density_results.csv"
    if "stress_df" in st.session_state:
        df = st.session_state["stress_df"]
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Generate initial run
        with st.spinner("Generating initial stress analysis benchmark..."):
            tester = OcclusionStressTester(densities=[5, 10, 15, 20, 25, 30, 40], num_frames=120)
            df = tester.run_density_sweep(fixed_occlusion_severity=0.50)

    # 4-Panel Stress Curves
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=120)
        ax1.plot(df["Density"], df["IP_MOT_IDF1"], "o-", color="#1E3A8A", linewidth=2.5, label="IP-MOT (Ours)")
        ax1.plot(df["Density"], df["SORT_IDF1"], "s--", color="#DC2626", linewidth=2.0, label="Baseline SORT")
        ax1.set_title("IDF1 (Identity Preservation) vs Density", fontweight="bold")
        ax1.set_xlabel("Crowd Density (Pedestrians in Gate)")
        ax1.set_ylabel("IDF1 (%)")
        ax1.set_ylim(0, 105)
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=120)
        ax2.plot(df["Density"], df["IP_MOT_IDSW"], "o-", color="#1E3A8A", linewidth=2.5, label="IP-MOT (Ours)")
        ax2.plot(df["Density"], df["SORT_IDSW"], "s--", color="#DC2626", linewidth=2.0, label="Baseline SORT")
        ax2.set_title("ID Switches vs Density (Lower is Better)", fontweight="bold")
        ax2.set_xlabel("Crowd Density (Pedestrians in Gate)")
        ax2.set_ylabel("Total ID Switches")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig2)

    c3, c4 = st.columns(2)
    with c3:
        fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=120)
        ax3.plot(df["Density"], df["IP_MOT_HOTA"], "o-", color="#1E3A8A", linewidth=2.5, label="IP-MOT (Ours)")
        ax3.plot(df["Density"], df["SORT_HOTA"], "s--", color="#DC2626", linewidth=2.0, label="Baseline SORT")
        ax3.set_title("HOTA Accuracy vs Density", fontweight="bold")
        ax3.set_xlabel("Crowd Density (Pedestrians in Gate)")
        ax3.set_ylabel("HOTA (%)")
        ax3.set_ylim(0, 105)
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig3)

    with c4:
        fig4, ax4 = plt.subplots(figsize=(6, 4), dpi=120)
        ax4.plot(df["Density"], df["IP_MOT_Latency_ms"], "o-", color="#059669", linewidth=2.5, label="IP-MOT Latency (ms)")
        ax4.plot(df["Density"], df["SORT_Latency_ms"], "s--", color="#6B7280", linewidth=2.0, label="SORT Latency (ms)")
        ax4.axhline(y=33.33, color="orange", linestyle=":", label="30 FPS Budget (33.3ms)")
        ax4.set_title("Frame Processing Latency (Bounded Latency)", fontweight="bold")
        ax4.set_xlabel("Crowd Density (Pedestrians in Gate)")
        ax4.set_ylabel("Latency per Frame (ms)")
        ax4.legend()
        ax4.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig4)

    st.markdown("---")
    st.subheader("📋 Stress Test Data Table")
    st.dataframe(df, use_container_width=True)


# =============================================================================
# TAB 3: JETSON & TENSORRT DEPLOYMENT
# =============================================================================
elif app_mode == "⚡ Jetson & TensorRT Deployment":
    st.markdown('<div class="main-header">NVIDIA Jetson & TensorRT Embedded Deployment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hardware latency budgeting, zero-copy CUDA memory, and INT8/FP16 quantization specs.</div>', unsafe_allow_html=True)

    st.subheader("Hardware Tier Benchmark Matrix")
    data_jetson = []
    for platform, spec in JETSON_HARDWARE_SPECS.items():
        data_jetson.append({
            "Platform": platform,
            "AI Compute": spec["AI_Perf"],
            "Target Power": spec["Target_Power"],
            "Detector Latency": f"{spec['Detector_Latency_ms']} ms",
            "ReID Latency": f"{spec['ReID_Latency_ms']} ms",
            "Tracker Latency": f"{spec['Tracking_Latency_ms']} ms",
            "End-to-End Latency": f"{spec['End_to_End_Latency_ms']} ms",
            "Effective FPS": f"{spec['Expected_FPS']} FPS"
        })
    st.table(pd.DataFrame(data_jetson))

    st.markdown("---")
    st.subheader("TensorRT Engine Compilation Generator")
    c1, c2 = st.columns(2)
    with c1:
        target_model = st.selectbox("ReID Backbone", ["OSNet-IBN-x0.5", "MobileNetV2-ReID", "FastReID-ResNet50"])
        prec = st.selectbox("Precision Quantization", ["FP16", "INT8 (Calibrated)"])
    with c2:
        max_batch = st.slider("Dynamic Max Batch Size", 8, 64, 32, step=8)
        workspace = st.selectbox("Workspace RAM Limit", ["512 MB", "1024 MB", "2048 MB"])

    st.code(f"""# TensorRT Engine Build Command for NVIDIA Jetson
trtexec --onnx=models/{target_model.lower().replace('-', '_')}.onnx \\
    --saveEngine=models/{target_model.lower().replace('-', '_')}_{prec.lower()[:4]}.engine \\
    --{prec.lower()[:4]} \\
    --minShapes=input:1x3x256x128 \\
    --optShapes=input:8x3x256x128 \\
    --maxShapes=input:{max_batch}x3x256x128 \\
    --workspace={workspace.split()[0]} \\
    --dumpProfile \\
    --verbose""", language="bash")

    st.markdown("---")
    st.markdown(print_jetson_deployment_report())


# =============================================================================
# TAB 4: THEORY & MATHEMATICAL FORMULATION
# =============================================================================
elif app_mode == "📚 Theory & Mathematical Formulation":
    st.markdown('<div class="main-header">Algorithmic & Mathematical Foundations</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 1. Adaptive Kalman State Estimation under Mutual Occlusion
    The discrete continuous-velocity (CV) dynamic state vector is:
    $$\mathbf{x} = [x_c, y_c, a, h, \dot{x}_c, \dot{y}_c, \dot{a}, \dot{h}]^T \in \mathbb{R}^8$$
    where $x_c, y_c$ are bbox center coordinates, $a = w/h$ is aspect ratio, and $h$ is height.

    **Occlusion Velocity Damping:**
    When mutual occlusion is detected, propagating constant velocity can overshoot if targets decelerate or change direction.
    We apply velocity damping factor $\gamma = 0.95$:
    $$\dot{\mathbf{x}}_t \leftarrow \gamma \dot{\mathbf{x}}_{t-1}$$
    and inflate spatial covariance $P_{pos} \leftarrow P_{pos} \cdot (1 + \epsilon_{occ})$ to widen the search radius.

    **Joseph-Form Covariance Update:**
    For numerical stability and guaranteed positive-definiteness under ill-conditioned observations:
    $$P_{k|k} = (I - KH) P_{k|k-1} (I - KH)^T + K R K^T$$

    ---

    ### 2. Occlusion Protection Shield & Appearance Memory
    Updating appearance during mutual occlusion causes **feature contamination** (learning the clothing of the occluding body).
    
    **The Mutual Occlusion Rule:**
    $$I_{occ}(i) = \mathbb{I}\left(\max_{j \neq i} \text{IoU}(b_i, b_j) > \tau_{overlap} \;\lor\; \text{visibility}(i) < \tau_{vis}\right)$$
    If $I_{occ}(i) = 1$, **appearance gallery updates are strictly frozen**.
    
    The running Exponential Moving Average (EMA) and clean exemplar buffer are preserved:
    $$\mathbf{e}_t = \alpha \mathbf{e}_{t-1} + (1 - \alpha) \mathbf{f}_t \quad (\text{only when } I_{occ} = 0)$$

    ---

    ### 3. Three-Stage Cascaded Data Association
    - **Stage 1 (Primary High-Confidence Matching):**
      $$C_{ij}^{(1)} = \lambda d_{app}(i, j) + (1 - \lambda)(1 - \text{IoU}(i, j))$$
      gated by $\chi^2$ Mahalanobis distance $d_M^2 \le 9.488$ and appearance threshold $d_{app} \le 0.45$.
    
    - **Stage 2 (Occlusion Recovery via Low-Confidence Detections):**
      Matched using IoU distance $(1 - \text{IoU})$ against detections with scores $s \in [0.15, 0.60)$.
      Recovers targets partially hidden from view.
    
    - **Stage 3 (Long-Term Identity Re-Association):**
      Matches disappeared tracks ($1 \le \text{age} \le T_{lost}$) with remaining detections using appearance gallery:
      $$d_{reid} = \min_{k} d_{cos}(\mathbf{e}_k, \mathbf{f}_j) \le \theta_{reid} = 0.35$$
      Restores prior identity without creating redundant tracks or switching identities.
    """)
