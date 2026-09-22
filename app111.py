#!/usr/bin/env python
# coding: utf-8

# In[1]:

from datetime import datetime, timedelta
import io
import json
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

register_heif_opener()

# -----------------------------------------------------------------------------
# 0. (Local Storage setup)
# -----------------------------------------------------------------------------
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
JSON_PATH = os.path.join(DATA_DIR, "events_db.json")

os.makedirs(IMAGES_DIR, exist_ok=True)

DEFAULT_EVENTS = [
    {
        "id": 1,
        "title": "First Walk with Someone Special (5km)",
        "category": "Romantic Moment (Historical Peak)",
        "date": "2023-06-25 21:52:00.000",
        "location": "Olympic Forest Park, Beijing",
        "novelty_score": 96.0,
        "ml_joy_prob": 0.94,
        "hr_peak": 125,
        "hr_baseline": 72,
        "hrv_rmssd": 95.0,
        "hrv_baseline": 42.0,
        "gsr_delta": 0.75,
        "motion_vm": 0.02,
        "description": " My heart is fluttering",
        "image": "https://picsum.photos/600/400?night",
    },
    {
        "id": 2,
        "title": "First Successful Dish Cooked Abroad",
        "category": "Independence Milestone (Historical Peak)",
        "date": "2023-09-08 16:48:00.000",
        "location": "Brooklyn Apartment, NYC",
        "novelty_score": 92.0,
        "ml_joy_prob": 0.91,
        "hr_peak": 104,
        "hr_baseline": 70,
        "hrv_rmssd": 88.0,
        "hrv_baseline": 40.0,
        "gsr_delta": 0.58,
        "motion_vm": 0.01,
        "description": "taste good",
        "image": "https://picsum.photos/600/400?food",
    },
]


def load_persistent_events():
    """"""
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_EVENTS
    else:
        save_persistent_events(DEFAULT_EVENTS)
        return DEFAULT_EVENTS


def save_persistent_events(events_list):
    """"""
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(events_list, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# 1. Page Configuration & WESAD ML Model Initialization
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Life Echo - Historical Peak Mining & Real-Time Analytics", layout="wide"
)


@st.cache_resource
def load_wesad_ml_model():
    np.random.seed(42)
    x_neutral = np.random.normal(loc=[72, 45, 0.3, 1, 0.15], scale=5, size=(60, 5))
    x_stress = np.random.normal(loc=[110, 25, 1.2, 8, 0.20], scale=5, size=(60, 5))
    x_joy = np.random.normal(loc=[95, 85, 0.7, 5, 0.02], scale=5, size=(60, 5))

    X = np.vstack([x_neutral, x_stress, x_joy])
    y = np.array([0] * 60 + [1] * 60 + [2] * 60)

    clf = RandomForestClassifier(n_estimators=25, random_state=42)
    clf.fit(X, y)
    return clf


model = load_wesad_ml_model()

# 初始化 session_state
if "events" not in st.session_state:
    st.session_state.events = load_persistent_events()

if "selected_event_id" not in st.session_state:
    st.session_state.selected_event_id = st.session_state.events[0]["id"]


# 安全加载图片辅助函数
def safe_load_image(file_obj):
    try:
        image_bytes = io.BytesIO(file_obj.getvalue())
        raw_img = Image.open(image_bytes)
        fixed_img = ImageOps.exif_transpose(raw_img)
        if fixed_img.mode != "RGB":
            fixed_img = fixed_img.convert("RGB")
        return fixed_img
    except Exception as err:
        st.error(f"⚠️ 图片读取失败：{err}")
        return None


# -----------------------------------------------------------------------------
# 2. Synchronized 3-Track Waveform Generator
# -----------------------------------------------------------------------------
def generate_3track_time_series(event):
    try:
        center_time = datetime.strptime(event["date"], "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        center_time = datetime.strptime(event["date"], "%Y-%m-%d %H:%M:%S")

    time_points = pd.date_range(
        start=center_time - timedelta(seconds=2),
        end=center_time + timedelta(seconds=2),
        periods=200,
    )
    t = np.linspace(-2, 2, 200)

    hr_base = event["hr_baseline"]
    hr_surge = (event["hr_peak"] - hr_base) * np.exp(-((t) ** 2) / 0.6)
    hr_series = hr_base + hr_surge + np.random.normal(0, 0.4, 200)

    hrv_base = event["hrv_baseline"]
    hrv_surge = (event["hrv_rmssd"] - hrv_base) * np.exp(-((t - 0.1) ** 2) / 0.5)
    hrv_series = hrv_base + hrv_surge + np.random.normal(0, 1.0, 200)

    gsr_base = 0.2
    gsr_surge = event["gsr_delta"] * (1 / (1 + np.exp(-6 * (t + 0.1))))
    gsr_series = gsr_base + gsr_surge + np.random.normal(0, 0.005, 200)

    return time_points, hr_series, hrv_series, gsr_series, time_points[100]


# -----------------------------------------------------------------------------
# 3. Sidebar: Live Wearable Tracking Simulation
# -----------------------------------------------------------------------------
st.sidebar.title("📡 Live Wearable Tracking")
st.sidebar.caption("Simulating automated capture during live bio-signal streaming")

sim_location = st.sidebar.text_input("GPS Location", "San Francisco Bay Area")
sim_scene = st.sidebar.text_input("Activity / Scene Description", "Watching Sunset by the Beach")
sim_hr = st.sidebar.slider("Live Heart Rate (BPM)", 50, 150, 110)
sim_hrv = st.sidebar.slider("HRV (RMSSD ms)", 10.0, 120.0, 85.0)
sim_gsr = st.sidebar.slider("GSR Conductance (μS)", 0.1, 2.0, 0.8)
sim_peaks = st.sidebar.slider("GSR Peak Frequency", 0, 10, 5)
sim_motion = st.sidebar.slider("Motion Acceleration (g)", 0.0, 0.5, 0.02)
sim_photo = st.sidebar.file_uploader(
    "📸 Attach Live Snapshot Photo", type=["jpg", "jpeg", "png", "heic"], key="sidebar_uploader"
)

if st.sidebar.button("⚡ Trigger ML Classification"):
    features = np.array([[sim_hr, sim_hrv, sim_gsr, sim_peaks, sim_motion]])
    probs = model.predict_proba(features)[0]
    joy_prob = float(probs[2]) if len(probs) > 2 else float(probs[-1])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    if joy_prob >= 0.60:
        new_id = (
            max([e["id"] for e in st.session_state.events]) + 1
            if st.session_state.events
            else 1
        )
        img_path = "https://picsum.photos/600/400?sunset"

        if sim_photo is not None:
            loaded_img = safe_load_image(sim_photo)
            if loaded_img is not None:
                img_filename = f"event_{new_id}_{int(datetime.now().timestamp())}.png"
                save_path = os.path.join(IMAGES_DIR, img_filename)
                loaded_img.save(save_path, "PNG")
                img_path = save_path

        new_event = {
            "id": new_id,
            "title": f"Live Capture: [{sim_scene}]",
            "category": "🔥 Live Tracking Peak",
            "date": timestamp,
            "location": sim_location,
            "novelty_score": 90.0,
            "ml_joy_prob": joy_prob,
            "hr_peak": int(sim_hr),
            "hr_baseline": 70,
            "hrv_rmssd": float(sim_hrv),
            "hrv_baseline": 40.0,
            "gsr_delta": float(sim_gsr),
            "motion_vm": float(sim_motion),
            "description": f"Live tracking detected significant bio-signal baseline variance at [{timestamp}].",
            "image": img_path,
        }
        st.session_state.events.insert(0, new_event)
        st.session_state.selected_event_id = new_id
        save_persistent_events(st.session_state.events)
        st.sidebar.balloons()
        st.sidebar.success("🎉 Real-Time Peak Captured & Saved Successfully!")

# -----------------------------------------------------------------------------
# 4. Main Interface: Bio-Signal Waveforms & Metric Cards
# -----------------------------------------------------------------------------
st.title("🌌 Life Echo: Bio-Signal Peak Detection System")

active_event = next(
    (
        e
        for e in st.session_state.events
        if e["id"] == st.session_state.selected_event_id
    ),
    st.session_state.events[0],
)

time_points, hr_s, hrv_s, gsr_s, trigger_point = generate_3track_time_series(
    active_event
)

st.markdown(
    f"### 📈 **{active_event['title']}** — [{active_event['date']}]"
)
st.caption(f"📍 **Location**: {active_event['location']} | 🏷️ **Category**: {active_event['category']}")

# Plotly Subplots
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "💓 Heart Rate Surge (HR - BPM)",
        "🧠 Heart Rate Variability (HRV RMSSD - ms)",
        "⚡ Skin Conductance Response (GSR - μS)",
    ),
)

fig.add_trace(
    go.Scatter(
        x=time_points,
        y=hr_s,
        name="Heart Rate (BPM)",
        line=dict(color="#FF4B4B", width=2.5),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=time_points,
        y=hrv_s,
        name="HRV (ms)",
        line=dict(color="#0068C9", width=2.5),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=time_points,
        y=gsr_s,
        name="GSR (μS)",
        line=dict(color="#FF9000", width=2.5),
    ),
    row=3,
    col=1,
)

for r in range(1, 4):
    y_val = hr_s[100] if r == 1 else (hrv_s[100] if r == 2 else gsr_s[100])
    fig.add_trace(
        go.Scatter(
            x=[trigger_point],
            y=[y_val],
            mode="markers+text",
            name="🚨 Historical Peak" if r == 1 else "",
            showlegend=(r == 1),
            marker=dict(color="#ff2b2b", size=14, symbol="star"),
            text=[f" Peak Detected [{active_event['date']}]"] if r == 1 else [],
            textposition="top center",
            textfont=dict(color="#ff2b2b", size=12, family="Arial Black"),
        ),
        row=r,
        col=1,
    )
    fig.add_vline(
        x=trigger_point,
        line_width=1.5,
        line_dash="dash",
        line_color="#ff2b2b",
        row=r,
        col=1,
    )

fig.update_layout(
    height=480,
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
    showlegend=True,
)
st.plotly_chart(fig, use_container_width=True)

# Metric KPI Display
hr_diff = (
    (active_event["hr_peak"] - active_event["hr_baseline"])
    / active_event["hr_baseline"]
    * 100
)
hrv_diff = (
    (active_event["hrv_rmssd"] - active_event["hrv_baseline"])
    / active_event["hrv_baseline"]
    * 100
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="Peak HR vs. Baseline",
        value=f"{active_event['hr_peak']} BPM",
        delta=f"↑ +{hr_diff:.1f}% (Baseline: {active_event['hr_baseline']} BPM)",
    )
with col2:
    st.metric(
        label="HRV Variance (RMSSD)",
        value=f"{active_event['hrv_rmssd']} ms",
        delta=f"↑ +{hrv_diff:.1f}% (Baseline: {active_event['hrv_baseline']} ms)",
    )
with col3:
    st.metric(
        label="GSR Surge (Delta)",
        value=f"{active_event['gsr_delta']} μS",
        delta="↑ Emotional Arousal",
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. Memory Vault Management & Event Selection
# -----------------------------------------------------------------------------
st.subheader("📚 Memory Vault: Review Historical Peaks & Add Reflections")

# Event Selection Buttons
cols = st.columns(len(st.session_state.events))
for idx, ev in enumerate(st.session_state.events):
    with cols[idx]:
        if st.button(
            f"📍 {ev['title']}",
            key=f"select_{ev['id']}",
            type=(
                "primary"
                if ev["id"] == st.session_state.selected_event_id
                else "secondary"
            ),
        ):
            st.session_state.selected_event_id = ev["id"]
            st.rerun()

# Editable Memory Log & Image Uploader
st.markdown("#### ✏️ Add Photos & Journal Reflections for This Timestamp")
edit_col1, edit_col2 = st.columns([1, 1.2])

with edit_col1:
    st.image(
        active_event["image"],
        caption="Current Saved Photo",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader(
        "📸 Upload / Replace Photo for This Event",
        type=["jpg", "jpeg", "png", "heic"],
        key=f"uploader_event_{active_event['id']}",
    )

    if uploaded_file is not None:
        fixed_img = safe_load_image(uploaded_file)
        if fixed_img is not None:
            # 保存到本地文件系统
            img_filename = f"event_{active_event['id']}_{int(datetime.now().timestamp())}.png"
            local_save_path = os.path.join(IMAGES_DIR, img_filename)
            fixed_img.save(local_save_path, "PNG")

            # 更新数据并写入持久化 JSON
            active_event["image"] = local_save_path
            save_persistent_events(st.session_state.events)
            st.success("Photo Permanently Saved!")
            st.rerun()

with edit_col2:
    new_desc = st.text_area(
        "📝 Record Feelings & Journal Reflections (Memory Log)",
        value=active_event["description"],
        height=150,
        key=f"desc_event_{active_event['id']}",
    )›
    
    
    btn_col1, btn_col2 = st.columns([1, 1])
    
    with btn_col1:
        if st.button("💾 Save Reflection", key=f"save_event_{active_event['id']}", use_container_width=True):
            active_event["description"] = new_desc
            save_persistent_events(st.session_state.events)
            st.success("Reflection Permanently Saved!")

    with btn_col2:
        if st.button("🗑️ Delete This Memory", key=f"del_event_{active_event['id']}", type="secondary", use_container_width=True):
            
            st.session_state.events = [e for e in st.session_state.events if e["id"] != active_event["id"]]
            
            
            save_persistent_events(st.session_state.events)
            
            
            if st.session_state.events:
                st.session_state.selected_event_id = st.session_state.events[0]["id"]
            
            
            st.rerun()