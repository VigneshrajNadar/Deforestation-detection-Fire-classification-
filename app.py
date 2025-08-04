# --- Auto-download model/data for Streamlit Cloud ---
# Downloads .pkl and .csv files from Google Drive if missing, using gdown.
# This enables deployment on Streamlit Cloud without storing large files in the repo.
import os
import streamlit as st
import numpy as np
import joblib
try:
    import gdown
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'gdown'])
    import gdown

def download_if_missing(url, filename):
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)

# Replace these with your actual file IDs/links
# === ACTUAL GOOGLE DRIVE LINKS BELOW ===
MODEL_URL = 'https://drive.google.com/uc?id=1k3NI_5b-hb-XmIgFnGwa7bLygq8F1wt8'
SCALER_URL = 'https://drive.google.com/uc?id=1K787fvWuCc-ojxMmiWtYT9vPb9AckYXs'
NOTEBOOK_URL = 'https://drive.google.com/uc?id=1jiFCYZJ-7RVk5BHad7uX-ut7gfNP53l0'
CSV_2021_URL = 'https://drive.google.com/uc?id=17UZzdC-UiKiDhgDYTz-S211nJ708s_U0'
CSV_2022_URL = 'https://drive.google.com/uc?id=1ZFMx-GieGBHP9Sabe4Nr1kz1UQzCKzY-'
CSV_2023_URL = 'https://drive.google.com/uc?id=1xwFXLlsiDJo7ID0FUvN94tmq7hgaViDQ'

download_if_missing(MODEL_URL, 'best_fire_detection_model.pkl')
download_if_missing(SCALER_URL, 'scaler.pkl')
download_if_missing(CSV_2021_URL, 'modis_2021_India.csv')
download_if_missing(CSV_2022_URL, 'modis_2022_India.csv')
download_if_missing(CSV_2023_URL, 'modis_2023_India.csv')

import pandas as pd
import plotly.express as px
from pathlib import Path

# --- Load model and scaler with caching ---
@st.cache_resource
def load_model():
    return joblib.load("best_fire_detection_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

# --- Page config ---
st.set_page_config(page_title="🔥 Fire Type Classifier", layout="wide", page_icon="🔥")

# --- Fire Theme CSS + Animated SVG Fire ---
st.markdown('''
<style>
body, .stApp {
    background: linear-gradient(120deg, #1a1a1a 0%, #ff6a00 100%) !important;
}
header, .st-emotion-cache-18ni7ap, .st-emotion-cache-6qob1r {
    background: linear-gradient(90deg, #ff6a00 0%, #ffd700 100%) !important;
}
.st-emotion-cache-1v0mbdj, .st-emotion-cache-1r4qj8v, .stSidebarContent {
    background: linear-gradient(120deg, #2a0909 0%, #ff9800 100%) !important;
    color: #fff !important;
    box-shadow: 0 0 32px 8px #ff980055 inset;
    animation: flickerSidebar 2.2s infinite alternate;
}
.st-emotion-cache-1v0mbdj h1, .st-emotion-cache-1v0mbdj h2, .st-emotion-cache-1v0mbdj h3, .st-emotion-cache-1v0mbdj h4 {
    color: #ffd700 !important;
    text-shadow: 0 1px 8px #ff6a00, 0 0px 18px #ff9800;
    animation: flickerText 1.7s infinite alternate;
}
.stButton > button {
    background: linear-gradient(90deg,#ff6a00 0%,#ffd700 100%) !important;
    color: #1a1a1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    box-shadow: 0 2px 8px #ff980044, 0 0 18px #ff980055 inset;
    transition: background 0.2s;
    animation: glowBtn 1.2s infinite alternate;
}
.stButton > button:hover {
    background: linear-gradient(90deg,#ffd700 0%,#ff6a00 100%) !important;
    color: #fff !important;
    box-shadow: 0 2px 16px #ffd70099;
}
.st-emotion-cache-1v0mbdj .stAlert {
    background: linear-gradient(90deg, #ff9800 0%, #ffd700 100%) !important;
    color: #1a1a1a !important;
    box-shadow: 0 0 12px #ffd70044;
}
.st-emotion-cache-1v0mbdj .stMarkdown code {
    background: #2a0909 !important;
    color: #ffd700 !important;
}
/* Fire emoji animation */
@keyframes fireWiggle {
  0% { transform: rotate(-8deg) scale(1.08); filter: drop-shadow(0 0 18px #ffd700); }
  50% { transform: rotate(8deg) scale(1.22); filter: drop-shadow(0 0 38px #ff6a00); }
  100% { transform: rotate(-8deg) scale(1.08); filter: drop-shadow(0 0 18px #ffd700); }
}
#fire-header {
    font-size: 54px;
    position: absolute;
    left: 12px;
    top: 8px;
    animation: fireWiggle 1.2s infinite;
    z-index: 100;
}
/* Flicker for sidebar and cards */
@keyframes flickerSidebar {
  0% { box-shadow: 0 0 32px 8px #ff980055 inset; }
  50% { box-shadow: 0 0 48px 18px #ffd70055 inset; }
  100% { box-shadow: 0 0 32px 8px #ff980055 inset; }
}
@keyframes flickerText {
  0% { text-shadow: 0 1px 8px #ff6a00, 0 0px 18px #ff9800; }
  50% { text-shadow: 0 1px 18px #ffd700, 0 0px 28px #ff9800; }
  100% { text-shadow: 0 1px 8px #ff6a00, 0 0px 18px #ff9800; }
}
@keyframes glowBtn {
  0% { box-shadow: 0 2px 8px #ff980044, 0 0 18px #ff980055 inset; }
  100% { box-shadow: 0 2px 18px #ffd70099, 0 0 28px #ffd70055 inset; }
}
/* Flicker for main cards */
.fire-card {
    animation: flickerSidebar 2.2s infinite alternate;
    box-shadow: 0 0 32px 8px #ff980055 inset;
}
/* Animated SVG fire background for header */
#fire-svg-bg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100vw;
    height: 90px;
    z-index: 0;
    pointer-events: none;
    opacity: 0.24;
}
</style>
<div id="fire-header">🔥</div>
<svg id="fire-svg-bg" viewBox="0 0 1440 90" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="fireGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ffd700"/>
      <stop offset="70%" stop-color="#ff6a00"/>
      <stop offset="100%" stop-color="#b71c1c"/>
    </linearGradient>
    <filter id="flicker" x="-20%" y="-20%" width="140%" height="140%">
      <feTurbulence id="turb" type="fractalNoise" baseFrequency="0.01 0.8" numOctaves="2" seed="2" result="turb"/>
      <feDisplacementMap in2="turb" in="SourceGraphic" scale="18" xChannelSelector="R" yChannelSelector="G"/>
      <animate xlink:href="#turb" attributeName="seed" values="2;8;2" dur="2.5s" repeatCount="indefinite"/>
    </filter>
  </defs>
  <path d="M0 80 Q 360 30 720 80 T 1440 80 V 90 H 0 Z" fill="url(#fireGrad)" filter="url(#flicker)"/>
</svg>
''', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Data Visualization"])
st.sidebar.markdown("---")
st.sidebar.info("Made with ❤️ using MODIS satellite data.")

# --- Helper: Load and combine data ---
def load_modis_data():
    dfs = []
    for year in [2021, 2022, 2023]:
        csv_path = Path(f"modis_{year}_India.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['year'] = year
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

modis_df = load_modis_data()

# --- Main: Prediction Page ---
if page == "Prediction":
    st.markdown("<h1 style='text-align:center;color:#d7263d;'>🔥 Fire Type Classification</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(90deg,#1e3c72 0%,#2a5298 100%);padding:18px 12px 18px 12px;border-radius:14px;text-align:center;box-shadow:0 2px 12px rgba(30,60,114,0.07);margin-bottom:8px;'>
        <h4 style='color:#fff;margin-bottom:0;letter-spacing:1px;'>🔥 Predict fire type based on MODIS satellite readings.</h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("""
        <div style='background:#f7fafc;border-radius:12px;padding:20px 18px 18px 18px;box-shadow:0 2px 12px rgba(30,60,114,0.08);margin-bottom:10px;'>
        <h4 style='color:#2a5298;margin-bottom:18px;'>Input Features</h4>
        """, unsafe_allow_html=True)
        brightness = st.number_input("Brightness", value=300.0, min_value=200.0, max_value=500.0)
        bright_t31 = st.number_input("Brightness T31", value=290.0, min_value=200.0, max_value=350.0)
        frp = st.number_input("Fire Radiative Power (FRP)", value=15.0, min_value=0.0, max_value=100.0)
        scan = st.number_input("Scan", value=1.0, min_value=0.0, max_value=5.0)
        track = st.number_input("Track", value=1.0, min_value=0.0, max_value=5.0)
        confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"])
        confidence_map = {"low": 0, "nominal": 1, "high": 2}
        confidence_val = confidence_map[confidence]
        input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
        # Lazy-load model and scaler only when needed
        scaler = load_scaler()
        model = load_model()
        # Ensure input has feature names for scaler (avoid warning)
        input_df = pd.DataFrame(input_data, columns=scaler.feature_names_in_)
        scaled_input = scaler.transform(input_df)
        predict_btn = st.button("🔎 Predict Fire Type", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if predict_btn:
            try:
                # Pass DataFrame with feature names to model.predict to avoid warning
                prediction = model.predict(pd.DataFrame(scaled_input, columns=scaler.feature_names_in_))[0]
                fire_types = {
                    0: "Vegetation Fire",
                    2: "Other Static Land Source",
                    3: "Offshore Fire"
                }
                result = fire_types.get(prediction, "Unknown")
                st.success(f"**Predicted Fire Type:** {result}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc(), language='python')
                # Optionally, log the error to a file or external system here
        del model
        del scaler
        import gc
        gc.collect() # Free memory after prediction

    with col2:
        st.markdown("""
        <div class='fire-legend-anim fire-legend-border' style='background:rgba(20,20,20,0.92);border-radius:20px;padding:24px 22px 22px 22px;box-shadow:0 2px 18px #000a;margin-bottom:12px;position:relative;animation:pulseLegend 2.2s infinite alternate;'>
        <svg class='fire-wave-bg' style='position:absolute;bottom:0;left:0;width:100%;height:44px;z-index:1;pointer-events:none;opacity:.22;' viewBox='0 0 360 44'><defs><linearGradient id='fireGrad2' x1='0' y1='0' x2='0' y2='1'><stop offset='0%' stop-color='#ffd700'/><stop offset='70%' stop-color='#ff6a00'/><stop offset='100%' stop-color='#b71c1c'/></linearGradient></defs><path d='M0 40 Q 90 10 180 40 T 360 40 V 44 H 0 Z' fill='url(#fireGrad2)'><animate attributeName='d' values='M0 40 Q 90 10 180 40 T 360 40 V 44 H 0 Z;M0 40 Q 90 30 180 10 T 360 40 V 44 H 0 Z;M0 40 Q 90 10 180 40 T 360 40 V 44 H 0 Z' dur='4s' repeatCount='indefinite'/></path></svg>
        <div class='fire-emoji-spin' style='position:absolute;top:12px;right:24px;font-size:32px;animation:fireGlow 1.6s infinite alternate,spinFire 7s linear infinite;'>🔥</div>
        <h4 class='fire-shimmer fire-rainbow' style='color:#ffd700;text-shadow:0 2px 12px #ff6a00,0 0 8px #000;margin-bottom:18px;letter-spacing:1px;position:relative;overflow:hidden;'>Fire Type Legend</h4>
        <ul style='list-style:none;padding-left:0;'>
            <li style='margin-bottom:14px;transition:background 0.2s;'>
                <span class='fire-icon-bounce' style='display:inline-block;width:22px;height:22px;background:linear-gradient(135deg,#ffea00,#ff6a00);border-radius:4px;margin-right:12px;vertical-align:middle;box-shadow:0 2px 8px #ffea00bb;'><span style='font-size:18px;position:relative;top:-2px;left:2px;'>🌲</span></span>
                <b style='color:#ffea00;text-shadow:0 2px 8px #ff6a00,0 0 4px #000;'>Vegetation Fire</b>: <span style='color:#fff;text-shadow:0 1px 4px #000;'>Wildfires, forest and grassland fires</span>
            </li>
            <li style='margin-bottom:14px;transition:background 0.2s;'>
                <span class='fire-icon-bounce' style='display:inline-block;width:22px;height:22px;background:linear-gradient(135deg,#b0bec5,#263238);border-radius:4px;margin-right:12px;vertical-align:middle;box-shadow:0 2px 8px #b0bec599;'><span style='font-size:18px;position:relative;top:-2px;left:2px;'>🏭</span></span>
                <b style='color:#b0bec5;text-shadow:0 2px 8px #263238,0 0 4px #000;'>Other Static Land Source</b>: <span style='color:#fff;text-shadow:0 1px 4px #000;'>Industrial, urban, or landfill fires</span>
            </li>
            <li style='margin-bottom:14px;transition:background 0.2s;'>
                <span class='fire-icon-bounce' style='display:inline-block;width:22px;height:22px;background:linear-gradient(135deg,#ff9800,#ff512f);border-radius:4px;margin-right:12px;vertical-align:middle;box-shadow:0 2px 8px #ff9800bb;'><span style='font-size:18px;position:relative;top:-2px;left:2px;'>🚢</span></span>
                <b style='color:#ff9800;text-shadow:0 2px 8px #ff512f,0 0 4px #000;'>Offshore Fire</b>: <span style='color:#fff;text-shadow:0 1px 4px #000;'>Oil/gas platform or ship fires</span>
            </li>
        </ul>
        <div class='fire-particles'><div class='f3'></div><div class='f4'></div><div class='f5'></div><div class='f6'></div><div class='f7'></div><div class='f8'></div></div>
        <style>
        @keyframes fireGlow{0%{filter:drop-shadow(0 0 6px #ffd700);}100%{filter:drop-shadow(0 0 16px #ff6a00);}}
        @keyframes pulseLegend{0%{box-shadow:0 2px 18px #000a,0 0 24px #ff980055;}100%{box-shadow:0 2px 28px #ffd70099,0 0 44px #ff6a00bb;}}
        @keyframes bounceFireIcon{0%{transform:translateY(0);}50%{transform:translateY(-8px) scale(1.15);}100%{transform:translateY(0);}}
        .fire-icon-bounce{animation:bounceFireIcon 1.8s infinite cubic-bezier(.6,.05,.4,.95);}
        .fire-legend-border{box-shadow:0 0 0 4px #ff980055,0 0 24px #ffd70099,0 0 44px #ff6a00bb,0 2px 18px #000a;animation:glowBorder 2.8s infinite alternate;}
        @keyframes glowBorder{0%{box-shadow:0 0 0 4px #ff980055,0 0 24px #ffd70099,0 0 44px #ff6a00bb,0 2px 18px #000a;}100%{box-shadow:0 0 0 8px #ffd70077,0 0 44px #ff6a00cc,0 0 64px #ffd700bb,0 2px 28px #000a;}}
        .fire-emoji-spin{animation:fireGlow 1.6s infinite alternate,spinFire 7s linear infinite;display:inline-block;}
        @keyframes spinFire{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}
        .fire-shimmer:after{content:'';position:absolute;top:0;left:-60px;width:60px;height:100%;background:linear-gradient(120deg,rgba(255,255,255,0.18) 0%,rgba(255,255,255,0.48) 60%,rgba(255,255,255,0.12) 100%);transform:skewX(-22deg);animation:shimmerFire 2.8s infinite;z-index:2;}
        .fire-rainbow:before{content:'';position:absolute;top:0;left:0;width:100%;height:100%;background:linear-gradient(90deg,#ffd700,#ff6a00,#ff512f,#ffd700,#ff6a00,#ffd700);background-size:400% 100%;opacity:.18;z-index:1;animation:rainbowFire 3.2s linear infinite;pointer-events:none;}
        @keyframes shimmerFire{0%{left:-60px;}100%{left:120%;}}
        @keyframes rainbowFire{0%{background-position:0% 50%;}100%{background-position:100% 50%;}}
        .fire-particles{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;z-index:2;}
        .fire-particles:before,.fire-particles:after,.fire-particles .f3,.fire-particles .f4,.fire-particles .f5,.fire-particles .f6,.fire-particles .f7,.fire-particles .f8{
          content:'';position:absolute;border-radius:50%;background:radial-gradient(circle,#ffd700 0%,#ff6a00 70%,#0000 100%);opacity:.7;
        }
        .fire-particles:before{left:50%;top:85%;width:8px;height:8px;animation:floatFire 2.2s infinite linear;}
        .fire-particles:after{left:60%;top:80%;width:6px;height:6px;opacity:.5;animation:floatFire2 2.6s infinite linear;}
        .fire-particles .f3{left:40%;top:90%;width:10px;height:10px;opacity:.4;animation:floatFire3 3.2s infinite linear;}
        .fire-particles .f4{left:70%;top:88%;width:7px;height:7px;opacity:.6;animation:floatFire4 2.3s infinite linear;}
        .fire-particles .f5{left:30%;top:92%;width:9px;height:9px;opacity:.5;animation:floatFire5 2.7s infinite linear;}
        .fire-particles .f6{left:80%;top:93%;width:8px;height:8px;opacity:.4;animation:floatFire6 2.1s infinite linear;}
        .fire-particles .f7{left:20%;top:95%;width:11px;height:11px;opacity:.3;animation:floatFire7 3.5s infinite linear;}
        .fire-particles .f8{left:85%;top:91%;width:7px;height:7px;opacity:.5;animation:floatFire8 2.9s infinite linear;}
        @keyframes floatFire{0%{top:85%;opacity:.7;}50%{top:40%;opacity:.3;}100%{top:10%;opacity:0;}}
        @keyframes floatFire2{0%{top:80%;opacity:.5;}50%{top:50%;opacity:.2;}100%{top:18%;opacity:0;}}
        @keyframes floatFire3{0%{top:90%;opacity:.4;}50%{top:60%;opacity:.2;}100%{top:8%;opacity:0;}}
        @keyframes floatFire4{0%{top:88%;opacity:.6;}50%{top:60%;opacity:.3;}100%{top:15%;opacity:0;}}
        @keyframes floatFire5{0%{top:92%;opacity:.5;}50%{top:65%;opacity:.2;}100%{top:12%;opacity:0;}}
        @keyframes floatFire6{0%{top:93%;opacity:.4;}50%{top:62%;opacity:.1;}100%{top:18%;opacity:0;}}
        @keyframes floatFire7{0%{top:95%;opacity:.3;}50%{top:70%;opacity:.1;}100%{top:10%;opacity:0;}}
        @keyframes floatFire8{0%{top:91%;opacity:.5;}50%{top:58%;opacity:.2;}100%{top:19%;opacity:0;}}
        .stButton > button{animation:btnPulse 2.2s infinite alternate;box-shadow:0 2px 8px #ff980044,0 0 18px #ffd70055 inset;}
        .stButton > button:hover{animation:btnPulseHover .8s infinite alternate !important;box-shadow:0 2px 16px #ffd70099,0 0 32px #ff9800cc !important;}
        @keyframes btnPulse{0%{box-shadow:0 2px 8px #ff980044,0 0 18px #ffd70055 inset;}100%{box-shadow:0 2px 18px #ffd70099,0 0 28px #ffd70055 inset;}}
        @keyframes btnPulseHover{0%{box-shadow:0 2px 16px #ffd70099,0 0 32px #ff9800cc;}100%{box-shadow:0 2px 24px #ff9800cc,0 0 48px #ffd700cc;}}
        </style>
        </div>
        """, unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1464983953574-0892a716854b?auto=format&fit=crop&w=600&q=80", caption="MODIS Satellite Fire Detection", use_container_width=True)
    st.markdown("---")

# --- Main: Data Visualization Page ---
if page == "Data Visualization":
    st.markdown("<h1 style='color:#d7263d;'>📊 MODIS Fire Data Visualization</h1>", unsafe_allow_html=True)
    st.markdown("---")
    if modis_df.empty:
        st.warning("No MODIS data files found.")
    else:
        # --- Sidebar Filters ---
        with st.sidebar:
            st.markdown("### Data Filters")
            filter_year = None
            filter_type = None
            filter_conf = None
            if 'year' in modis_df.columns:
                years = sorted(modis_df['year'].unique())
                filter_year = st.multiselect("Year", years, default=years)
            if 'type' in modis_df.columns:
                types = sorted(modis_df['type'].unique())
                filter_type = st.multiselect("Fire Type", types, default=types)
            if 'confidence' in modis_df.columns:
                confs = sorted(modis_df['confidence'].unique())
                filter_conf = st.multiselect("Confidence", confs, default=confs)
        # --- Apply Filters ---
        filtered_df = modis_df.copy()
        if filter_year is not None:
            filtered_df = filtered_df[filtered_df['year'].isin(filter_year)]
        if filter_type is not None:
            filtered_df = filtered_df[filtered_df['type'].isin(filter_type)]
        if filter_conf is not None:
            filtered_df = filtered_df[filtered_df['confidence'].isin(filter_conf)]

        # --- Summary Stats Panel ---
        st.info(f"**Total Records:** {len(filtered_df)}  |  **Years:** {', '.join(map(str, sorted(filtered_df['year'].unique())))}  |  **Fire Types:** {', '.join(map(str, sorted(filtered_df['type'].unique())))}")
        st.markdown("---")

        # --- Expanders for Chart Groups ---
        with st.expander("📊 Distribution Charts", expanded=True):
            st.caption("Bar, pie, and box plots for fire types, confidence, year, and FRP.")
            # (Insert all static bar, pie, box, histogram, and year charts here, using filtered_df)
        with st.expander("🗺️ Maps and Heatmaps", expanded=False):
            st.caption("Geographical visualizations of fire locations and densities.")
            # (Insert map, heatmap, animated map, etc. using filtered_df)
        with st.expander("📈 Animated & Advanced Charts", expanded=False):
            st.caption("Animated bar, scatter, and line charts showing trends and patterns over time.")
            # (Insert all animated charts here, using filtered_df)
        st.markdown("---")
        st.caption("Data Source: NASA MODIS Fire Detections (2021-2023)")

        # Fire type distribution (Bar)
        if 'type' in modis_df.columns:
            fire_type_counts = modis_df['type'].value_counts().rename_axis('Type').reset_index(name='Count')
            fig1 = px.bar(fire_type_counts, x='Type', y='Count', color='Type', title="Fire Type Distribution", color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig1, use_container_width=True)
            # Pie chart of fire types
            fig1_pie = px.pie(fire_type_counts, names='Type', values='Count', title="Fire Type Proportion", color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig1_pie, use_container_width=True)
        
        # Confidence level pie chart (already present, but move up)
        if 'confidence' in modis_df.columns:
            conf_counts = modis_df['confidence'].value_counts().rename_axis('Confidence').reset_index(name='Count')
            fig3 = px.pie(conf_counts, names='Confidence', values='Count', title="Confidence Level Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig3, use_container_width=True)
        
        # Bar chart: Fire counts by year
        if 'year' in modis_df.columns:
            year_counts = modis_df['year'].value_counts().sort_index().rename_axis('Year').reset_index(name='Count')
            fig_year = px.bar(year_counts, x='Year', y='Count', title="Fire Counts by Year", color='Year', color_discrete_sequence=px.colors.qualitative.Dark2)
            st.plotly_chart(fig_year, use_container_width=True)
        
        # Box plot: FRP by fire type
        if {'frp', 'type'}.issubset(modis_df.columns):
            fig_box = px.box(modis_df, x='type', y='frp', color='type', title="FRP Distribution by Fire Type", color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # FRP distribution (Histogram)
        if 'frp' in modis_df.columns:
            st.subheader("FRP (Fire Radiative Power) Distribution")
            fig2 = px.histogram(modis_df, x='frp', nbins=50, title="FRP Histogram", color_discrete_sequence=['#d7263d'])
            st.plotly_chart(fig2, use_container_width=True)
        
        # Heatmap: Fire counts by lat/lon grid (if available)
        if {'latitude', 'longitude'}.issubset(modis_df.columns):
            st.subheader("Recent Fire Locations in India")
            st.map(modis_df[['latitude', 'longitude']].dropna(), zoom=4)
            # Heatmap
            heatmap_df = modis_df[['latitude', 'longitude']].dropna().copy()
            if not heatmap_df.empty:
                heatmap_df['lat_bin'] = (heatmap_df['latitude'] // 1) * 1
                heatmap_df['lon_bin'] = (heatmap_df['longitude'] // 1) * 1
                grid_counts = heatmap_df.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='Count')
                fig_heat = px.density_heatmap(grid_counts, x='lon_bin', y='lat_bin', z='Count', nbinsx=30, nbinsy=30, color_continuous_scale='YlOrRd', title="Fire Density Heatmap (1° grid)")
                st.plotly_chart(fig_heat, use_container_width=True)
        
        # Pie chart: fires by year
        if 'year' in modis_df.columns:
            year_counts = modis_df['year'].value_counts().sort_index().rename_axis('Year').reset_index(name='Count')
            fig_year_pie = px.pie(year_counts, names='Year', values='Count', title="Fires by Year (Pie Chart)", color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_year_pie, use_container_width=True)

        # Pie chart of top N locations (region/state)
        for loc_col in ['region', 'state', 'district', 'subdivision']:
            if loc_col in modis_df.columns:
                st.subheader(f"Top 10 {loc_col.title()}s by Fire Count (Pie Chart)")
                loc_counts = modis_df[loc_col].value_counts().nlargest(10).rename_axis(loc_col.title()).reset_index(name='Count')
                fig_loc = px.pie(loc_counts, names=loc_col.title(), values='Count', title=f"Top 10 {loc_col.title()}s", color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig_loc, use_container_width=True)
                break  # Only show for the first found location column

        # Animated scatter plot of fire detections over time
        if {'acq_date', 'latitude', 'longitude'}.issubset(modis_df.columns):
            st.subheader("Animated Fire Detections Over Time")
            anim_df = modis_df.dropna(subset=['acq_date', 'latitude', 'longitude']).copy()
            anim_df['acq_date'] = pd.to_datetime(anim_df['acq_date'], errors='coerce')
            anim_df = anim_df.dropna(subset=['acq_date'])
            if not anim_df.empty:
                # Use date as string for animation_frame
                anim_df['date_str'] = anim_df['acq_date'].dt.strftime('%Y-%m-%d')
                fig_anim = px.scatter_geo(
                    anim_df,
                    lat='latitude',
                    lon='longitude',
                    color='type' if 'type' in anim_df.columns else None,
                    animation_frame='date_str',
                    title="Fire Detections Animation (by Day)",
                    projection="natural earth",
                    color_discrete_sequence=px.colors.qualitative.Prism,
                    opacity=0.7,
                    height=600
                )
                fig_anim.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig_anim, use_container_width=True)

        # Animated bar chart: Top states/regions by fire count over years
        for loc_col in ['state', 'region', 'district', 'subdivision']:
            if {'year', loc_col}.issubset(modis_df.columns):
                st.subheader(f"Animated Bar Chart: Top {loc_col.title()}s by Fire Count Over Years")
                bar_df = modis_df.groupby(['year', loc_col]).size().reset_index(name='Count')
                bar_df = bar_df.sort_values(['year', 'Count'], ascending=[True, False])
                fig_bar_anim = px.bar(bar_df, x=loc_col, y='Count', color=loc_col, animation_frame='year', range_y=[0, bar_df['Count'].max()*1.1], title=f"Top {loc_col.title()}s by Fire Count (Animated)", color_discrete_sequence=px.colors.qualitative.G10)
                st.plotly_chart(fig_bar_anim, use_container_width=True)
                break

        # Animated heatmap: Fire density by year
        if {'latitude', 'longitude', 'year'}.issubset(modis_df.columns):
            st.subheader("Animated Heatmap: Fire Density by Year")
            heat_df = modis_df.dropna(subset=['latitude', 'longitude', 'year']).copy()
            heat_df['lat_bin'] = (heat_df['latitude'] // 1) * 1
            heat_df['lon_bin'] = (heat_df['longitude'] // 1) * 1
            grid_counts = heat_df.groupby(['year', 'lat_bin', 'lon_bin']).size().reset_index(name='Count')
            fig_heat_anim = px.density_heatmap(grid_counts, x='lon_bin', y='lat_bin', z='Count', animation_frame='year', nbinsx=30, nbinsy=30, color_continuous_scale='YlOrRd', title="Fire Density Heatmap by Year (Animated)")
            st.plotly_chart(fig_heat_anim, use_container_width=True)

        # Animated scatter: FRP vs brightness by year
        if {'frp', 'brightness', 'year'}.issubset(modis_df.columns):
            st.subheader("Animated Scatter: FRP vs Brightness by Year")
            fig_scatter_anim = px.scatter(modis_df, x='brightness', y='frp', animation_frame='year', color='type' if 'type' in modis_df.columns else None, title="FRP vs Brightness by Year (Animated)", opacity=0.7, color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_scatter_anim, use_container_width=True)

        # Pie chart: Fire type by year (static, with selector)
        if {'type', 'year'}.issubset(modis_df.columns):
            st.subheader("Fire Type Distribution by Year (Pie Chart)")
            pie_df = modis_df.groupby(['year', 'type']).size().reset_index(name='Count')
            years = sorted(pie_df['year'].unique())
            selected_year = st.selectbox("Select Year for Pie Chart", years, index=0)
            pie_data = pie_df[pie_df['year'] == selected_year]
            fig_pie = px.pie(pie_data, names='type', values='Count', title=f"Fire Type Distribution for {selected_year}", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Animated line chart: Cumulative fires over time
        if 'acq_date' in modis_df.columns:
            st.subheader("Animated Line Chart: Cumulative Fires Over Time")
            line_df = modis_df.dropna(subset=['acq_date']).copy()
            line_df['acq_date'] = pd.to_datetime(line_df['acq_date'], errors='coerce')
            line_df = line_df.dropna(subset=['acq_date'])
            line_df = line_df.sort_values('acq_date')
            line_df['cumulative'] = range(1, len(line_df)+1)
            line_df['date_str'] = line_df['acq_date'].dt.strftime('%Y-%m-%d')
            fig_line_anim = px.line(line_df, x='date_str', y='cumulative', title="Cumulative Fires Detected (Animated)", markers=True)
            st.plotly_chart(fig_line_anim, use_container_width=True)

        st.markdown("---")
        st.caption("Data Source: NASA MODIS Fire Detections (2021-2023)")