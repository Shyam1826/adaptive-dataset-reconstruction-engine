"""
AutoDataPrep Web Interface
Clean, minimal, and premium data processing web app powered by Streamlit and AutoDataPrepPipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure pipeline modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.auto_prep import AutoDataPrepPipeline

# Initialize Canvas Viewport Config
st.set_page_config(
    page_title="AutoDataPrep | Modern Data Processing Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State tracking for the custom theme if not present
if "app_theme" not in st.session_state:
    st.session_state["app_theme"] = "light"

def main():
    # ─── NAVIGATION BAR LIGHT/DARK MULTI-THEME TOGGLE ────────────────────
    # Create an asymmetrical column layout at the very top of the main canvas
    nav_col1, nav_col2 = st.columns([12, 1])
    
    with nav_col1:
        # Application Title Headers
        st.markdown("<h1 class='app-title' style='margin-top: -25px;'>⚡ AutoDataPrep</h1>", unsafe_allow_html=True)
    
    with nav_col2:
        # Minimalist toggle using only Sun and Moon icons
        current_theme = st.session_state["app_theme"]
        theme_icon = "🌙" if current_theme == "light" else "☀️"
        
        # Explicit style container wrapper to align button cleanly with the header text
        st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
        if st.button(theme_icon, help="Toggle Interface Theme", use_container_width=True):
            if current_theme == "light":
                st.session_state["app_theme"] = "dark"
            else:
                st.session_state["app_theme"] = "light"
            st.rerun()

    # Dynamic Theme CSS Engine Configuration
    if st.session_state["app_theme"] == "dark":
        bg_color = "#0f172a"        # Deep slate dark background
        text_color = "#f8fafc"      # Crisp white text
        card_bg = "#1e293b"         # Lighter slate for inner containers
        sidebar_bg = "#1e293b"      # Match sidebar to dark container palette
        subtitle_color = "#94a3b8"  # Soft gray text
        uploader_border = "#334155" # Slate gray border line
        sidebar_arrow_color = "#f8fafc" # Force arrow icon to white
    else:
        bg_color = "#ffffff"        # Crisp white background
        text_color = "#0f172a"      # Dark text
        card_bg = "#f8fafc"         # Soft gray-white for inner containers
        sidebar_bg = "#f1f5f9"      # Clean cool light-gray sidebar base
        subtitle_color = "#6b7280"  # Slate gray text
        uploader_border = "#e2e8f0" # Light gray border line
        sidebar_arrow_color = "#0f172a" # Force arrow icon to dark slate

    # Premium Google Typeface Branding Stylesheet Injection with full Global Overrides
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght=300;400;500;600;700&display=swap');
        
        /* Apply dynamic theme colors to the MAIN app container */
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
            font-family: 'Outfit', sans-serif;
        }}
        
        /* Apply dynamic theme colors to the SIDEBAR container */
        [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
            background-color: {sidebar_bg} !important;
            color: {text_color} !important;
        }}
        
        /* Ensure text inside sidebar matches the theme palette changes */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
            color: {text_color} !important;
        }}
        
        /* Style background metrics cards and inner containers dynamically */
        [data-testid="stMetricContainer"], .stDataFrame, div[data-testid="stBlock"] {{
            background-color: {card_bg} !important;
            border-radius: 8px;
        }}

        /* 📁 BULLETPROOF UPLOAD ZONE OVERRIDES FOR SHADOW DOM ELEMENT LEAKS */
        [data-testid="stFileUploader"], 
        [data-testid="stFileUploader"] > section, 
        [data-testid="stFileUploader"] div[role="button"],
        .stFileUploaderDropzone {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
            border-color: {uploader_border} !important;
        }}
        
        /* Override internal child text vectors inside the drag boxes */
        [data-testid="stFileUploader"] text,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] p {{
            color: {text_color} !important;
        }}
        
        /* Ensure the individual text items inside the drag box look sharp */
        div[data-testid="stFileUploaderDropzone"] {{
            border: 2px dashed {uploader_border} !important;
            background-color: {card_bg} !important;
        }}
        
        /* ✅ FIXED SIDEBAR ARROW VISIBILITY BUTTON */
        [data-testid="stSidebarCollapseButton"] {{
            background-color: transparent !important;
            z-index: 999999 !important;
        }}
        [data-testid="stSidebarCollapseButton"] button {{
            color: {sidebar_arrow_color} !important;
            background-color: {card_bg} !important;
            border-radius: 50% !important;
            border: 1px solid {uploader_border} !important;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1) !important;
        }}
        [data-testid="stSidebarCollapseButton"] svg {{
            fill: {sidebar_arrow_color} !important;
            color: {sidebar_arrow_color} !important;
        }}
        
        /* Hide developer / hosting bars */
        header, [data-testid="stHeader"], .stAppHeader {{
            background-color: transparent !important;
            background: transparent !important;
        }}
        .stAppDeployButton, [data-testid="stAppDeployButton"], [data-testid="stDecoration"] {{
            display: none !important;
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        [data-testid="stStatusWidget"] {{display: none !important;}}
        
        .app-title {{
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            margin-bottom: 0.2rem;
        }}
        
        .app-subtitle {{
            font-size: 1.1rem;
            color: {subtitle_color};
            margin-bottom: 2rem;
            font-weight: 400;
        }}
        </style>
        """, unsafe_allow_html=True)

    st.markdown("<p class='app-subtitle'>Downstream Adaptive Dataset Profiling, Cleaning, & Model Preprocessing Engine</p>", unsafe_allow_html=True)

    # Instantiate layout tracking holders
    raw_df = None
    available_columns = []
    default_index = 0

    # Main Workspace Upload Zone Canvas
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload raw messy datasets (CSV or Excel formats)", 
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )

    # Process file upload data-stream immediately to feed reactive sidebar features
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
                
            available_columns = raw_df.columns.tolist()
            
            # Predict probable label column matches
            common_targets = ['selling_price', 'price', 'is_churn', 'target', 'label', 'churn']
            for target_word in common_targets:
                if target_word in available_columns:
                    default_index = available_columns.index(target_word)
                    break
        except Exception as e:
            st.error(f"Error parsing loaded configuration blocks: {e}")
            return

    # Sidebar Parameters Workspace Config
    st.sidebar.markdown("### ⚙️ Pipeline Configuration")
    
    if uploaded_file is not None and len(available_columns) > 0:
        target_column = st.sidebar.selectbox(
            "Target Column Name",
            options=available_columns,
            index=default_index,
            help="Select the column you want your machine learning model to learn how to predict."
        )
    else:
        target_column = st.sidebar.text_input(
            "Target Column Name",
            value="",
            disabled=True,
            placeholder="Upload data file first..."
        )
        
    isolate_target = st.sidebar.checkbox(
        "Isolate Target Column", 
        value=True, 
        help="Bypasses and preserves target values from standard scaling loops."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Distribution & Cardinality Help")
    
    with st.sidebar.expander("❓ Not sure what these numbers mean?", expanded=False):
        st.markdown("""
        **1. Skewness (Default: 0.5)**
        * Measures the lopsidedness of numerical data. Values above this limit force robust median tracking instead of standard means.
        
        **2. Correlation Threshold (Default: 0.85)**
        * Identifies feature-to-feature overlaps. If two columns tell the same story, it drops the one with a weaker relationship to the target column.
        
        **3. Categorical Limit (Default: 10)**
        * The maximum unique option list allowed for text columns before switching from One-Hot Encoding to Frequency mapping.
        
        **4. Noise Threshold (Default: 0.80)**
        * Automatically strips away entirely unique structural elements like tracking hashes, system IDs, or primary keys that don't add predictive value to your models.
        """)
    
    st.sidebar.markdown("---")
    
    # Threshold sliders setup
    skewness_threshold = st.sidebar.slider("Skewness Threshold", 0.1, 2.0, 0.5, 0.1)
    correlation_threshold = st.sidebar.slider("Feature-to-Feature Correlation Limit", 0.50, 0.99, 0.85, 0.05)
    cardinality_threshold = st.sidebar.slider("Categorical Limit (Cardinality)", 2, 50, 10, 1)
    noise_threshold = st.sidebar.slider("Noise Threshold (High Cardinality)", 0.50, 1.00, 0.80, 0.05)

    # Active Computation Workspace Processing Layout Routing
    if uploaded_file is not None and raw_df is not None:
        st.markdown(f"**Loaded Dataset:** `{uploaded_file.name}` | **Shape:** `{raw_df.shape[0]} rows x {raw_df.shape[1]} columns`")
        
        st.markdown("---")
        st.markdown("### 🎛️ Select your desired data output target:")
        output_target = st.radio(
            "Output target selection",
            [
                "Human-Readable Cleaned Dataset Only (Optimized for Analysts & BI)",
                "Machine Learning Ready Matrix Only (Fully Encoded, Scaled, & Pruned)",
                "Dual Export (Generate Both Datasets Concurrently)"
            ],
            label_visibility="collapsed"
        )
        
        # Smart dynamic checking criteria validations
        is_ml_required = "Machine Learning" in output_target or "Dual Export" in output_target
        active_target = target_column.strip() if target_column and target_column.strip() in raw_df.columns else None
        
        if is_ml_required and isolate_target and not active_target:
            st.error("⚠️ Please choose a valid predictive target column option in the left configuration dropdown menu.")
            return

        # Core composite memory caching hashes
        cache_hash = (
            f"{uploaded_file.name}_{uploaded_file.size}_"
            f"{active_target}_{isolate_target}_{skewness_threshold}_"
            f"{correlation_threshold}_{cardinality_threshold}_{noise_threshold}_{output_target}"
        )
        
        if "cache_hash" not in st.session_state or st.session_state["cache_hash"] != cache_hash:
            with st.spinner("Executing adaptive target-aware prep layers..."):
                try:
                    pipeline = AutoDataPrepPipeline(
                        target_column=active_target,
                        isolate_target=isolate_target if active_target else False,
                        skewness_threshold=skewness_threshold,
                        correlation_threshold=correlation_threshold,
                        cardinality_threshold=cardinality_threshold,
                        noise_threshold=noise_threshold
                    )
                    cleaned, model_ready = pipeline.fit_transform_dual(raw_df)
                    
                    st.session_state["cleaned_df"] = cleaned
                    st.session_state["model_ready_df"] = model_ready
                    st.session_state["cache_hash"] = cache_hash
                except Exception as e:
                    st.error(f"Pipeline execution encountered an error: {e}")
                    return

        cleaned_df = st.session_state["cleaned_df"]
        model_ready_df = st.session_state["model_ready_df"]

        st.markdown("---")
        
        # Display Workspace: Alternative Branch View A
        if "Human-Readable" in output_target:
            st.markdown("### 📋 Cleaned Analyst Dataset")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("Missing values imputed and fields handled. Structural identifiers and text strings left unencoded for crisp BI visualization.")
            with col2:
                st.download_button(
                    label="📥 Download Cleaned CSV", data=cleaned_df.to_csv(index=False).encode('utf-8'), file_name="cleaned_dataset.csv", mime="text/csv", use_container_width=True
                )
            st.dataframe(cleaned_df, use_container_width=True)

        # Display Workspace: Alternative Branch View B
        elif "Machine Learning" in output_target:
            st.markdown("### 🧮 Model-Ready Mathematical Matrix")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("Fully preprocessed numerical matrix. Pairwise collinear features pruned based on target correlation tier weight distributions.")
            with col2:
                st.download_button(
                    label="📥 Download ML-Ready CSV", data=model_ready_df.to_csv(index=False).encode('utf-8'), file_name="model_ready_matrix.csv", mime="text/csv", use_container_width=True
                )
            st.dataframe(model_ready_df, use_container_width=True)
            
            st.markdown("### 📊 Pearson Correlation Heatmap")
            numeric_cols = model_ready_df.select_dtypes(include=[np.number])
            if len(numeric_cols.columns) > 1:
                corr_styled = numeric_cols.corr(method="pearson").style.background_gradient(cmap="coolwarm", axis=None)
                st.dataframe(corr_styled, use_container_width=True)
            else:
                st.warning("Insufficient numeric columns to render correlation map.")

        # Display Workspace: Alternative Branch View C
        elif "Dual Export" in output_target:
            st.markdown("### ♊ Dual Output Workspace")
            pane1, pane2 = st.columns(2)
            
            with pane1:
                st.markdown("#### 📋 Cleaned Analyst Dataset")
                st.download_button(
                    label="📥 Download Cleaned CSV", data=cleaned_df.to_csv(index=False).encode('utf-8'), file_name="cleaned_dataset.csv", mime="text/csv", use_container_width=True, key="clean_dual_download"
                )
                st.dataframe(cleaned_df, use_container_width=True)
                
            with pane2:
                st.markdown("#### 🧮 Model-Ready Mathematical Matrix")
                st.download_button(
                    label="📥 Download ML-Ready CSV", data=model_ready_df.to_csv(index=False).encode('utf-8'), file_name="model_ready_matrix.csv", mime="text/csv", use_container_width=True, key="model_dual_download"
                )
                st.dataframe(model_ready_df, use_container_width=True)

    else:
        st.info("💡 Application dormant. Please upload a structured text data report to activate processing layers.")

if __name__ == "__main__":
    main()