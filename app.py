import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium

from src.gis_checks import screen_plot
from src.ai_change_detection import load_ai_bundle, evaluate_temporal_change
from src.hpr_report import generate_hpr_pdf
from src.case_store import get_all_cases, update_case_status, add_new_case

st.set_page_config(
    page_title="NUHGA 2.0 | Urban Hydro-Governance",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Civic-Tech Console CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Top Header Bar */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #38bdf8;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    
    .tag-pill {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 3px 8px;
        border-radius: 4px;
        background: #1e293b;
        border: 1px solid #475569;
        color: #38bdf8;
        margin-right: 6px;
    }
    
    /* Custom Metric Cards */
    .metric-card {
        background: #111c30;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .metric-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    
    .badge-danger {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        padding: 8px 16px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Create Folium map with Esri Satellite & Clean OSM
def build_satellite_map(center=[23.2030, 77.4245], zoom=15):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None)
    
    # 1. Esri High-Resolution World Imagery
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite (Esri World Imagery)",
        overlay=False,
        control=True
    ).add_to(m)
    
    # 2. OpenStreetMap Street View
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap contributors",
        name="Street View (OpenStreetMap)",
        overlay=False,
        control=True
    ).add_to(m)
    
    return m

# Load Layers
@st.cache_data
def load_spatial_layers():
    pilot = gpd.read_file("data/demo/pilot_boundary.geojson")
    water = gpd.read_file("data/demo/water_bodies.geojson")
    drainage = gpd.read_file("data/demo/drainage_corridors.geojson")
    plots = gpd.read_file("data/demo/sample_plots.geojson")
    flags = gpd.read_file("data/demo/sample_ai_flags.geojson")
    return pilot, water, drainage, plots, flags

pilot_gdf, water_gdf, drainage_gdf, plots_gdf, flags_gdf = load_spatial_layers()

# Header Layout
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="hero-title">NUHGA 2.0 <span style="color:#64748b; font-weight:300;">|</span> AI-GIS Urban Hydro-Governance</h1>
            <div class="hero-subtitle">Automated Development Screening & Satellite-Based Water Corridor Protection</div>
        </div>
        <div style="text-align: right;">
            <span class="tag-pill">PILOT: SHAHPURA LAKE, BHOPAL</span>
            <span class="tag-pill">NGT COMPLIANT</span>
            <span class="tag-pill" style="color:#10b981; border-color:#10b981;">STATUS: OPERATIONAL</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "01 // HydroMap Surveillance",
    "02 // HydroScan AI Sentinel Split-View",
    "03 // Automated Plot Pre-Scrutiny & HPR",
    "04 // Municipal Decision Queue",
    "05 // Data Provenance & Model Card"
])

# ==========================================
# TAB 1: HYDROMAP SURVEILLANCE
# ==========================================
with tab1:
    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        st.markdown("### Layer Console")
        show_water = st.checkbox("Protected Water Bodies (EPCO/FTL)", value=True)
        show_buffer = st.checkbox("30m NGT Protection Buffer", value=True)
        show_drainage = st.checkbox("Trunk & Modelled Drainage Channels", value=True)
        show_flags = st.checkbox("Spectral AI Encroachment Flags", value=True)
        show_plots = st.checkbox("Submitted Development Parcels", value=True)

        st.markdown("---")
        st.markdown("### Telemetry Status")
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Catchment Area Monitored</div>
            <div class="metric-value">4.82 km²</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Critical Channels Identified</div>
            <div class="metric-value" style="color: #38bdf8;">3 Active</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Active AI Risk Alerts</div>
            <div class="metric-value" style="color: #ef4444;">2 Detected</div>
        </div>
        """, unsafe_allow_html=True)

    with col_map:
        m = build_satellite_map(center=[23.2030, 77.4245], zoom=15)

        # Pilot Boundary
        folium.GeoJson(
            pilot_gdf,
            name="Catchment Boundary",
            style_function=lambda x: {"fillColor": "none", "color": "#0ea5e9", "weight": 2.5, "dashArray": "6, 6"},
            tooltip=folium.GeoJsonTooltip(fields=["name", "source"])
        ).add_to(m)

        # 30m NGT Buffer Zone
        if show_buffer:
            water_m = water_gdf.to_crs("EPSG:32643")
            water_m["geometry"] = water_m.geometry.buffer(30.0)
            water_buffered = water_m.to_crs("EPSG:4326")
            folium.GeoJson(
                water_buffered,
                name="30m Buffer Zone",
                style_function=lambda x: {"fillColor": "#f59e0b", "color": "#d97706", "weight": 1.5, "fillOpacity": 0.2, "dashArray": "4, 4"}
            ).add_to(m)

        # Water Bodies
        if show_water:
            folium.GeoJson(
                water_gdf,
                name="Water Spread",
                style_function=lambda x: {"fillColor": "#0284c7", "color": "#38bdf8", "weight": 2.5, "fillOpacity": 0.65},
                tooltip=folium.GeoJsonTooltip(fields=["feature_id", "name", "source", "authority_status"])
            ).add_to(m)

        # Drainage Corridors
        if show_drainage:
            for _, row in drainage_gdf.iterrows():
                is_official = "Official" in row["alignment_type"]
                col = "#38bdf8" if is_official else "#f59e0b"
                dash = None if is_official else "6, 6"
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x, c=col, d=dash: {"color": c, "weight": 4, "dashArray": d},
                    tooltip=f"{row['name']} | {row['alignment_type']}"
                ).add_to(m)

        # AI Flags
        if show_flags:
            folium.GeoJson(
                flags_gdf,
                name="AI Anomaly Flags",
                style_function=lambda x: {"fillColor": "#ef4444", "color": "#ff0000", "weight": 3, "fillOpacity": 0.75},
                tooltip=folium.GeoJsonTooltip(fields=["flag_id", "classification", "confidence", "reason"])
            ).add_to(m)

        # Submitted Plots
        if show_plots:
            folium.GeoJson(
                plots_gdf,
                name="Development Plots",
                style_function=lambda x: {"fillColor": "#a855f7", "color": "#d8b4fe", "weight": 2, "fillOpacity": 0.4},
                tooltip=folium.GeoJsonTooltip(fields=["plot_id", "applicant", "description"])
            ).add_to(m)

        folium.LayerControl(position="topright").add_to(m)
        st_folium(m, width=950, height=540, returned_objects=[])

# ==========================================
# TAB 2: HYDROSCAN AI VISUAL SPLIT-VIEW
# ==========================================
with tab2:
    st.subheader("HydroScan AI: Multi-Temporal Spectral Shoreline Change Analysis")
    st.caption("Side-by-side verification comparing baseline Sentinel-2 multi-spectral observations against present shoreline configuration.")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### BASELINE: Nov 2021 (Dry Season Reference)")
        st.markdown("**Sentinel-2 L2A (10m)** — Verified Natural Water Extent ($NDWI \\ge +0.35$)")
        m_t1 = build_satellite_map(center=[23.2045, 77.4275], zoom=16)
        folium.GeoJson(
            water_gdf,
            style_function=lambda x: {"fillColor": "#0284c7", "color": "#38bdf8", "weight": 2, "fillOpacity": 0.75}
        ).add_to(m_t1)
        st_folium(m_t1, width=460, height=360, returned_objects=[], key="map_t1")

    with col_t2:
        st.markdown("#### CURRENT: Feb 2026 (HydroScan Detection)")
        st.markdown("**Sentinel-2 L2A (10m)** — Suspected Land Filling & Drainage Encroachment Detected")
        m_t2 = build_satellite_map(center=[23.2045, 77.4275], zoom=16)
        # Highlight shoreline encroachment in glowing red
        folium.GeoJson(
            flags_gdf,
            style_function=lambda x: {"fillColor": "#ef4444", "color": "#ff0000", "weight": 3, "fillOpacity": 0.85},
            tooltip=folium.GeoJsonTooltip(fields=["flag_id", "classification", "confidence"])
        ).add_to(m_t2)
        st_folium(m_t2, width=460, height=360, returned_objects=[], key="map_t2")

    st.markdown("---")
    st.markdown("#### Interactive Spectral Pixel Classifier (Sentinel-2 Bands B02, B03, B04, B08)")
    
    preset = st.selectbox(
        "Load Multi-Spectral Inspection Coordinate:",
        [
            "Shoreline Coordinate (77.4282 E, 23.2055 N): Water (2021) -> Landfill/Concrete (2026)",
            "Northern Inlet Channel (77.4241 E, 23.2120 N): Riparian Marsh -> Concrete Pavement",
            "Deep Reservoir Central Point (77.4250 E, 23.2020 N): Stable Water Body"
        ]
    )

    if "Shoreline" in preset:
        d1 = {"b02": 0.08, "b03": 0.12, "b04": 0.07, "b08": 0.04}
        d2 = {"b02": 0.19, "b03": 0.21, "b04": 0.25, "b08": 0.22}
    elif "Northern" in preset:
        d1 = {"b02": 0.06, "b03": 0.10, "b04": 0.06, "b08": 0.39}
        d2 = {"b02": 0.18, "b03": 0.20, "b04": 0.24, "b08": 0.21}
    else:
        d1 = {"b02": 0.08, "b03": 0.11, "b04": 0.07, "b08": 0.04}
        d2 = {"b02": 0.08, "b03": 0.11, "b04": 0.07, "b08": 0.05}

    ai_eval = evaluate_temporal_change(d1, d2, overlaps_sensitive_zone=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">2021 Classified State</div>
            <div class="metric-value">{ai_eval['t1_result']['predicted_class']}</div>
            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#38bdf8;">NDWI: {ai_eval['t1_result']['NDWI']}</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">2026 Classified State</div>
            <div class="metric-value">{ai_eval['t2_result']['predicted_class']}</div>
            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#f59e0b;">NDWI: {ai_eval['t2_result']['NDWI']}</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Transition Detected</div>
            <div class="metric-value" style="color:#ef4444; font-size:1.15rem; margin-top:8px;">{ai_eval['transition']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Random Forest Confidence</div>
            <div class="metric-value" style="color:#10b981;">{int(ai_eval['combined_confidence']*100)}%</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 3: AUTOMATED PLOT PRE-SCRUTINY & HPR
# ==========================================
with tab3:
    st.subheader("Development Boundary Screening Engine (ABPAS Integration Sandbox)")
    
    col_scr1, col_scr2 = st.columns([1, 1])

    with col_scr1:
        selected_plot_id = st.selectbox("Select Application Cadastral Parcel:", plots_gdf["plot_id"].tolist())
        selected_plot_gdf = plots_gdf[plots_gdf["plot_id"] == selected_plot_id]
        plot_row = selected_plot_gdf.iloc[0]

        buffer_m = st.slider("National Green Tribunal (NGT) Buffer Distance (m):", 10, 50, 30, 5)
        has_drain_plan = st.checkbox("Applicant Submitted Drainage Continuity Plan", value=bool(plot_row["submitted_drainage_plan"]))

        result = screen_plot(
            plot=selected_plot_gdf,
            water_bodies=water_gdf,
            drainage=drainage_gdf,
            buffer_distance_m=float(buffer_m),
            submitted_drainage_plan=has_drain_plan
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if result["status"] == "REVIEW_REQUIRED":
            st.markdown('<span class="badge-danger">STATUS: REVIEW REQUIRED (VIOLATION DETECTED)</span>', unsafe_allow_html=True)
        elif result["status"] == "NO_MAPPED_CONFLICT_IDENTIFIED":
            st.markdown('<span class="badge-success">STATUS: NO MAPPED CONFLICT IDENTIFIED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warning">STATUS: INSUFFICIENT DATA</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Evidence-Linked Spatial Audit Findings:**")
        for finding in result["findings"]:
            st.markdown(f"- `{finding}`")

        pdf_bytes = generate_hpr_pdf(
            plot_id=plot_row["plot_id"],
            applicant=plot_row["applicant"],
            screening_result=result
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.download_button(
                label="📄 Export Hydrological Report (PDF)",
                data=pdf_bytes,
                file_name=f"HPR_{plot_row['plot_id'][:12]}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col_b2:
            if st.button("🚨 Forward Flag to BMC Review Queue", use_container_width=True):
                new_id = add_new_case(
                    location_ref=plot_row["plot_id"],
                    issue_type=f"Screening Alert: {result['status']}",
                    source_evidence="; ".join(result["findings"])[:120]
                )
                st.success(f"Dispatched to Municipal Queue as Case #{new_id}!")

    with col_scr2:
        centroid = selected_plot_gdf.geometry.iloc[0].centroid
        m_inspect = build_satellite_map(center=[centroid.y, centroid.x], zoom=17)
        
        folium.GeoJson(water_gdf, style_function=lambda x: {"fillColor": "#0284c7", "color": "#38bdf8", "weight": 2, "fillOpacity": 0.6}).add_to(m_inspect)
        folium.GeoJson(drainage_gdf, style_function=lambda x: {"color": "#38bdf8", "weight": 4}).add_to(m_inspect)
        
        # Draw 30m buffer around water
        water_buf = water_gdf.to_crs("EPSG:32643")
        water_buf["geometry"] = water_buf.geometry.buffer(float(buffer_m))
        folium.GeoJson(water_buf.to_crs("EPSG:4326"), style_function=lambda x: {"fillColor": "none", "color": "#f59e0b", "weight": 2, "dashArray": "4, 4"}).add_to(m_inspect)
        
        # Submitted plot
        folium.GeoJson(
            selected_plot_gdf,
            style_function=lambda x: {"fillColor": "#ef4444", "color": "#ffffff", "weight": 3, "fillOpacity": 0.7},
            tooltip=plot_row["plot_id"]
        ).add_to(m_inspect)

        st_folium(m_inspect, width=470, height=380, returned_objects=[], key="map_scr")

# ==========================================
# TAB 4: MUNICIPAL DECISION QUEUE
# ==========================================
with tab4:
    st.subheader("Bhopal Municipal Corporation (BMC) Town Planning Scrutiny Desk")
    st.caption("Human-in-the-loop audit log ensuring every automated AI-GIS flag is formally examined and resolved by an authorised official.")

    cases_df = get_all_cases()
    st.dataframe(cases_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Enter Official Adjudication Finding")
    q1, q2, q3, q4 = st.columns([1, 1, 1, 2])
    with q1:
        target_case = st.selectbox("Application Ref:", cases_df["case_id"].tolist())
    with q2:
        new_status = st.selectbox("Regulatory Finding:", ["Pending", "Verified Violation", "Dismissed (Exempt)", "Site Notice Issued"])
    with q3:
        officer_name = st.text_input("Authorised Engineer:", value="Er. S. Kulkarni (Town Planning)")
    with q4:
        reason_text = st.text_input("Statutory Reason & Field Audit Note:", value="Intersects NGT buffer zone; notice issued under MP Municipal Corp Act.")

    if st.button("Commit Audited Ruling to Permanent Store"):
        update_case_status(target_case, new_status, officer_name, reason_text)
        st.success(f"Audit record saved for {target_case}!")
        st.rerun()

# ==========================================
# TAB 5: DATA PROVENANCE & MODEL CARD
# ==========================================
with tab5:
    st.subheader("Data Layer Provenance & Explainable AI Model Card")
    
    col_d1, col_d2 = st.columns([1, 1])

    with col_d1:
        st.markdown("#### Spatial Layer Quality & Ground Truth Audit")
        audit_records = [
            {"Asset Layer": "Shahpura Lake Core Boundary", "Dataset Origin": "EPCO MP State Wetland Inventory", "Resolution": "Cadastral Scale (>2.25 ha)", "Audit Status": "Verified Core Water Spread"},
            {"Asset Layer": "Stormwater Drainage Alignment", "Dataset Origin": "BMC Drainage Master Plan & CartoDEM", "Resolution": "1:10,000 Scale", "Audit Status": "Modelled Channels Require FTL Ground Survey"},
            {"Asset Layer": "Multi-Spectral Satellite Series", "Dataset Origin": "Copernicus Sentinel-2 (ESA)", "Resolution": "10m Bands (B02, B03, B04, B08)", "Audit Status": "Cloud Masked (<5%), Season-Matched"},
            {"Asset Layer": "Elevation / Terrain Flow", "Dataset Origin": "ISRO Bhuvan CartoDEM", "Resolution": "30m DEM Grid", "Audit Status": "Used for Flow Accumulation Vectoring"}
        ]
        st.table(pd.DataFrame(audit_records))

    with col_d2:
        st.markdown("#### Scikit-Learn Random Forest Classifier Card")
        bundle = load_ai_bundle()
        
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric("Macro Precision", f"{bundle['precision_macro']*100:.1f}%")
        with m_c2:
            st.metric("Macro Recall", f"{bundle['recall_macro']*100:.1f}%")

        st.markdown("**Confusion Matrix (Rows = Actual Class, Cols = Predicted Class):**")
        cm_df = pd.DataFrame(bundle["confusion_matrix"], index=bundle["classes"], columns=bundle["classes"])
        st.dataframe(cm_df, use_container_width=True)
        st.caption("Validated on spatially isolated test quadrants. The model acts purely as a pre-screening filter to trigger human verification.")