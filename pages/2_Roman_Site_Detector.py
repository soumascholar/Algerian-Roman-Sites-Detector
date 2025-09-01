# pages/2_🔎_Roman_Site_Detector.py
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import cv2
from folium.plugins import Fullscreen

# --- Configuration & Constants ---
SENTINEL_HUB_API_KEY = "YOUR_SENTINEL_HUB_API_KEY"  # Replace with your actual key

# Pre-defined regions of interest with their coordinates (center point)
REGIONS = {
    "Timgad (Thamugadi)": {"lat": 35.484167, "lon": 6.468611, "zoom": 14},
    "Tipasa": {"lat": 36.591944, "lon": 2.449444, "zoom": 13},
    "Djémila (Cuicul)": {"lat": 36.316667, "lon": 5.733333, "zoom": 14},
    "Tiddis": {"lat": 36.463333, "lon": 6.483889, "zoom": 14},
    "Cherchell (Caesarea)": {"lat": 36.607500, "lon": 2.190000, "zoom": 13},
    "Tobna": {"lat": 35.348460, "lon": 5.345840, "zoom": 13},
    "Cirta (Constantine)": {"lat": 36.367500, "lon": 6.611944, "zoom": 13},
    "Hippo Regius (Annaba)": {"lat": 36.882500, "lon": 7.750000, "zoom": 13},
    "Lambaesis (Lambèse)": {"lat": 35.488889, "lon": 6.255833, "zoom": 13},
}

# --- Helper Functions ---

@st.cache_data
def load_pleiades_data(filepath='pleiades_sites.csv'):
    """
    Loads verified Roman sites from a local CSV file (exported from Pleiades).
    Uses Streamlit's cache to avoid reloading data on every interaction.
    """
    try:
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Tipasa Amphitheater', 'Timgad Arch of Trajan', 'Djémila Forum',
                     'Cherchell Aqueduct', 'Lambaesis Praetorium'],
            'latitude': [36.594, 35.484, 36.319, 36.611, 35.489],
            'longitude': [2.444, 6.468, 5.736, 2.190, 6.257],
            'description': [
                'Large Roman amphitheater ruins.',
                'Triumphal arch dedicated to Emperor Trajan.',
                'Well-preserved Roman forum and market.',
                'Remains of a major Roman aqueduct.',
                'Headquarters of the Legio III Augusta.'
            ]
        }
        df = pd.DataFrame(data)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        return gdf
    except FileNotFoundError:
        st.error(f"Pleiades dataset not found at '{filepath}'. Please ensure the file is in the root directory.")
        return gpd.GeoDataFrame()

def fetch_sentinel_image(lat, lon, zoom):
    """
    MOCK FUNCTION: Simulates fetching a Sentinel-2 satellite image.
    """
    st.info("Simulating Sentinel-2 image download for the selected region...")
    image = np.zeros((512, 512, 4), dtype=np.uint8)
    cv2.rectangle(image, (100, 150), (200, 250), (180, 160, 140, 200), -1)
    cv2.line(image, (50, 400), (450, 350), (150, 150, 120, 180), 3)
    cv2.circle(image, (350, 100), 30, (160, 160, 130, 190), -1)
    return image

def mock_yolo_detection(image):
    """
    MOCK FUNCTION: Simulates running a YOLOv8 model.
    """
    st.info("Simulating AI detection (YOLOv8)...")
    detections = [
        {"box": [100, 150, 200, 250], "confidence": 0.92, "class": "Rectangle",
         "explanation": "Strong rectangular anomaly detected."},
        {"box": [45, 345, 455, 405], "confidence": 0.78, "class": "Line",
         "explanation": "Clear linear feature likely a road or aqueduct."},
        {"box": [320, 70, 380, 130], "confidence": 0.85, "class": "Circle",
         "explanation": "Circular pattern detected."},
        {"box": [400, 400, 450, 450], "confidence": 0.65, "class": "Rectangle",
         "explanation": "Low confidence detection."}
    ]
    return [d for d in detections if d['confidence'] >= 0.70]

def create_map(center_lat, center_lon, zoom):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None)
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
    folium.TileLayer('CartoDB positron', name='Simple Map').add_to(m)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     attr='Esri', name='Esri Satellite').add_to(m)
    Fullscreen(position="topright", title="Expand map", title_cancel="Exit fullscreen").add_to(m)
    return m

def add_sites_to_map(m, sites_gdf, layer_name, icon, color):
    fg = folium.FeatureGroup(name=layer_name, show=True)
    for _, row in sites_gdf.iterrows():
        popup_html = f"<strong>{row['name']}</strong><hr>{row['description']}"
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=popup_html,
            tooltip=row['name'],
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(fg)
    fg.add_to(m)

def add_detections_to_map(m, detections, image_shape, region_coords, image_snippet):
    fg = folium.FeatureGroup(name="AI-Detected Potential Sites", show=True)
    lat_span, lon_span = 0.05, 0.05
    for det in detections:
        box = det['box']
        center_x = (box[0]+box[2])/2
        center_y = (box[1]+box[3])/2
        det_lat = region_coords['lat'] + (0.5 - center_y / image_shape[0]) * lat_span
        det_lon = region_coords['lon'] + (center_x / image_shape[1] - 0.5) * lon_span

        popup_html = f"""
        <div style="width: 300px;">
            <h4>AI Detection: {det['class']}</h4>
            <b>Confidence:</b> {det['confidence']:.2%}<hr>
            <b>Explanation:</b> {det['explanation']}
        </div>
        """
        folium.CircleMarker(
            location=[det_lat, det_lon],
            radius=8, color='red', fill=True, fill_color='red', fill_opacity=0.7,
            popup=popup_html,
            tooltip=f"{det['class']} ({det['confidence']:.0%})"
        ).add_to(fg)
    fg.add_to(m)

# --- Streamlit App UI ---
st.set_page_config(page_title="Roman Site Detector", page_icon="🔎", layout="wide")

# 🖼️ Show your Canva header image
st.image("header.png", use_column_width=True)

# Subtitle below the header
st.markdown("### Select a region to analyze for potential Roman archaeological sites.")

with st.sidebar:
    st.header("Controls")
    selected_region_name = st.selectbox("Choose a Region of Interest:", list(REGIONS.keys()), key="region_select")
    region_coords = REGIONS[selected_region_name]
    analyze_button = st.button("🛰️ Analyze Region with AI", type="primary", use_container_width=True, key="analyze_btn")
    st.info("Use the layer control on the map to toggle views.")

if 'detections' not in st.session_state: st.session_state.detections = []
if 'image' not in st.session_state: st.session_state.image = None
if 'map_key' not in st.session_state: st.session_state.map_key = 0

if analyze_button:
    st.session_state.map_key += 1
    with st.spinner("Running analysis..."):
        image = fetch_sentinel_image(region_coords['lat'], region_coords['lon'], region_coords['zoom'])
        st.session_state.image = image
        detections = mock_yolo_detection(image)
        st.session_state.detections = detections
    if not detections:
        st.warning("No potential sites found with confidence >= 70%.")
    else:
        st.success(f"Found {len(detections)} potential sites.")

m = create_map(region_coords['lat'], region_coords['lon'], region_coords['zoom'])
pleiades_sites = load_pleiades_data()
if not pleiades_sites.empty:
    add_sites_to_map(m, pleiades_sites, "Verified Roman Sites", "landmark", "blue")
if st.session_state.detections and st.session_state.image is not None:
    add_detections_to_map(m, st.session_state.detections, st.session_state.image.shape, region_coords, st.session_state.image)
folium.LayerControl().add_to(m)
st_folium(m, key=f"map_{st.session_state.map_key}", width='100%', height=600)

# --- Explanation cards & download ---
if st.session_state.detections:
    st.markdown("---")
    st.subheader("AI Detection Details")
    for i, det in enumerate(st.session_state.detections):
        with st.expander(f"📍 Detection #{i+1}: {det['class']} (Confidence: {det['confidence']:.0%})", expanded=False):
            st.write(det['explanation'])
    
    st.markdown("---")
    st.subheader("Download Detections")
    dl_data = []
    lat_span, lon_span = 0.05, 0.05
    for det in st.session_state.detections:
        box = det['box']
        center_x = (box[0]+box[2])/2
        center_y = (box[1]+box[3])/2
        det_lat = region_coords['lat'] + (0.5 - center_y / st.session_state.image.shape[0]) * lat_span
        det_lon = region_coords['lon'] + (center_x / st.session_state.image.shape[1] - 0.5) * lon_span
        dl_data.append({
            'class': det['class'],
            'confidence': det['confidence'],
            'explanation': det['explanation'],
            'latitude': det_lat,
            'longitude': det_lon
        })

    dl_gdf = gpd.GeoDataFrame(
        pd.DataFrame(dl_data),
        geometry=gpd.points_from_xy(pd.DataFrame(dl_data).longitude, pd.DataFrame(dl_data).latitude),
        crs="EPSG:4326"
    )

    st.download_button(
        label="📥 Download as GeoJSON",
        data=dl_gdf.to_json(),
        file_name=f"{selected_region_name}_detections.geojson",
        mime="application/json",
        use_container_width=True,
        key=f"download_geojson_{selected_region_name}_{st.session_state.map_key}"
    )

else:
    st.info("Click 'Analyze Region with AI' to search for potential sites.")

