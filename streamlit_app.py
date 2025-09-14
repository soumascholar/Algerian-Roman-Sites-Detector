import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from folium.plugins import Fullscreen

# --- Configuration & Constants ---
SENTINEL_HUB_API_KEY = "YOUR_SENTINEL_HUB_API_KEY"  # Replace with your actual key

# ✅ Pre-defined Roman sites with corrected coordinates
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

# --- Remote Sensing Indicators Dictionary ---
VISIBILITY_INDICATORS = {
    "Rectangle": ["🟤 Soil marks – color/tones may reveal foundations", "🌱 Cropmarks – stunted or lighter crops over walls"],
    "Line": ["☀️ Shadow marks – low-angle sunlight highlights linear features", "🌱 Cropmarks – linear crop anomalies", "💧 Moisture marks – ditches retain water differently"],
    "Circle": ["🏞️ Topography/Micro-relief – circular mounds or amphitheater bases", "💧 Moisture marks – circular depressions may collect water"]
}

# --- Language Switch ---
language = st.sidebar.radio("🌍 Language / اللغة", ["English", "العربية"])

# --- Helper Functions ---
@st.cache_data
def load_pleiades_data(filepath='pleiades_sites.csv'):
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
    st.info("Simulating Sentinel-2 image download for the selected region..." if language == "English" else "محاكاة تحميل صورة Sentinel-2 للمنطقة المحددة...")

    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Simulated anomalies
    draw.rectangle([100, 150, 200, 250], fill=(180, 160, 140, 200))
    draw.line([50, 400, 450, 350], fill=(150, 150, 120, 180), width=3)
    draw.ellipse([320, 70, 380, 130], fill=(160, 160, 130, 190))

    return np.array(image)

def mock_yolo_detection(image):
    st.info("Simulating AI detection (YOLOv8)..." if language == "English" else "محاكاة الكشف بالذكاء الاصطناعي (YOLOv8)...")
    detections = [
        {"box": [100, 150, 200, 250], "confidence": 0.92, "class": "Rectangle",
         "explanation": "Strong rectangular anomaly detected." if language == "English" else "تم اكتشاف شذوذ مستطيل قوي."},
        {"box": [45, 345, 455, 405], "confidence": 0.78, "class": "Line",
         "explanation": "Clear linear feature likely a road or aqueduct." if language == "English" else "معلم خطي واضح على الأرجح طريق أو قناة مائية."},
        {"box": [320, 70, 380, 130], "confidence": 0.85, "class": "Circle",
         "explanation": "Circular pattern detected." if language == "English" else "تم اكتشاف نمط دائري."},
        {"box": [400, 400, 450, 450], "confidence": 0.65, "class": "Rectangle",
         "explanation": "Low confidence detection." if language == "English" else "كشف بثقة منخفضة."}
    ]
    return [d for d in detections if d['confidence'] >= 0.70]

def create_map(center_lat, center_lon, zoom):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None)
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
    folium.TileLayer('CartoDB positron', name='Simple Map').add_to(m)
    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
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

def add_region_markers(m, regions):
    fg = folium.FeatureGroup(name="All Roman Sites (verification)" if language == "English" else "جميع المواقع الرومانية (التحقق)", show=True)
    for name, r in regions.items():
        folium.Marker(
            location=[r['lat'], r['lon']],
            popup=f"<strong>{name}</strong>",
            tooltip=name,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(fg)
    fg.add_to(m)

def add_detections_to_map(m, detections, image_shape, region_coords, image_snippet):
    fg = folium.FeatureGroup(name="AI-Detected Potential Sites" if language == "English" else "المواقع المحتملة المكتشفة بالذكاء الاصطناعي", show=True)
    lat_span, lon_span = 0.05, 0.05
    for det in detections:
        box = det['box']
        center_x = (box[0]+box[2])/2
        center_y = (box[1]+box[3])/2
        det_lat = region_coords['lat'] + (0.5 - center_y / image_shape[0]) * lat_span
        det_lon = region_coords['lon'] + (center_x / image_shape[1] - 0.5) * lon_span

        indicators = VISIBILITY_INDICATORS.get(det['class'], [])
        indicators_html = "<ul>" + "".join(f"<li>{ind}</li>" for ind in indicators) + "</ul>"

        popup_html = f"""
        <div style="width: 300px;">
            <h4>AI Detection: {det['class']}</h4>
            <b>Confidence:</b> {det['confidence']:.2%}<hr>
            <b>Interpretation:</b> {det['explanation']}<br>
            <b>Visibility indicators:</b>{indicators_html}
        </div>
        """
        folium.CircleMarker(
            location=[det_lat, det_lon],
            radius=8, color='red', fill=True, fill_color='red', fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"{det['class']} ({det['confidence']:.0%})"
        ).add_to(fg)
    fg.add_to(m)

# --- Streamlit App UI ---
st.set_page_config(page_title="Roman Site Detector", page_icon="🔎", layout="wide")

st.image("header.png", use_container_width=True)
if language == "English":
    st.markdown("Select a region to analyze for potential Roman archaeological sites.")
else:
    st.markdown("اختر منطقة لتحليلها بحثًا عن المواقع الأثرية الرومانية المحتملة.")

with st.sidebar:
    st.header("Controls" if language == "English" else "التحكم")
    selected_region_name = st.selectbox("Choose a Region of Interest:" if language == "English" else "اختر منطقة:", list(REGIONS.keys()))
    region_coords = REGIONS[selected_region_name]
    analyze_button = st.button("🛰️ Analyze Region with AI" if language == "English" else "🛰️ تحليل المنطقة بالذكاء الاصطناعي", type="primary", use_container_width=True)
    st.info("Use the layer control on the top-right of the map to toggle views." if language == "English" else "استخدم التحكم في الطبقات أعلى اليمين لتبديل طرق العرض.")

if 'detections' not in st.session_state: st.session_state.detections = []
if 'image' not in st.session_state: st.session_state.image = None
if 'map_key' not in st.session_state: st.session_state.map_key = 0

# --- Analysis ---
if analyze_button:
    st.session_state.map_key += 1
    with st.spinner("Running analysis..." if language == "English" else "جاري التحليل..."):
        image = fetch_sentinel_image(region_coords['lat'], region_coords['lon'], region_coords['zoom'])
        st.session_state.image = image
        detections = mock_yolo_detection(image)
        st.session_state.detections = detections
    if not detections:
        st.warning("No potential sites found with confidence >= 70%." if language == "English" else "لم يتم العثور على مواقع محتملة بدرجة ثقة أكبر من 70٪.")
    else:
        st.success(f"Found {len(detections)} potential sites." if language == "English" else f"تم العثور على {len(detections)} مواقع محتملة.")

# --- Map ---
m = create_map(region_coords['lat'], region_coords['lon'], region_coords['zoom'])
add_region_markers(m, REGIONS)
pleiades_sites = load_pleiades_data()
if not pleiades_sites.empty:
    add_sites_to_map(m, pleiades_sites, "Verified Roman Sites" if language == "English" else "المواقع الرومانية الموثقة", "landmark", "blue")
if st.session_state.detections and st.session_state.image is not None:
    add_detections_to_map(m, st.session_state.detections, st.session_state.image.shape, region_coords, st.session_state.image)
folium.LayerControl().add_to(m)
st_folium(m, key=f"map_{st.session_state.map_key}", width='100%', height=600)

# --- AI Detection Report ---
for i, det in enumerate(st.session_state.detections):
    if language == "English":
        with st.expander(f"📍 Detection #{i+1}: {det['class']} (Confidence: {det['confidence']:.0%})"):
            class_to_feature = {
                "Rectangle": "Potential building foundation or wall structure",
                "Line": "Potential ancient road, wall, or aqueduct",
                "Circle": "Possible burial mound, cistern, or amphitheater base"
            }
            interpretation = class_to_feature.get(det['class'], det['explanation'])
            st.write(f"**Interpretation:** {interpretation}")
            indicators = VISIBILITY_INDICATORS.get(det['class'], [])
            if indicators:
                st.write("**Visibility indicators:**")
                for ind in indicators:
                    st.write(f"- {ind}")
            st.write("*Note: Field verification recommended for all AI-detected anomalies.*")
    else:
        with st.expander(f"📍 الكشف رقم {i+1}: {det['class']} (الثقة: {det['confidence']:.0%})"):
            class_to_feature = {
                "Rectangle": "أساس بناء محتمل أو جدار",
                "Line": "طريق أو جدار أو قناة مائية (قنطرة) قديمة محتملة",
                "Circle": "مدفن دائري محتمل أو صهريج أو أساس مدرج"
            }
            interpretation = class_to_feature.get(det['class'], det['explanation'])
            st.write(f"**التفسير:** {interpretation}")
            indicators = VISIBILITY_INDICATORS.get(det['class'], [])
            if indicators:
                st.write("**مؤشرات الرؤية:**")
                for ind in indicators:
                    if "Shadow" in ind:
                        st.write("☀️ ظلال – يبرز ضوء الشمس المنخفض الزاوية المعالم الخطية")
                    elif "Cropmarks" in ind:
                        st.write("🌱 علامات المحاصيل – شذوذات خطية في نمو النباتات")
                    elif "Moisture" in ind:
                        st.write("💧 علامات الرطوبة – تحتفظ الخنادق بالماء بشكل مختلف")
                    elif "Soil" in ind:
                        st.write("🟤 علامات التربة – اختلاف الألوان فوق الهياكل المدفونة")
                    elif "Topography" in ind:
                        st.write("🏞️ الطبوغرافيا / التضاريس الدقيقة – بروزات أو انخفاضات طفيفة")
            st.write("*ملاحظة: يُوصى بالتحقق الميداني لجميع الشذوذات المكتشفة بواسطة الذكاء الاصطناعي.*")

# ✅ Download Section
if st.session_state.detections:
    st.markdown("---")
    st.subheader("Download Detections" if language == "English" else "تحميل الكشوفات")

    dl_data = []
    lat_span, lon_span = 0.05, 0.05
    for det in st.session_state.detections:
        box = det['box']
        center_x = (box[0]+box[2])/2
        center_y = (box[1]+box[3])/2
        det_lat = region_coords['lat'] + (0.5 - center_y / st.session_state.image.shape[0]) * lat_span
        det_lon = region_coords['lon'] + (center_x / st.session_state.image.shape[1] - 0.5) * lon_span
        dl_data.append({'class': det['class'], 'confidence': det['confidence'],
                        'explanation': det['explanation'], 'latitude': det_lat, 'longitude': det_lon})

    dl_gdf = gpd.GeoDataFrame(
        pd.DataFrame(dl_data),
        geometry=gpd.points_from_xy(pd.DataFrame(dl_data).longitude, pd.DataFrame(dl_data).latitude),
        crs="EPSG:4326"
    )

    st.download_button(
        label="📥 Download as GeoJSON" if language == "English" else "📥 تحميل كملف GeoJSON",
        data=dl_gdf.to_json(),
        file_name=f"{selected_region_name}_detections.geojson",
        mime="application/json",
        use_container_width=True,
        key="geojson_download"
    )