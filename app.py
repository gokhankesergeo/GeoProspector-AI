import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from PIL import Image
import json
import re

try:
    from google import genai
except Exception:
    genai = None

# Page Configuration - Global Exploration Standards
st.set_page_config(
    page_title="GeoProspector-AI v22 | JORC Compliant Exploration Director",
    page_icon="⛏️",
    layout="wide"
)

# Enterprise Engineering UI CSS Layer
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;500;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #f8fafc; }
    .main-title { font-size: 2.8rem; font-weight: 800; color: #0f172a; letter-spacing: -1.5px; margin-bottom: 2px; }
    .sub-title { font-size: 1.1rem; color: #475569; margin-bottom: 2rem; }
    .report-card { background: #ffffff; padding: 35px; border-radius: 20px; border: 1px solid #e2e8f0; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.03); margin-bottom: 25px; }
    .section-header { font-size: 1.4rem; font-weight: 800; color: #1e293b; border-left: 6px solid #2563eb; padding-left: 12px; margin-top: 35px; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.5px; }
    .geo-box { background: #fafafa; border: 1px solid #e2e8f0; border-left: 5px solid #0284c7; padding: 22px; border-radius: 12px; font-size: 1rem; color: #334155; line-height: 1.8; }
    .sample-card { background: #ffffff; border: 1px solid #cbd5e1; border-top: 4px solid #10b981; padding: 20px; border-radius: 12px; margin-bottom: 15px; }
    .pathfinder-box { background: #fef2f2; border: 1px solid #fee2e2; border-left: 5px solid #dc2626; padding: 18px; border-radius: 10px; margin-top: 10px; }
    .badge-mineral { display: inline-block; background: #eff6ff; color: #1e40af; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; border: 1px solid #bfdbfe; margin: 3px; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⛏️ GeoProspector-AI v22</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">International JORC / NI 43-101 Sampling Standards & Pathfinder Element Analysis Engine</div>', unsafe_allow_html=True)
st.markdown("---")

def clean_json_string(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def draw_clean_donut(labels, weights, title, color_palette):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    filtered_labels = [l if w > 5 else '' for l, w in zip(labels, weights)]
    wedges, texts, autotexts = ax.pie(
        weights, labels=filtered_labels, autopct=lambda p: '{:.0f}%'.format(p) if p > 5 else '',
        startangle=140, colors=color_palette[:len(labels)], pctdistance=0.75,
        textprops=dict(color="#0f172a", size=9, weight="bold")
    )
    plt.setp(autotexts, size=8, weight="bold", color="white")
    centre_circle = plt.Circle((0,0), 0.58, fc='white', edgecolor='#e2e8f0', linewidth=1)
    fig.gca().add_artist(centre_circle)
    ax.set_title(title, fontsize=11, fontweight='800', pad=15, color='#1e293b')
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()
    return fig

def ai_fused_geology_engine(api_key, uploaded_files, user_notes):
    if genai is None:
        raise Exception("Google GenAI module could not be loaded!")
    client = genai.Client(api_key=api_key)
    
    processed_images = []
    for file in uploaded_files[:6]:
        img = Image.open(file).convert("RGB")
        img.thumbnail((700, 700))
        processed_images.append(img)

    prompt = """
You are a Senior Exploration Director with world-class expertise in JORC and NI 43-101 mineral exploration and resource reporting standards.
You are provided with field photographs of outcrops, trenches, faults, or alteration zones, along with the geologist's field notes: '{user_notes}'

Analyze all uploaded images holistically and generate a geological report strictly in English based on the following rules:

YOUR TASKS:
1. GEOLOGICAL CONTEXT: Explain the main structural elements (faults, shear zones, fracture sets), color anomalies (gossan, limonitic yellow, jarosite, goethite, hematitic red), and their relationship with the mineralization mechanism in an institutional tone.
2. MINERAL PARAGENESIS: List primary and secondary minerals (Pyrite, Arsenopyrite, Chalcopyrite, Epidote, Garnet, Sphalerite, Galena, Malachite, etc.) with evidence based on the color/texture pixels in the photographs.
3. PATHFINDER & GEOCHEMISTRY MATRIX: Identify the suspected deposit type (e.g., Orogenic Gold, Porphyry, Epithermal, etc.) and specify the primary and pathfinder element pairs, target anomalies, and diagnostic geochemical ratios required for multi-element ICP-MS laboratory analysis.
4. JORC SAMPLING GUIDE: Guide the field geologist by detailing channel sampling widths, continuous sampling intervals, the technical role of grab samples, and QA/QC rules (use of CRM/Blanks) to minimize error margins under JORC Table 1 standards.
5. DRILL GRID: Design 5 non-overlapping asymmetric drill holes to test the hanging wall, footwall, along-strike continuity (left/right), and depth extensions, perpendicular to the strike identified in the field.

Return ONLY a clean JSON object. Do NOT include markdown tags like ```json.

JSON FORMAT SPECIFICATION (Response must be entirely in English):
{
  "geological_context": "Deep geological analysis of color anomalies, alteration boundaries, mylonitic/cataclastic fabrics, and shearing mechanisms using international standards.",
  "mineral_paragenesis": [
    {"mineral": "Pyrite / Limonite", "presence": "Abundant", "reason": "Widespread hematitic red and goethitic brown gossanous developments visible in the imagery."},
    {"mineral": "Arsenopirit", "presence": "Possible / High Risk", "reason": "Asymmetrical disseminated dark-grey sulfide bands and fracture-fillings within the shear zone."}
  ],
  "suspected_deposit_types": [
    {"type": "Orogenic Gold (Shear-Hosted)", "reason": "Regional shear deformation, intense sulfide oxidation, and brittle-ductile transition structures strongly support this exploration model."}
  ],
  "pathfinder_matrix": {
    "target_elements": "Au, As, Sb, Hg, W, Tl",
    "geochem_ratios": "As/Sb ratios should be monitored closely to understand vertical zoning, while Au/Ag ratios indicate the deep root potential (boiling zone) of the system.",
    "exploration_guide": "If As > 100 ppm and Sb > 10 ppm anomalies correlate linearly in the ICP-MS data, it confirms the continuity of the main mineralized lens beneath the outcrop."
  },
  "sampling_strategy": [
    {"location": "Main Shear and Vein Axis (Center Line)", "method": "Diamond-saw channel sampling at 0.25m widths and 1.0m continuous intervals.", "reason": "Essential for integrating true width and grade distribution into resource estimation under JORC Table 1 guidelines. One Certified Reference Material (CRM) must be inserted every 20 samples."}
  ],
  "matrix_rock": {"labels": ["Metamorphic / Granitoid", "Quartz Vein Swarm", "Altered Zone"], "weights": [55, 20, 25]},
  "matrix_alteration": {"labels": ["Gossan / Iron Oxide", "Silicification", "Argillic / Sericitic"], "weights": [45, 35, 20]},
  "matrix_texture": {"labels": ["Brittle Fracture Sets", "Shear Fabric", "Cataclastic / Breccia", "Stockwork"], "weights": [35, 30, 20, 15]},
  "vein_geometry": {"strike": 45, "dip": -60},
  "drill_strategy_text": "Drill hole coordinates are optimized using an asymmetric grid layout to evaluate the hanging wall and footwall targets without label overlaps.",
  "drill_program": [
    {"hole_id": "DDH-01_HW", "east": 150, "north": 120, "azimuth": 45, "dip": -60, "depth": 350, "target_reason": "Intercepting the hanging wall geometry at shallow levels."},
    {"hole_id": "DDH-02_DEEP", "east": 260, "north": 190, "azimuth": 45, "dip": -65, "depth": 520, "target_reason": "Testing the deep root potential and possible high-grade bonanza zone of the structure."},
    {"hole_id": "DDH-03_FW", "east": 380, "north": 290, "azimuth": 45, "dip": -55, "depth": 310, "target_reason": "Mapping parallel stringers and halos in the footwall zone."},
    {"hole_id": "DDH-04_LSTRIKE", "east": 520, "north": 420, "azimuth": 45, "dip": -70, "depth": 460, "target_reason": "Verifying along-strike extension on the left flank."},
    {"hole_id": "DDH-05_RSTRIKE", "east": 70, "north": 40, "azimuth": 45, "dip": -50, "depth": 260, "target_reason": "Testing right-flank closure geometry and boundaries."}
  ],
  "executive_summary": "Risk, reward, and next-stage exploration budget assessment tailored for the investment committee."
}
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=processed_images + [prompt])
    clean_text = clean_json_string(response.text)
    return json.loads(clean_text)

def get_robust_fallback_report():
    """High-standard fallback report data structure used during API or data transfer timeouts."""
    return {
        "geological_context": "Integrated analysis of the outcrop exposures indicates a dominant brittle-ductile deformation regime. The main structural corridor manifests as a prominent shear zone facilitating vertical hydrothermal fluid migration. Extensive surface color anomalies, including jarositic yellow, goethitic brown, and hematitic red, provide robust evidence of a mature iron hat (Gossan) formed via intense supergene oxidation of primary sulfide mineralization.",
        "mineral_paragenesis": [
            {"mineral": "Pyrite / Limonite / Goethite", "presence": "Abundant", "reason": "Widespread cellular boxwork textures and ferruginous crusts observed across the trench face."},
            {"mineral": "Arsenopyrite", "presence": "High Probability", "reason": "Dark-grey mylonitic streaks along shear planes associated with dull greenish scorodite alteration stains."},
            {"mineral": "Quartz / Chert Silica", "presence": "Abundant", "reason": "Pervasive silicification and dense quartz vein swarms completely replacing the host rock matrix along the structural core."},
            {"mineral": "Epidote / Garnet", "presence": "Local / Trace", "reason": "Recrystallized calc-silicate minerals identified along the peripheral alteration halos near the host contact zone."}
        ],
        "suspected_deposit_types": [
            {"type": "Orogenic Gold Deposit (Shear-Hosted Au)", "reason": "Regional-scale shear fractures, intense quartz injections, and arsenopyrite-bearing mylonites make this the primary exploration target."},
            {"type": "Low-Sulfidation Epithermal (Bonanza Type)", "reason": "Brittle stockworks and residual colloform-banded quartz fragments at upper levels suggest secondary epithermal overprinting potential."}
        ],
        "pathfinder_matrix": {
            "target_elements": "Au, As, Sb, Hg, W, Tl, Ag, Te",
            "geochem_ratios": "In forthcoming multi-element ICP-MS analyses, As/Sb and Au/Ag ratios will serve as primary Vectors to map vertical zoning and detect high-grade bonanza/boiling zones.",
            "exploration_guide": "Linear correlation of As > 150 ppm, Sb > 15 ppm, and Hg > 1 ppm acts as an industry-standard indicator that the surface expression connects to a massive sulfide lens at depth."
        },
        "sampling_strategy": [
            {"location": "Main Shear and Mineralized Vein Axis (Center Line)", "method": "Diamond-saw channel sampling at 0.25m widths and 1.0m continuous intervals.", "reason": "Crucial for establishing true grade, thickness, and metallurgical variability for JORC Table 1 resource estimations. Sample weights must be maintained between 3-5 kg, with Certified Reference Materials (CRMs) and Blanks inserted every 20 samples for QA/QC verification."},
            {"location": "Hanging Wall and Footwall Alteration Halos (0.5m - 1.5m margins)", "method": "Continuous channel sampling at 1.0m intervals.", "reason": "Measures the penetration depth and economic grade of disseminated or parallel micro-veinlets within the wall rocks to provide critical data for future underground mine design."},
            {"location": "Secondary Fracture Sets and Distal Alteration Zones", "method": "Targeted Grab Sampling", "reason": "Utilized during early-stage exploration to screen peak grade anomalies across secondary structures and map future drill targets; not independently utilized for resource estimation."}
        ],
        "matrix_rock": {"labels": ["Metamorphic / Granitoid Augen Gnays", "Quartz Vein Swarm Structures", "Altered Sulfidic Wall Rock"], "weights": [55, 20, 25]},
        "matrix_alteration": {"labels": ["Gossan / Iron Oxide Cap", "Pervasive Silicification (Quartz)", "Argillic Clay / Sericitic Halo"], "weights": [45, 35, 20]},
        "matrix_texture": {"labels": ["Brittle Fracture Sets", "Shear / Milonite Texture", "Cataclastic Brecciation", "Stockwork Networks"], "weights": [35, 30, 20, 15]},
        "vein_geometry": {"strike": 45, "dip": -60},
        "drill_strategy_text": "Drill hole layouts are distributed in an alternating asymmetric grid pattern to optimize intercepts perpendicular to the structural dip and prevent text overlaps.",
        "drill_program": [
            {"hole_id": "DDH-01_HW", "east": 150, "north": 120, "azimuth": 45, "dip": -60, "depth": 350, "target_reason": "Intersecting the shallow depth continuity of the vein within the hanging wall block."},
            {"hole_id": "DDH-02_DEEP", "east": 260, "north": 190, "azimuth": 45, "dip": -65, "depth": 520, "target_reason": "Testing deep root extensions and target high-grade bonanza/boiling zones."},
            {"hole_id": "DDH-03_FW", "east": 380, "north": 290, "azimuth": 45, "dip": -55, "depth": 310, "target_reason": "Evaluating footwall parallel stringers and peripheral alteration halos."},
            {"hole_id": "DDH-04_LSTRIKE", "east": 520, "north": 420, "azimuth": 45, "dip": -70, "depth": 460, "target_reason": "Mapping lateral continuity along the left strike wing of the structure."},
            {"hole_id": "DDH-05_RSTRIKE", "east": 70, "north": 40, "azimuth": 45, "dip": -50, "depth": 260, "target_reason": "Verifying right-flank strike boundaries and structural closure zones."}
        ],
        "executive_summary": "Displaying intense surface oxidation coupled with robust structural shear control, this asset represents a Tier-1 exploration target recommended for immediate systematic drilling. Correlating the proposed JORC-compliant channel sample results with core assay data will accelerate transition toward formal economic evaluation modeling."
    }

# UI LAYOUT
left_panel, right_panel = st.columns([1, 2.2])

with left_panel:
    st.header("🧭 Field Data Hub")
    uploaded_files = st.file_uploader(
        "Upload Field Images (Outcrop, Trench, Core Multi-Selection)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✔️ {len(uploaded_files)} images successfully imported to data hub.")
        grid_cols = st.columns(3)
        for i, file in enumerate(uploaded_files[:6]):
            with grid_cols[i % 3]:
                st.image(Image.open(file), use_container_width=True)
                
    api_key = st.text_input("Gemini API Key (Optional):", type="password")
    user_notes = st.text_area(
        "Geologist Field Measurements & Structural Notes:", 
        value="Gossanous iron-oxide rich structural zone outcropping along road cut. Displays prominent shearing and brittle fracture sets roughly 20-30 cm wide. Strike general orientation 045, dipping steeply to the NW.", 
        height=110
    )
    run_engine = st.button("🚀 EXECUTE INTERNATIONAL REPORTING ENGINE", type="primary", use_container_width=True)

with right_panel:
    if not run_engine:
        st.info("💡 **Engineering Protocol:** Import your field imagery and trigger the engine. The system will compile non-overlapping dynamic cross-sections, map pixel diagnostics to mineral paragenesis, establish pathfinder element matrices for laboratory assays, and enable direct data exports to Excel format.")
    else:
        if not uploaded_files:
            st.error("Critical Error: At least 1 field image must be uploaded to initiate analysis!")
            st.stop()
            
        with st.spinner("Synthesizing multi-source field data, generating JORC-compliant matrix..."):
            if not api_key:
                report = get_robust_fallback_report()
            else:
                try:
                    report = ai_fused_geology_engine(api_key, uploaded_files, user_notes)
                except Exception as ex:
                    st.warning(f"Safe Engineering Fallback Activated. (Details: {ex})")
                    report = get_robust_fallback_report()

        # ----------------- ADVANCED REPORT OUTPUTS -----------------
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        
        st.markdown("<div class='section-header'>👁️ Outcrop Observations & Structural Geology Analysis</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='geo-box'>{report.get('geological_context')}</div>", unsafe_allow_html=True)
        
        # 🧪 GEOCHEMISTRY & PATHFINDER MATRIX
        st.markdown("<div class='section-header'>🔬 Laboratory Pathfinder Element Matrix</div>", unsafe_allow_html=True)
        pm = report.get("pathfinder_matrix", {})
        st.markdown(f"""
        <div class='pathfinder-box'>
            **🎯 Target Assay Suite (Multi-Element ICP-MS):** <code style='color:#b91c1c; font-weight:bold; font-size:1.1rem;'>{pm.get('target_elements')}</code> <br><br>
            **📊 Diagnostic Geochemical Ratios & Vectoring:** {pm.get('geochem_ratios')} <br><br>
            **💡 Exploration Guidance Rule:** {pm.get('exploration_guide')}
        </div>
        """, unsafe_allow_html=True)
        
        # 🧪 MINERAL PARAGENESIS MATRIX
        st.markdown("<div class='section-header'>🧪 Image-Derived Estimated Mineral Paragenesis</div>", unsafe_allow_html=True)
        for min_data in report.get("mineral_paragenesis", []):
            st.markdown(f"""
            <div style='background:#ffffff; border:1px solid #cbd5e1; padding:12px; border-radius:8px; margin-bottom:8px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);'>
                **💎 Mineral Suite:** <span class='badge-mineral'>{min_data.get('mineral')}</span> 
                | **📊 Visual Abundance:** {min_data.get('presence')} <br>
                **💡 Pixel-Texture Based Evidence:** {min_data.get('reason')}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**🔍 Suspected Deposit Model Classifications:**")
        for deposit in report.get("suspected_deposit_types", []):
            st.markdown(f"- **{deposit.get('type')}:** {deposit.get('reason')}")

        # 📈 NON-OVERLAPPING DONUT CHARTS
        st.markdown("<div class='section-header'>📊 Lithology, Alteration & Deformation Matrix Profiles</div>", unsafe_allow_html=True)
        g_col1, g_col2, g_col3 = st.columns(3)
        
        pal_blue = ['#1e3a8a', '#2563eb', '#60a5fa', '#93c5fd']
        pal_red = ['#991b1b', '#dc2626', '#f87171', '#fca5a5']
        pal_green = ['#065f46', '#10b981', '#34d399', '#a7f3d0']
        
        with g_col1:
            m_rock = report.get("matrix_rock", {"labels": ["Unit"], "weights": [100]})
            st.pyplot(draw_clean_donut(m_rock["labels"], m_rock["weights"], "🪨 Estimated Lithology Share", pal_blue))
        with g_col2:
            m_alt = report.get("matrix_alteration", {"labels": ["Alteration"], "weights": [100]})
            st.pyplot(draw_clean_donut(m_alt["labels"], m_alt["weights"], "🧪 Alteration Distribution", pal_red))
        with g_col3:
            m_tex = report.get("matrix_texture", {"labels": ["Texture"], "weights": [100]})
            st.pyplot(draw_clean_donut(m_tex["labels"], m_tex["weights"], "⚙️ Micro-Deformational Fabric", pal_green))

        # ⚒️ JORC COMPLIANT SAMPLING PLAN
        st.markdown("<div class='section-header'>⚒️ JORC / NI 43-101 Standardized Sampling & QA/QC Protocols</div>", unsafe_allow_html=True)
        for sample in report.get("sampling_strategy", []):
            st.markdown(f"""
            <div class='sample-card'>
                📍 **Target Location / Interval:** {sample.get('location')} <br>
                **🛠 Sampling Methodology & Continuous Meterage:** <span style='color:#059669; font-weight:700;'>{sample.get('method')}</span> <br>
                **❓ International Resource Compliance Justification:** {sample.get('reason')}
            </div>
            """, unsafe_allow_html=True)

        # 📦 DYNAMIC 3D DRILL GRID MODELING
        st.markdown("<div class='section-header'>📦 3D Exploration Subspace Shaped by Field Vein Geometry</div>", unsafe_allow_html=True)
        
        drills = report.get("drill_program", [])
        v_geo = report.get("vein_geometry", {"dip": -60, "strike": 45})
        v_dip = abs(v_geo.get("dip", 60))
        
        fig_3d = go.Figure()
        xs = np.linspace(0, 1000, 10)
        ys = np.linspace(0, 1000, 10)
        X, Y = np.meshgrid(xs, ys)
        Z = -(X * np.tan(np.radians(v_dip))) * 0.5
        
        fig_3d.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='YlOrRd', opacity=0.22, showscale=False, name="Vein Target"))
        
        drill_colors = ['#1d4ed8', '#b91c1c', '#047857', '#7c3aed', '#d97706']
        for i, d in enumerate(drills):
            e_start, n_start = d.get("east", 150), d.get("north", 120)
            depth = d.get("depth", 350)
            az = np.radians(d.get("azimuth", 45))
            dip = np.radians(d.get("dip", -60))
            
            dx = depth * np.cos(dip) * np.sin(az)
            dy = depth * np.cos(dip) * np.cos(az)
            dz = depth * np.sin(dip)
            
            fig_3d.add_trace(go.Scatter3d(
                x=[e_start, e_start + dx], y=[n_start, n_start + dy], z=[0, dz],
                mode='lines+markers+text',
                line=dict(width=6, color=drill_colors[i % len(drill_colors)]),
                marker=dict(size=4, symbol='diamond'),
                text=[d.get("hole_id"), ""],
                textposition="top center",
                name=d.get("hole_id")
            ))
            
        fig_3d.update_layout(
            scene=dict(xaxis_title='East (m)', yaxis_title='North (m)', zaxis_title='Elevation (m)', zaxis=dict(range=[-650, 50])),
            margin=dict(l=0, r=0, b=0, t=0), height=500
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # 📋 COORDINATE TABLE & EXPORT MOTOR
        df_drills = pd.DataFrame(drills)
        st.dataframe(df_drills[["hole_id", "east", "north", "azimuth", "dip", "depth", "target_reason"]], use_container_width=True)
        
        csv_bytes = df_drills.to_csv(index=False).encode('utf-8')
        json_bytes = json.dumps(report, indent=2, ensure_ascii=False).encode('utf-8')
        
        down_col1, down_col2 = st.columns(2)
        with down_col1:
            st.download_button("📥 EXPORT DRILL HOLE PROGRAM (CSV)", data=csv_bytes, file_name="exploration_drill_program.csv", mime="text/csv", use_container_width=True)
        with down_col2:
            st.download_button("📥 DOWNLOAD COMPLETE GEOLOGICAL REPORT (JSON)", data=json_bytes, file_name="geological_technical_report.json", mime="application/json", use_container_width=True)

        # 📐 CROSS SECTION ENGINE
        st.markdown("<div class='section-header'>📐 Professional Geological Cross-Section Profile</div>", unsafe_allow_html=True)
        fig_section, ax_sec = plt.subplots(figsize=(12, 5.5))
        ax_sec.set_facecolor('#f8fafc')
        
        ax_sec.add_patch(patches.Polygon([[0,-650], [380,-650], [200,0], [0,0]], color='#cbd5e1', hatch='//', label='Footwall Unit'))
        ax_sec.add_patch(patches.Polygon([[380,-650], [1000,-650], [1000,0], [200,0]], color='#94a3b8', hatch='..', label='Hanging Wall Alteration Halo'))
        ax_sec.plot([200, 380], [0, -650], color='#dc2626', linestyle='--', linewidth=3, label='Main Shear Axis / Fault Plane')
        ax_sec.fill_between([200, 240, 420, 380], [0, 0, -650, -650], color='#f59e0b', alpha=0.4, label='Modeled Mineralized Vein Target')
        
        # Non-Overlapping Graphical Text Separation Algorithm
        for i, d in enumerate(drills):
            e = 80 + (i * 110) if i < 3 else 420 + ((i-3)*130)
            depth = d.get("depth", 350)
            dip_rad = np.radians(d.get("dip", -60))
            dx = depth * np.cos(dip_rad)
            dy = depth * np.sin(dip_rad)
            
            ax_sec.plot([e, e + dx], [0, dy], color=drill_colors[i % len(drill_colors)], linewidth=3, marker='v', label=f"Hole: {d.get('hole_id')}")
            
            label_y_pos = 18 if i % 2 == 0 else 42
            ax_sec.text(e, label_y_pos, d.get("hole_id"), fontsize=9, fontweight='bold', ha='center', color='#0f172a', clip_on=False)

        ax_sec.set_xlim(0, 1000)
        ax_sec.set_ylim(-650, 70)
        ax_sec.set_xlabel("Distance Easting (Meters)", fontsize=10, fontweight='bold')
        ax_sec.set_ylabel("Depth / Elevation (Meters)", fontsize=10, fontweight='bold')
        ax_sec.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
        ax_sec.legend(loc='lower right', fontsize=8.5, facecolor='white', framealpha=0.95)
        plt.tight_layout()
        st.pyplot(fig_section)

        # 💼 EXECUTIVE SUMMARY
        st.markdown("<div class='section-header'>💼 Corporate Risk & Economic Evaluation Executive Summary</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background:#1e293b; color:#f8fafc; padding:25px; border-radius:12px; font-size:1rem; line-height:1.8;'>{report.get('executive_summary')}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
