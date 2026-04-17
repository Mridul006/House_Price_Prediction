import streamlit as st
import pandas as pd
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(r"C:\Users\mridu\OneDrive\Desktop\coding\projects_apna_college\house_prediction\model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(r"C:\Users\mridu\OneDrive\Desktop\coding\projects_apna_college\house_prediction\model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    return model, model_columns

model, model_columns = load_model()

st.title("🏠 House Price Predictor")
st.caption("Fill in the details below and get an instant estimated sale price using a trained Linear Regression model.")
st.divider()

# ── Section 1: Property Basics ────────────────────────────────────────────────
st.subheader("📋 Property Basics")

col1, col2 = st.columns(2)
with col1:
    ms_subclass = st.selectbox(
        "Building Class Code",
        [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
        help="Type of dwelling (e.g. 20 = 1-story modern, 60 = 2-story modern)"
    )
    year_built = st.number_input(
        "Year Built",
        min_value=1800, max_value=2024, value=2000,
        help="Original year of construction"
    )

with col2:
    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=500, max_value=100000, value=8000,
        help="Total lot size in square feet"
    )
    year_remod = st.number_input(
        "Year Remodelled",
        min_value=1800, max_value=2024, value=2000,
        help="Same as Year Built if no remodelling done"
    )

overall_cond = st.select_slider(
    "Overall Condition",
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    value=5,
    format_func=lambda x: f"{x} — {'Poor' if x<=2 else 'Fair' if x<=4 else 'Average' if x<=6 else 'Good' if x<=8 else 'Excellent'}",
    help="Overall condition rating of the house"
)

st.divider()

# ── Section 2: Basement ───────────────────────────────────────────────────────
st.subheader("🏗️ Basement Details")

col3, col4 = st.columns(2)
with col3:
    total_bsmt_sf = st.number_input(
        "Total Basement Area (sq ft)",
        min_value=0, max_value=10000, value=800,
        help="Total square footage of basement area"
    )
with col4:
    bsmt_fin_sf2 = st.number_input(
        "Basement Finished Area Type 2 (sq ft)",
        min_value=0, max_value=2000, value=0,
        help="Finished basement area of a second type (often 0)"
    )

st.divider()

# ── Section 3: Location & Style ───────────────────────────────────────────────
st.subheader("📍 Location & Style")

col5, col6 = st.columns(2)
with col5:
    ms_zoning = st.selectbox(
        "Zoning Classification",
        ["RL", "RM", "C (all)", "FV", "RH"],
        help="RL = Residential Low Density (most common)"
    )
    bldg_type = st.selectbox(
        "Building Type",
        ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"],
        help="Type of dwelling"
    )

with col6:
    lot_config = st.selectbox(
        "Lot Configuration",
        ["Corner", "Inside", "CulDSac", "FR2", "FR3"],
        help="How the lot is positioned on the street"
    )
    exterior1st = st.selectbox(
        "Exterior Material",
        ["VinylSd", "MetalSd", "Wd Sdng", "HdBoard", "BrkFace",
         "WdShing", "CemntBd", "Plywood", "AsbShng", "Stucco",
         "BrkComm", "AsphShn", "Stone", "ImStucc", "CBlock"],
        help="Primary exterior covering of the house"
    )

st.divider()

# ── Predict Button ────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Predict Sale Price", use_container_width=True, type="primary")

if predict_btn:
    with st.spinner("Running prediction..."):
        input_data = {col: 0 for col in model_columns}

        input_data["MSSubClass"]   = ms_subclass
        input_data["LotArea"]      = lot_area
        input_data["OverallCond"]  = overall_cond
        input_data["YearBuilt"]    = year_built
        input_data["YearRemodAdd"] = year_remod
        input_data["BsmtFinSF2"]   = bsmt_fin_sf2
        input_data["TotalBsmtSF"]  = total_bsmt_sf

        for prefix, value in [
            ("MSZoning",    ms_zoning),
            ("LotConfig",   lot_config),
            ("BldgType",    bldg_type),
            ("Exterior1st", exterior1st),
        ]:
            col_name = f"{prefix}_{value}"
            if col_name in input_data:
                input_data[col_name] = 1

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

    st.success(f"### 💰 Estimated Sale Price: Rupee {prediction:,.0f}")

    # ── Breakdown metrics ─────────────────────────────────────────────────
    st.subheader("📊 Your Input Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lot Area",   f"{lot_area:,} sqft")
    m2.metric("Year Built", year_built)
    m3.metric("Condition",  f"{overall_cond}/10")
    m4.metric("Basement",   f"{total_bsmt_sf:,} sqft")

    with st.expander("🔎 See full input details"):
        display_df = pd.DataFrame({
            "Feature": [
                "Building Class", "Lot Area", "Overall Condition", "Year Built",
                "Year Remodelled", "Basement Finished SF2", "Total Basement SF",
                "Zoning", "Lot Config", "Building Type", "Exterior"
            ],
            "Value": [
                ms_subclass, f"{lot_area:,} sqft", f"{overall_cond}/10", year_built,
                year_remod, f"{bsmt_fin_sf2:,} sqft", f"{total_bsmt_sf:,} sqft",
                ms_zoning, lot_config, bldg_type, exterior1st
            ]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

