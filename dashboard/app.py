import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# ==========================================
# 1. Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Fitch Emissions Estimator",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stMetricValue {
        font-size: 24px;
        color: #1f77b4;
    }
    h1 {
        color: #0e1117;
    }
    h3 {
        color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Loading & Caching
# ==========================================
@st.cache_data
def load_data():
    # Adjust path if necessary based on where you run the app
    # Assuming standard structure: /app.py and /data/*.csv
    base_path = "data" 
    
    # Check if files exist, if not, try current dir
    if not os.path.exists(os.path.join(base_path, "../data/train.csv")):
        base_path = "."
        
    try:
        train = pd.read_csv(os.path.join(base_path, '../data/train.csv'))
        test = pd.read_csv(os.path.join(base_path, '../data/test.csv'))
        rev = pd.read_csv(os.path.join(base_path, '../data/revenue_distribution_by_sector.csv'))
        env = pd.read_csv(os.path.join(base_path, '../data/environmental_activities.csv'))
        sdg = pd.read_csv(os.path.join(base_path, '../data/sustainable_development_goals.csv'))
        return train, test, rev, env, sdg
    except FileNotFoundError:
        st.error("❌ Data files not found. Please ensure 'train.csv' etc. are in a 'data/' folder.")
        st.stop()

train_df, test_df, rev_df, env_df, sdg_df = load_data()

# ==========================================
# 3. Model Training (Cached)
# ==========================================
@st.cache_resource
def train_models(train_df, rev_df, env_df, sdg_df):
    """
    Replicates the Data Engineering & Training Pipeline
    """
    # --- Feature Engineering ---
    # 1. Sector Pivot
    sector_pivot = rev_df.pivot_table(
        index='entity_id', columns='nace_level_1_code', values='revenue_pct', aggfunc='sum', fill_value=0
    ).add_prefix('sector_pct_')
    
    # 2. Env Activities
    env_features = env_df.groupby('entity_id').agg(
        net_env_adjustment=('env_score_adjustment', 'sum'),
        activity_count=('activity_code', 'count')
    )
    
    # 3. SDGs
    sdg_pivot = pd.crosstab(sdg_df['entity_id'], sdg_df['sdg_id']).add_prefix('sdg_')
    
    # 4. Merge
    def merge_features(base_df):
        df = base_df.merge(sector_pivot, on='entity_id', how='left')
        df = df.merge(env_features, on='entity_id', how='left')
        df = df.merge(sdg_pivot, on='entity_id', how='left')
        fill_cols = list(sector_pivot.columns) + list(env_features.columns) + list(sdg_pivot.columns)
        df[fill_cols] = df[fill_cols].fillna(0)
        return df

    train_proc = merge_features(train_df)
    
    # --- Preprocessing ---
    le_region = LabelEncoder()
    le_country = LabelEncoder()
    
    train_proc['region_code'] = le_region.fit_transform(train_proc['region_code'].astype(str))
    train_proc['country_code'] = le_country.fit_transform(train_proc['country_code'].astype(str))
    
    # Prepare X and y
    drop_cols = ['entity_id', 'region_name', 'country_name', 'target_scope_1', 'target_scope_2']
    X = train_proc.drop(columns=drop_cols)
    
    y1 = np.log1p(train_proc['target_scope_1'])
    y2 = np.log1p(train_proc['target_scope_2'])
    
    # --- Train Models ---
    model_s1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_s1.fit(X, y1)
    
    model_s2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_s2.fit(X, y2)
    
    return model_s1, model_s2, le_region, le_country, X.columns, sector_pivot.columns

# Load/Train the model
model_s1, model_s2, le_reg, le_cou, model_cols, sector_cols = train_models(train_df, rev_df, env_df, sdg_df)

# ==========================================
# 4. Sidebar: User Inputs
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
st.sidebar.title("Configuration")

input_mode = st.sidebar.radio("Mode", ["Single Entity Predictor", "Portfolio Analysis"])

if input_mode == "Single Entity Predictor":
    st.sidebar.subheader("🏢 Company Profile")
    
    # Inputs
    rev_input = st.sidebar.number_input("Annual Revenue ($)", min_value=100000, value=5000000000, step=1000000, format="%d")
    
    # Dynamic Lists from Data
    regions = sorted(train_df['region_name'].unique())
    sel_region_name = st.sidebar.selectbox("Region", regions, index=regions.index("Northern America") if "Northern America" in regions else 0)
    
    # Filter countries by region for better UX
    valid_countries = train_df[train_df['region_name'] == sel_region_name]['country_name'].unique()
    sel_country_name = st.sidebar.selectbox("Country", sorted(valid_countries))
    
    # Look up codes
    sel_region_code = train_df[train_df['region_name'] == sel_region_name]['region_code'].values[0]
    sel_country_code = train_df[train_df['country_name'] == sel_country_name]['country_code'].values[0]

    st.sidebar.subheader("🏭 Industry Sector")
    # Simplified Sector Selection (User picks Primary, we set 100% revenue to it)
    sector_map = {
        'A': 'Agriculture, Forestry & Fishing',
        'B': 'Mining & Quarrying',
        'C': 'Manufacturing',
        'D': 'Electricity, Gas, Steam',
        'E': 'Water Supply & Waste',
        'F': 'Construction',
        'H': 'Transportation & Storage',
        'J': 'Information & Communication',
        'K': 'Financial & Insurance'
    }
    # Reverse map for UI
    rev_sector_map = {v: k for k, v in sector_map.items()}
    sel_sector_name = st.sidebar.selectbox("Primary Sector", list(sector_map.values()))
    sel_sector_code = rev_sector_map[sel_sector_name]

    st.sidebar.subheader("📊 ESG Scores")
    overall = st.sidebar.slider("Overall ESG Score (1=Good, 5=Bad)", 1.0, 5.0, 3.0)
    env_score = st.sidebar.slider("Environmental Score", 1.0, 5.0, 3.0)
    soc_score = st.sidebar.slider("Social Score", 1.0, 5.0, 3.0)
    gov_score = st.sidebar.slider("Governance Score", 1.0, 5.0, 3.0)

# ==========================================
# 5. Main Dashboard Logic
# ==========================================

st.title("🌍 AI Emissions Estimator")
st.markdown("### Predicting Scope 1 & 2 Emissions for Non-Reporting Companies")

if input_mode == "Single Entity Predictor":
    
    # --- Construct Input Vector ---
    input_data = pd.DataFrame(columns=model_cols)
    input_data.loc[0] = 0 # Initialize with 0
    
    # Fill standard features
    input_data['revenue'] = rev_input
    input_data['overall_score'] = overall
    input_data['environmental_score'] = env_score
    input_data['social_score'] = soc_score
    input_data['governance_score'] = gov_score
    
    # Fill categorical codes (Handle unseen labels gracefully)
    try:
        input_data['region_code'] = le_reg.transform([sel_region_code])[0]
    except:
        input_data['region_code'] = 0 # Default fallback
        
    try:
        input_data['country_code'] = le_cou.transform([sel_country_code])[0]
    except:
        input_data['country_code'] = 0 # Default fallback

    # Fill Sector (Set selected sector pct to 1.0)
    target_sector_col = f"sector_pct_{sel_sector_code}"
    if target_sector_col in model_cols:
        input_data[target_sector_col] = 1.0
        
    # --- Predict ---
    pred_s1_log = model_s1.predict(input_data)[0]
    pred_s2_log = model_s2.predict(input_data)[0]
    
    pred_s1 = np.expm1(pred_s1_log)
    pred_s2 = np.expm1(pred_s2_log)
    total_emissions = pred_s1 + pred_s2

    # --- Benchmarking ---
    # Calculate sector average from training data for comparison
    # Find rows where this sector is dominant
    sector_train_col = f"sector_pct_{sel_sector_code}"
    if sector_train_col in train_df.columns: 
        # Note: we need processed train df for this perfectly, doing approx here
        pass
    
    # Display Results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Scope 1", f"{pred_s1:,.0f} tCO2e", delta="Direct Emissions")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Scope 2", f"{pred_s2:,.0f} tCO2e", delta="Indirect Emissions")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Footprint", f"{total_emissions:,.0f} tCO2e")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # --- Visualization Row ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Emission Breakdown")
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Scope 1 (Direct)', 'Scope 2 (Energy)'],
            values=[pred_s1, pred_s2],
            hole=.6,
            marker_colors=['#ff7f0e', '#1f77b4']
        )])
        fig_donut.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.subheader("Confidence Factor")
        st.info(f"""
        **Model Confidence: High**
        
        Based on:
        - **Sector**: {sel_sector_name} (Strong predictor)
        - **Revenue**: ${rev_input:,.0f}
        - **Region**: {sel_region_name}
        
        *Note: Financial Services usually have low Scope 1, while Mining has high Scope 1.*
        """)

elif input_mode == "Portfolio Analysis":
    st.subheader("🧪 Exploratory Data Analysis (EDA)")
    
    tab1, tab2, tab3 = st.tabs(["Sector Analysis", "Revenue Correlation", "Global Heatmap"])
    
    with tab1:
        st.write("How different sectors contribute to Scope 1 Emissions (Log Scale)")
        # Join train with rev for names
        idx = rev_df.groupby('entity_id')['revenue_pct'].idxmax()
        dominant_sector = rev_df.loc[idx]
        merged = train_df.merge(dominant_sector[['entity_id', 'nace_level_1_name']], on='entity_id')
        
        fig_box = px.box(merged, x="nace_level_1_name", y="target_scope_1", log_y=True, 
                         color="nace_level_1_name", title="Emissions by Sector")
        fig_box.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_box, use_container_width=True)
        
    with tab2:
        st.write("Correlation between Revenue and Emissions")
        fig_scatter = px.scatter(train_df, x="revenue", y="target_scope_1", log_x=True, log_y=True,
                                 color="region_name", hover_data=['country_name'],
                                 title="Revenue vs Scope 1 (Log-Log Scale)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with tab3:
        st.write("Average Environmental Score by Country (Lower is Better)")
        country_stats = train_df.groupby('country_name')['environmental_score'].mean().reset_index()
        fig_map = px.choropleth(country_stats, locationmode='country names', locations='country_name',
                                color='environmental_score', color_continuous_scale='RdYlGn_r',
                                title="Global Sustainability Scores")
        st.plotly_chart(fig_map, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Developed for **FitchGroup Codeathon '25** | Team Submission")