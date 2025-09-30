# app.py - Geography of Convenience Explorer
# Purpose: Main Streamlit application for retail site selection analysis
# Dependencies: Databricks SQL connector for Unity Catalog access
# Assumptions: Running on Databricks Apps with 2 vCPU, 6GB RAM
# Process: Multi-page app with session state management
# Risks: Large H3 datasets may require pagination

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from databricks import sql
import os
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Geography of Convenience",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
INDUSTRY_VERTICALS = {
    'RCT': {
        'name': 'Retail, Consumer & Travel',
        'subindustries': {
            'Retail': ['Grocery', 'Drugstores', 'Big Box', 'Specialty', 'E-commerce'],
            'CPG': ['Food & Beverage', 'Apparel', 'Household Goods'],
            'Transportation': ['Airlines', 'Hotels', 'QSR', 'Logistics']
        }
    },
    'FINS': {
        'name': 'Financial Services',
        'subindustries': {
            'Banking': ['Retail Banking', 'Digital Banking', 'Credit Unions'],
            'Insurance': ['Property & Casualty', 'Life & Health'],
            'Capital Markets': ['Investment Banks', 'Asset Managers']
        }
    },
    'CME': {
        'name': 'Communications, Media & Entertainment',
        'subindustries': {
            'Telecom': ['Wireless', 'Broadband', 'Mobile Networks'],
            'Media': ['Streaming', 'Digital Publishers', 'AdTech'],
            'Gaming': ['Online Gaming', 'E-sports', 'Publishers']
        }
    },
    'HLS': {
        'name': 'Healthcare & Life Sciences',
        'subindustries': {
            'Providers': ['Hospital Systems', 'Payers', 'Care Networks'],
            'Life Sciences': ['Pharma', 'Biotech', 'Clinical Research'],
            'Medical Devices': ['Device Manufacturers', 'Remote Monitoring']
        }
    }
}

RETAILERS = [
    'Kroger', 'Albertsons', 'Ahold Delhaize', 'Publix', 'H-E-B',
    'Meijer', 'Aldi', 'Whole Foods', 'Trader Joe\'s', 'Walmart',
    'Target', 'Costco', 'Sam\'s Club', 'Wegmans', 'Hy-Vee'
]

COMPETITOR_CATEGORIES = {
    'Premium': ['Whole Foods', 'Trader Joe\'s', 'Wegmans'],
    'Discount': ['Aldi', 'Walmart', 'Dollar General'],
    'Warehouse': ['Costco', 'Sam\'s Club', 'BJ\'s'],
    'Traditional': ['Kroger', 'Albertsons', 'Publix'],
    'Regional': ['H-E-B', 'Meijer', 'Hy-Vee']
}

# Custom CSS
st.markdown("""
<style>
    /* Dark theme with gradient backgrounds */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Header styling */
    h1 {
        color: #FFFFFF;
        font-weight: 800;
    }
    
    /* Card components */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: #FF3621;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(255, 54, 33, 0.3);
        background: #E62E1C;
    }
    
    /* Progress indicators */
    .progress-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin: 0 4px;
    }
    
    .progress-active {
        background: #FF3621;
    }
    
    .progress-complete {
        background: #00A972;
    }
    
    .progress-pending {
        background: #475569;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.1);
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'industry'
    st.session_state.industry = 'RCT'
    st.session_state.subindustry = 'Retail'
    st.session_state.segment = 'Grocery'
    st.session_state.retailer = None
    st.session_state.competitors = []
    st.session_state.geographic_scope = 'National'
    st.session_state.selected_state = None
    st.session_state.spend_rate = 0.124
    st.session_state.penetration = 'moderate'
    st.session_state.min_opportunity = 500000
    st.session_state.max_distance = None
    st.session_state.data_cache = {}

# Database connection (using environment variables from Databricks Apps)
@st.cache_resource
def get_db_connection():
    """Create connection to Databricks SQL warehouse"""
    try:
        connection = sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN")
        )
        return connection
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# Data loading functions with caching
@st.cache_data(ttl=3600)
def load_state_data():
    """Load state-level aggregated data"""
    # Mock data for demonstration
    states_df = pd.DataFrame({
        'state_code': ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI'],
        'state_name': ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania', 
                       'Illinois', 'Ohio', 'Georgia', 'North Carolina', 'Michigan'],
        'population': [39538223, 29145505, 21538187, 20201249, 13002700,
                      12812508, 11799448, 10711908, 10439388, 10077331],
        'households': [13044266, 10036776, 8436926, 7777229, 5194054,
                      4906019, 4798332, 3999345, 4098987, 4045776],
        'median_income': [85000, 65000, 60000, 72000, 63000, 68000, 58000, 61000, 57000, 59000],
        'opportunity_score': [92, 88, 85, 90, 78, 82, 75, 83, 79, 76]
    })
    states_df['tam'] = states_df['households'] * states_df['median_income'] * 0.124
    return states_df

@st.cache_data(ttl=3600)
def load_h3_data(bounds=None, limit=1000):
    """Load H3 hexagon data for mapping"""
    # Generate mock H3 data
    np.random.seed(42)
    count = min(limit, 1000)
    
    # Generate random H3 cells across US
    lats = np.random.uniform(25, 48, count)
    lons = np.random.uniform(-125, -66, count)
    
    h3_df = pd.DataFrame({
        'h3_index': [f'h3_{i}' for i in range(count)],
        'latitude': lats,
        'longitude': lons,
        'population': np.random.randint(100, 10000, count),
        'households': np.random.randint(50, 4000, count),
        'median_income': np.random.randint(40000, 120000, count),
        'opportunity_score': np.random.uniform(20, 100, count),
        'convenience_score': np.random.uniform(30, 95, count),
        'nearest_grocery_miles': np.random.uniform(0.5, 15, count),
        'grocery_density': np.random.uniform(0, 10, count)
    })
    
    h3_df['annual_potential'] = h3_df['households'] * h3_df['median_income'] * 0.124
    h3_df['opportunity_category'] = pd.cut(
        h3_df['opportunity_score'],
        bins=[0, 40, 60, 80, 100],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    if bounds:
        min_lat, min_lon, max_lat, max_lon = bounds
        h3_df = h3_df[
            (h3_df['latitude'] >= min_lat) & 
            (h3_df['latitude'] <= max_lat) &
            (h3_df['longitude'] >= min_lon) & 
            (h3_df['longitude'] <= max_lon)
        ]
    
    return h3_df

def create_progress_bar():
    """Create visual progress indicator"""
    pages = ['Industry', 'Market', 'Config', 'Explorer', 'Analysis']
    current_idx = pages.index(st.session_state.page.title() if st.session_state.page != 'industry' else 'Industry')
    
    progress_html = '<div style="text-align: center; margin: 2rem 0;">'
    for i, page in enumerate(pages):
        if i < current_idx:
            progress_html += '<span class="progress-dot progress-complete"></span>'
        elif i == current_idx:
            progress_html += '<span class="progress-dot progress-active"></span>'
        else:
            progress_html += '<span class="progress-dot progress-pending"></span>'
    progress_html += '</div>'
    
    st.markdown(progress_html, unsafe_allow_html=True)

# Page: Industry Selection
def page_industry_selection():
    st.title("🗺️ Geography of Convenience Explorer")
    st.markdown("### AI-Powered Retail Site Selection Platform")
    
    create_progress_bar()
    
    st.markdown("---")
    st.markdown("### Select Your Industry Focus")
    st.info("💡 This platform can be configured for any industry vertical. Currently demonstrating **Retail/Grocery** analysis.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        industry = st.selectbox(
            "**Industry**",
            options=list(INDUSTRY_VERTICALS.keys()),
            format_func=lambda x: f"{x} - {INDUSTRY_VERTICALS[x]['name']}",
            index=list(INDUSTRY_VERTICALS.keys()).index(st.session_state.industry)
        )
        st.session_state.industry = industry
    
    with col2:
        subindustries = list(INDUSTRY_VERTICALS[industry]['subindustries'].keys())
        subindustry = st.selectbox(
            "**Sub-Industry**",
            options=subindustries,
            index=subindustries.index(st.session_state.subindustry) if st.session_state.subindustry in subindustries else 0
        )
        st.session_state.subindustry = subindustry
    
    with col3:
        segments = INDUSTRY_VERTICALS[industry]['subindustries'][subindustry]
        segment = st.selectbox(
            "**Segment**",
            options=segments,
            index=segments.index(st.session_state.segment) if st.session_state.segment in segments else 0
        )
        st.session_state.segment = segment
    
    st.markdown("---")
    
    # Display configuration summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FF3621; margin-bottom: 1rem;">Selected Configuration</h3>
            <p><b>Industry:</b> {industry} - {INDUSTRY_VERTICALS[industry]['name']}</p>
            <p><b>Sub-Industry:</b> {subindustry}</p>
            <p><b>Segment:</b> {segment}</p>
            <br>
            <p style="color: #94A3B8; font-size: 0.9rem;">
            This configuration determines the data sources, competitive landscape, 
            and opportunity scoring algorithms used throughout the analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Ready to Continue?")
        st.markdown("Configure your market parameters in the next step.")
        if st.button("Continue to Market Setup →", use_container_width=True):
            st.session_state.page = 'market'
            st.rerun()

# Page: Market Setup
def page_market_setup():
    st.title("📍 Market Setup")
    st.markdown("Define your competitive landscape and geographic focus")
    
    create_progress_bar()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Your Retail Brand")
        retailer = st.selectbox(
            "Select your brand",
            options=[None] + RETAILERS,
            format_func=lambda x: "Choose a retailer..." if x is None else x,
            index=0 if st.session_state.retailer is None else RETAILERS.index(st.session_state.retailer) + 1
        )
        st.session_state.retailer = retailer
        
        st.markdown("### Geographic Scope")
        scope = st.selectbox(
            "Analysis scope",
            options=["National", "State", "Metro"],
            index=["National", "State", "Metro"].index(st.session_state.geographic_scope)
        )
        st.session_state.geographic_scope = scope
        
        if scope == "State":
            states_df = load_state_data()
            state = st.selectbox(
                "Select state",
                options=states_df['state_code'].tolist(),
                format_func=lambda x: f"{x} - {states_df[states_df['state_code']==x]['state_name'].iloc[0]}"
            )
            st.session_state.selected_state = state
    
    with col2:
        st.markdown("### Competitor Tracking")
        
        for category, retailers_list in COMPETITOR_CATEGORIES.items():
            with st.expander(f"**{category} Retailers**"):
                for comp in retailers_list:
                    if comp != retailer:
                        if st.checkbox(comp, key=f"comp_{comp}", 
                                       value=comp in st.session_state.competitors):
                            if comp not in st.session_state.competitors:
                                st.session_state.competitors.append(comp)
                        elif comp in st.session_state.competitors:
                            st.session_state.competitors.remove(comp)
        
        if st.session_state.competitors:
            st.success(f"Tracking {len(st.session_state.competitors)} competitors")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Industry", use_container_width=True):
            st.session_state.page = 'industry'
            st.rerun()
    
    with col2:
        if st.button("Continue to Configuration →", use_container_width=True):
            if not retailer:
                st.error("Please select a retailer to continue")
            else:
                st.session_state.page = 'config'
                st.rerun()

# Page: Business Configuration
def page_configuration():
    st.title("⚙️ Business Configuration")
    st.markdown("Customize analysis parameters to match your business strategy")
    
    create_progress_bar()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Economic Assumptions")
        
        spend_rate = st.slider(
            "Grocery Spending Rate (% of income)",
            min_value=8.0,
            max_value=20.0,
            value=st.session_state.spend_rate * 100,
            step=0.5,
            help="BLS standard is 12.4%"
        )
        st.session_state.spend_rate = spend_rate / 100
        
        penetration = st.radio(
            "Market Penetration Scenario",
            options=['conservative', 'moderate', 'aggressive'],
            format_func=lambda x: {
                'conservative': 'Conservative (5-8%)',
                'moderate': 'Moderate (8-12%)',
                'aggressive': 'Aggressive (12-20%)'
            }[x],
            index=['conservative', 'moderate', 'aggressive'].index(st.session_state.penetration)
        )
        st.session_state.penetration = penetration
        
        min_opp = st.number_input(
            "Minimum Opportunity Threshold ($)",
            min_value=100000,
            max_value=5000000,
            value=st.session_state.min_opportunity,
            step=100000,
            format="%d"
        )
        st.session_state.min_opportunity = min_opp
    
    with col2:
        st.markdown("### Expansion Strategy")
        
        use_distance = st.checkbox("Limit distance from existing stores")
        
        if use_distance:
            distance = st.slider(
                "Maximum distance (miles)",
                min_value=10,
                max_value=250,
                value=50,
                step=10
            )
            st.session_state.max_distance = distance
        else:
            st.session_state.max_distance = None
        
        st.markdown("### Analysis Impact")
        states_df = load_state_data()
        total_tam = states_df['tam'].sum()
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Total Market Size",
                f"${total_tam / 1e9:.1f}B",
                f"{spend_rate:.1%} spending rate"
            )
        
        with col_b:
            penetration_rates = {
                'conservative': 0.065,
                'moderate': 0.10,
                'aggressive': 0.16
            }
            potential = total_tam * penetration_rates[penetration]
            st.metric(
                "Revenue Potential",
                f"${potential / 1e9:.1f}B",
                f"{penetration_rates[penetration]:.1%} penetration"
            )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Market Setup", use_container_width=True):
            st.session_state.page = 'market'
            st.rerun()
    
    with col2:
        if st.button("Launch Explorer →", use_container_width=True):
            st.session_state.page = 'explorer'
            st.rerun()

# Page: Opportunity Explorer
def page_explorer():
    st.title("🎯 Opportunity Explorer")
    st.markdown("Interactive map showing high-value expansion opportunities")
    
    create_progress_bar()
    
    # Load data
    h3_data = load_h3_data(limit=500)
    states_data = load_state_data()
    
    # Apply filters based on configuration
    if st.session_state.min_opportunity:
        h3_data = h3_data[h3_data['annual_potential'] >= st.session_state.min_opportunity]
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        
        min_score = st.slider(
            "Min Opportunity Score",
            min_value=0,
            max_value=100,
            value=60
        )
        
        max_grocery_distance = st.slider(
            "Max Distance to Grocery (mi)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.5
        )
        
        h3_data = h3_data[
            (h3_data['opportunity_score'] >= min_score) &
            (h3_data['nearest_grocery_miles'] <= max_grocery_distance)
        ]
        
        st.markdown("---")
        st.markdown(f"**Opportunities Found:** {len(h3_data):,}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📍 Map View", "📊 Analytics", "📋 Data Table"])
    
    with tab1:
        # Create PyDeck map
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v11',
            initial_view_state=pdk.ViewState(
                latitude=39.8283,
                longitude=-98.5795,
                zoom=4,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    'HexagonLayer',
                    data=h3_data,
                    get_position='[longitude, latitude]',
                    radius=50000,
                    elevation_scale=100,
                    elevation_range=[0, 3000],
                    pickable=True,
                    extruded=True,
                    coverage=0.8,
                    get_fill_color='[255, 100 - opportunity_score, 50]',
                ),
                pdk.Layer(
                    'ScatterplotLayer',
                    data=h3_data.nlargest(20, 'opportunity_score'),
                    get_position='[longitude, latitude]',
                    get_color='[255, 54, 33, 160]',
                    get_radius=30000,
                    pickable=True,
                )
            ],
            tooltip={
                "html": "<b>Opportunity Score:</b> {opportunity_score:.1f}<br/>"
                       "<b>Population:</b> {population}<br/>"
                       "<b>Annual Potential:</b> ${annual_potential:,.0f}",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }
        ))
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Opportunity distribution
            fig_dist = px.histogram(
                h3_data,
                x='opportunity_score',
                nbins=20,
                title='Opportunity Score Distribution',
                color_discrete_sequence=['#FF3621']
            )
            fig_dist.update_layout(
                template='plotly_dark',
                height=300
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Top states by opportunity
            state_opps = states_data.nlargest(5, 'opportunity_score')
            fig_states = px.bar(
                state_opps,
                x='opportunity_score',
                y='state_code',
                orientation='h',
                title='Top States by Opportunity',
                color_discrete_sequence=['#00A972']
            )
            fig_states.update_layout(
                template='plotly_dark',
                height=300
            )
            st.plotly_chart(fig_states, use_container_width=True)
        
        # Scatter plot: Income vs Convenience
        fig_scatter = px.scatter(
            h3_data.sample(min(100, len(h3_data))),
            x='median_income',
            y='convenience_score',
            size='annual_potential',
            color='opportunity_score',
            title='Income vs Convenience Analysis',
            color_continuous_scale='Viridis'
        )
        fig_scatter.update_layout(
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        # Display top opportunities
        display_cols = [
            'opportunity_score', 'annual_potential', 'population',
            'households', 'median_income', 'nearest_grocery_miles'
        ]
        
        top_opps = h3_data.nlargest(100, 'opportunity_score')[display_cols]
        top_opps['annual_potential'] = top_opps['annual_potential'].apply(lambda x: f'${x:,.0f}')
        top_opps['median_income'] = top_opps['median_income'].apply(lambda x: f'${x:,.0f}')
        
        st.dataframe(
            top_opps,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = top_opps.to_csv(index=False)
        st.download_button(
            label="📥 Download Top 100 Opportunities",
            data=csv,
            file_name=f"opportunities_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back to Configuration", use_container_width=True):
            st.session_state.page = 'config'
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        if st.button("Continue to Analysis →", use_container_width=True):
            st.session_state.page = 'analysis'
            st.rerun()

# Page: Site Analysis
def page_analysis():
    st.title("📈 Site Analysis & Comparison")
    st.markdown("Deep-dive analysis of selected opportunities")
    
    create_progress_bar()
    
    # Load sample data for comparison
    h3_data = load_h3_data(limit=100)
    top_sites = h3_data.nlargest(5, 'opportunity_score')
    
    st.markdown("### Top 5 Opportunities for Detailed Analysis")
    
    # Create comparison metrics
    cols = st.columns(5)
    for i, (idx, site) in enumerate(top_sites.iterrows()):
        with cols[i]:
            st.metric(
                f"Site {i+1}",
                f"{site['opportunity_score']:.1f}",
                f"${site['annual_potential']/1e6:.1f}M"
            )
    
    st.markdown("---")
    
    # Detailed comparison table
    st.markdown("### Comparative Analysis")
    
    comparison_data = {
        'Metric': ['Opportunity Score', 'Annual Potential', 'Population', 
                   'Median Income', 'Nearest Grocery', 'Convenience Score'],
        'Site 1': [
            f"{top_sites.iloc[0]['opportunity_score']:.1f}",
            f"${top_sites.iloc[0]['annual_potential']/1e6:.1f}M",
            f"{top_sites.iloc[0]['population']:,}",
            f"${top_sites.iloc[0]['median_income']:,}",
            f"{top_sites.iloc[0]['nearest_grocery_miles']:.1f} mi",
            f"{top_sites.iloc[0]['convenience_score']:.1f}"
        ],
        'Site 2': [
            f"{top_sites.iloc[1]['opportunity_score']:.1f}",
            f"${top_sites.iloc[1]['annual_potential']/1e6:.1f}M",
            f"{top_sites.iloc[1]['population']:,}",
            f"{top_sites.iloc[1]['median_income']:,}",
            f"{top_sites.iloc[1]['nearest_grocery_miles']:.1f} mi",
            f"{top_sites.iloc[1]['convenience_score']:.1f}"
        ],
        'Site 3': [
            f"{top_sites.iloc[2]['opportunity_score']:.1f}",
            f"${top_sites.iloc[2]['annual_potential']/1e6:.1f}M",
            f"{top_sites.iloc[2]['population']:,}",
            f"{top_sites.iloc[2]['median_income']:,}",
            f"{top_sites.iloc[2]['nearest_grocery_miles']:.1f} mi",
            f"{top_sites.iloc[2]['convenience_score']:.1f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Radar chart for comparison
    categories = ['Opportunity', 'Population', 'Income', 'Convenience', 'Market Gap']
    
    fig = go.Figure()
    
    for i in range(3):
        fig.add_trace(go.Scatterpolar(
            r=[
                top_sites.iloc[i]['opportunity_score'],
                top_sites.iloc[i]['population'] / 1000,
                top_sites.iloc[i]['median_income'] / 1000,
                top_sites.iloc[i]['convenience_score'],
                100 - top_sites.iloc[i]['grocery_density'] * 10
            ],
            theta=categories,
            fill='toself',
            name=f'Site {i+1}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        template='plotly_dark',
        title="Multi-Dimensional Site Comparison"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Summary and recommendations
    st.markdown("### 💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #FF3621;">Primary Opportunity</h4>
            <p><b>Site 1</b> offers the highest opportunity score with strong demographics
            and minimal competition. The area shows high income levels with limited
            grocery access, creating ideal conditions for market entry.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #00A972;">Expansion Strategy</h4>
            <p>Based on your {st.session_state.penetration} penetration scenario,
            focusing on the top 3 sites could generate ${(top_sites.head(3)['annual_potential'].sum() / 1e6):.1f}M
            in annual revenue potential.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Back to Explorer", use_container_width=True):
            st.session_state.page = 'explorer'
            st.rerun()
    
    with col2:
        if st.button("📊 Generate Report", use_container_width=True):
            st.success("Report generation feature coming soon!")
    
    with col3:
        if st.button("🏠 Start New Analysis", use_container_width=True):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Main app routing
def main():
    # Route to appropriate page based on session state
    if st.session_state.page == 'industry':
        page_industry_selection()
    elif st.session_state.page == 'market':
        page_market_setup()
    elif st.session_state.page == 'config':
        page_configuration()
    elif st.session_state.page == 'explorer':
        page_explorer()
    elif st.session_state.page == 'analysis':
        page_analysis()
    else:
        page_industry_selection()

if __name__ == "__main__":
    main()