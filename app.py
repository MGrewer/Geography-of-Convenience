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
css_style = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    h1 {
        color: #FFFFFF;
        font-weight: 800;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
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
    
    [data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.1);
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
    }
</style>
"""

st.markdown(css_style, unsafe_allow_html=True)

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
    st.session_state.map_bounds = None
    st.session_state.zoom_level = 4
    st.session_state.map_center = [39.8283, -98.5795]

# Database connection using Databricks Apps SQL warehouse resource
@st.cache_resource
def get_db_connection():
    """Create connection to SQL warehouse using Databricks SDK Config"""
    try:
        from databricks.sdk.core import Config
        
        # Use Databricks SDK Config which handles authentication automatically
        cfg = Config()
        
        # Specific warehouse HTTP path for Geography of Convenience
        http_path = "/sql/1.0/warehouses/85813e6551ef07c5"
        
        # Connect using the Config's authentication
        connection = sql.connect(
            server_hostname=cfg.host,
            http_path=http_path,
            credentials_provider=lambda: cfg.authenticate
        )
        
        # Test the connection
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        
        return connection
        
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        
        # Debug information
        st.info("Debug Information:")
        st.write(f"DATABRICKS_CLIENT_ID exists: {bool(os.getenv('DATABRICKS_CLIENT_ID'))}")
        st.write(f"DATABRICKS_CLIENT_SECRET exists: {bool(os.getenv('DATABRICKS_CLIENT_SECRET'))}")
        st.write(f"Host: {cfg.host if 'cfg' in locals() else 'Not available'}")
        
        st.info("""
        **Connection Troubleshooting:**
        1. Ensure the SQL warehouse is attached as a resource to this app
        2. Check that the app's service principal has SELECT access to valhalla catalog
        3. Verify the warehouse ID matches your 'Geography of Convenience' warehouse
        """)
        st.stop()
        return None

# Data loading functions - REAL DATA ONLY
@st.cache_data(ttl=3600)
def load_state_data():
    """Load state-level aggregated data from Unity Catalog"""
    conn = get_db_connection()
    if not conn:
        st.error("No database connection. Please attach a SQL warehouse to this app.")
        st.stop()
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            state_code,
            state_name,
            population_total as population,
            households_total as households,
            avg_median_income as median_income,
            market_size_score as opportunity_score,
            convenience_score,
            grocery_store_count,
            population_density
        FROM valhalla.4_app_data.state_all_metrics
        ORDER BY opportunity_score DESC
    """)
    
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    states_df = pd.DataFrame(data, columns=columns)
    states_df['tam'] = states_df['households'] * states_df['median_income'] * 0.124
    
    cursor.close()
    conn.close()
    return states_df

@st.cache_data(ttl=3600)
def load_h3_data(state_filter=None, bounds=None, limit=5000):
    """Load H3 hexagon data with opportunity scores"""
    conn = get_db_connection()
    if not conn:
        st.error("No database connection")
        st.stop()
    
    cursor = conn.cursor()
    
    query = """
        SELECT 
            h3.h3_index,
            h3.latitude,
            h3.longitude,
            h3.population,
            h3.households,
            h3.median_income,
            h3.opportunity_score,
            h3.convenience_score,
            h3.nearest_grocery_distance,
            h3.grocery_stores_within_5min,
            h3.grocery_stores_within_10min,
            h3.grocery_stores_within_15min,
            h3.demographic_diversity_index,
            h3.income_inequality_index,
            h3.households * h3.median_income * 0.124 as annual_potential
        FROM valhalla.4_app_data.h3_all_metrics h3
        WHERE h3.opportunity_score IS NOT NULL
    """
    
    if state_filter:
        query += f" AND h3.state_code = '{state_filter}'"
    
    if bounds:
        min_lat, min_lon, max_lat, max_lon = bounds
        query += f"""
            AND h3.latitude BETWEEN {min_lat} AND {max_lat}
            AND h3.longitude BETWEEN {min_lon} AND {max_lon}
        """
    
    query += f" ORDER BY h3.opportunity_score DESC LIMIT {limit}"
    
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    h3_df = pd.DataFrame(data, columns=columns)
    
    # Add opportunity categories for color mapping
    if not h3_df.empty:
        h3_df['opportunity_category'] = pd.cut(
            h3_df['opportunity_score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
    
    cursor.close()
    conn.close()
    return h3_df

@st.cache_data(ttl=3600)
def load_grocery_stores(state_filter=None, bounds=None):
    """Load enriched grocery store locations"""
    conn = get_db_connection()
    if not conn:
        st.error("No database connection")
        st.stop()
    
    cursor = conn.cursor()
    
    query = """
        SELECT 
            place_id,
            brand,
            name,
            latitude,
            longitude,
            full_address,
            city,
            state,
            zip_code,
            parent_company,
            store_type,
            confidence_score,
            estimated_annual_revenue,
            h3_index
        FROM valhalla.3_gold.carto_us_grocery_enriched
        WHERE confidence > 0.7
    """
    
    if state_filter:
        query += f" AND state = '{state_filter}'"
    
    if bounds:
        min_lat, min_lon, max_lat, max_lon = bounds
        query += f"""
            AND latitude BETWEEN {min_lat} AND {max_lat}
            AND longitude BETWEEN {min_lon} AND {max_lon}
        """
    
    query += " LIMIT 5000"
    
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    stores_df = pd.DataFrame(data, columns=columns)
    
    cursor.close()
    conn.close()
    return stores_df

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
        config_html = f"""
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
        """
        st.markdown(config_html, unsafe_allow_html=True)
    
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
    
    # Get store counts from database
    @st.cache_data(ttl=3600)
    def get_store_counts():
        """Get store counts by brand"""
        conn = get_db_connection()
        if not conn:
            # Return mock counts if no connection
            return {brand: np.random.randint(100, 5000) for brand in RETAILERS}
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                brand,
                COUNT(*) as store_count
            FROM valhalla.3_gold.carto_us_grocery_enriched
            WHERE confidence > 0.7
                AND brand IS NOT NULL
            GROUP BY brand
            ORDER BY store_count DESC
        """)
        
        data = cursor.fetchall()
        store_counts = {row[0]: row[1] for row in data}
        
        # Get unaffiliated count
        cursor.execute("""
            SELECT COUNT(*) as unaffiliated_count
            FROM valhalla.3_gold.carto_us_grocery_enriched
            WHERE confidence > 0.7
                AND (brand IS NULL OR brand = '' OR brand = 'Independent')
        """)
        
        unaffiliated = cursor.fetchone()[0]
        store_counts['Unaffiliated'] = unaffiliated
        
        cursor.close()
        conn.close()
        return store_counts
    
    store_counts = get_store_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏪 Your Retail Brand")
        retailer = st.selectbox(
            "Select your brand",
            options=[None] + RETAILERS,
            format_func=lambda x: "Choose a retailer..." if x is None else f"{x} ({store_counts.get(x, 0):,} stores)" if x in store_counts else x,
            index=0 if st.session_state.retailer is None else RETAILERS.index(st.session_state.retailer) + 1
        )
        st.session_state.retailer = retailer
        
        if retailer and retailer in store_counts:
            st.info(f"📊 {retailer} operates **{store_counts[retailer]:,}** stores nationwide")
        
        st.markdown("### 📍 Geographic Scope")
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
        st.markdown("### 🎯 Competitor Tracking")
        st.markdown("Select competitors to monitor in your analysis")
        
        # Initialize competitors list if needed
        if 'competitors' not in st.session_state:
            st.session_state.competitors = []
        
        # Track selections
        selected_competitors = st.session_state.competitors.copy()
        
        # Category toggles with store counts
        for category, retailers_list in COMPETITOR_CATEGORIES.items():
            # Filter out the selected retailer from the list
            available_retailers = [r for r in retailers_list if r != retailer]
            
            if available_retailers:
                # Calculate total stores in category
                category_total = sum(store_counts.get(r, 0) for r in available_retailers)
                
                # Create expander with category name and store count
                with st.expander(f"**{category} Retailers** ({len(available_retailers)} chains, {category_total:,} stores)"):
                    
                    # Select all checkbox for the category
                    col_all, col_none = st.columns(2)
                    with col_all:
                        if st.button(f"Select All {category}", key=f"all_{category}", use_container_width=True):
                            for r in available_retailers:
                                if r not in selected_competitors:
                                    selected_competitors.append(r)
                    
                    with col_none:
                        if st.button(f"Clear {category}", key=f"none_{category}", use_container_width=True):
                            selected_competitors = [c for c in selected_competitors if c not in available_retailers]
                    
                    st.markdown("---")
                    
                    # Individual retailer checkboxes with store counts
                    for comp in available_retailers:
                        store_count = store_counts.get(comp, 0)
                        label = f"{comp} ({store_count:,} stores)" if store_count > 0 else comp
                        
                        if st.checkbox(
                            label, 
                            key=f"comp_{comp}",
                            value=comp in selected_competitors
                        ):
                            if comp not in selected_competitors:
                                selected_competitors.append(comp)
                        else:
                            if comp in selected_competitors:
                                selected_competitors.remove(comp)
        
        # Unaffiliated stores option
        st.markdown("---")
        unaffiliated_count = store_counts.get('Unaffiliated', 30000)
        include_unaffiliated = st.checkbox(
            f"**Include {unaffiliated_count:,} unaffiliated/independent stores**",
            help="Small independent grocers, local chains, and stores without identified brand affiliations"
        )
        
        if include_unaffiliated and 'Unaffiliated' not in selected_competitors:
            selected_competitors.append('Unaffiliated')
        elif not include_unaffiliated and 'Unaffiliated' in selected_competitors:
            selected_competitors.remove('Unaffiliated')
        
        # Update session state
        st.session_state.competitors = selected_competitors
        
        # Summary of selections
        if selected_competitors:
            total_competitor_stores = sum(store_counts.get(c, 0) for c in selected_competitors)
            st.success(f"""
                **Tracking {len(selected_competitors)} competitors**  
                Total stores: {total_competitor_stores:,}
            """)
            
            # Quick actions
            col_clear, col_preset = st.columns(2)
            with col_clear:
                if st.button("Clear All", use_container_width=True):
                    st.session_state.competitors = []
                    st.rerun()
            
            with col_preset:
                if st.button("Select Top 10", use_container_width=True):
                    # Select top 10 by store count
                    top_brands = sorted(
                        [(b, c) for b, c in store_counts.items() if b != retailer and b != 'Unaffiliated'],
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                    st.session_state.competitors = [b[0] for b in top_brands]
                    st.rerun()
        else:
            st.info("No competitors selected - analysis will show all market opportunities")
    
    st.markdown("---")
    
    # Summary section
    st.markdown("### 📊 Market Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        your_stores = store_counts.get(retailer, 0) if retailer else 0
        st.metric("Your Stores", f"{your_stores:,}")
    
    with col2:
        comp_stores = sum(store_counts.get(c, 0) for c in selected_competitors)
        st.metric("Competitor Stores", f"{comp_stores:,}")
    
    with col3:
        total_market = sum(store_counts.values())
        st.metric("Total Market", f"{total_market:,}")
    
    if retailer and selected_competitors:
        your_share = your_stores / total_market * 100 if total_market > 0 else 0
        comp_share = comp_stores / total_market * 100 if total_market > 0 else 0
        
        st.markdown(f"""
        **Market Share Analysis:**
        - Your current share: {your_share:.1f}%
        - Selected competitors: {comp_share:.1f}%
        - Remaining market: {100 - your_share - comp_share:.1f}%
        """)
    
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
    st.markdown("Zoom in to explore opportunities • Stores at national level → H3 cells & isochrones at local level")
    
    create_progress_bar()
    
    # Initialize map state
    if 'map_bounds' not in st.session_state:
        st.session_state.map_bounds = None
    if 'zoom_level' not in st.session_state:
        st.session_state.zoom_level = 4
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [39.8283, -98.5795]
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🗺️ Map Controls")
        
        # Reset view button
        if st.button("🔄 Reset to National View", use_container_width=True):
            st.session_state.zoom_level = 4
            st.session_state.map_center = [39.8283, -98.5795]
            st.session_state.map_bounds = None
            st.rerun()
        
        st.markdown("---")
        
        # Current zoom level indicator
        zoom_description = {
            range(0, 5): "National",
            range(5, 8): "Regional", 
            range(8, 11): "Metro",
            range(11, 14): "City",
            range(14, 20): "Neighborhood"
        }
        
        current_zoom_desc = "National"
        for zoom_range, desc in zoom_description.items():
            if st.session_state.zoom_level in zoom_range:
                current_zoom_desc = desc
                break
        
        st.metric("View Level", current_zoom_desc, f"Zoom: {st.session_state.zoom_level}")
        
        # Dynamic layer visibility based on zoom
        st.markdown("### 📊 Visible Layers")
        
        if st.session_state.zoom_level < 8:
            st.info("""
            **National/Regional View:**
            • 📍 Store locations
            • 🔥 State heat map
            
            *Zoom in to see H3 cells & isochrones*
            """)
        else:
            st.success("""
            **Local View Active:**
            • 📍 Store locations
            • 🔷 H3 opportunity cells
            • ⭕ Drive-time isochrones
            """)
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filters")
        
        # Only show opportunity filter at detailed zoom
        if st.session_state.zoom_level >= 8:
            min_opportunity = st.slider(
                "Min Opportunity Score",
                min_value=0,
                max_value=100,
                value=50,
                help="Filter H3 cells by opportunity score"
            )
        else:
            min_opportunity = 0
        
        # Store type filter
        show_your_stores = st.checkbox(f"Show {st.session_state.retailer or 'Your'} Stores", value=True)
        show_competitor_stores = st.checkbox("Show Competitor Stores", value=True)
        show_unaffiliated = st.checkbox("Show Unaffiliated Stores", value=False)
        
        # Advanced options for detailed view
        if st.session_state.zoom_level >= 10:
            st.markdown("### ⚙️ Advanced Options")
            isochrone_minutes = st.selectbox(
                "Drive-time isochrones",
                options=[5, 10, 15, 20],
                index=1,
                help="Show areas reachable within X minutes"
            )
            
            show_3d = st.checkbox("3D View (Revenue Potential)", value=False)
        else:
            isochrone_minutes = 10
            show_3d = False
    
    # Main map area
    st.markdown(f"### 🗺️ Interactive Map - {current_zoom_desc} View")
    
    # Determine what data to load based on zoom level
    layers = []
    
    # Get current viewport bounds for data loading
    if st.session_state.map_bounds:
        bounds = st.session_state.map_bounds
    else:
        # Default US bounds
        bounds = [25, -125, 48, -66]  # [min_lat, min_lon, max_lat, max_lon]
    
    # Load store data (always visible)
    @st.cache_data(ttl=600)
    def load_stores_for_viewport(bounds, zoom):
        """Load stores with sampling based on zoom level"""
        conn = get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        cursor = conn.cursor()
        
        # Sample stores at national level for performance
        if zoom < 6:
            sample_clause = "TABLESAMPLE (10 PERCENT)"
        elif zoom < 8:
            sample_clause = "TABLESAMPLE (25 PERCENT)"
        else:
            sample_clause = ""  # Show all stores at local level
        
        query = f"""
        SELECT 
            place_id,
            brand,
            name,
            latitude,
            longitude,
            city,
            state,
            CASE 
                WHEN brand = '{st.session_state.retailer}' THEN 'your'
                WHEN brand IN ({','.join([f"'{c}'" for c in st.session_state.competitors if c != 'Unaffiliated'])}) THEN 'competitor'
                WHEN brand IS NULL OR brand = '' OR brand = 'Independent' THEN 'unaffiliated'
                ELSE 'other'
            END as store_type,
            estimated_annual_revenue
        FROM valhalla.3_gold.carto_us_grocery_enriched {sample_clause}
        WHERE confidence > 0.7
            AND latitude BETWEEN {bounds[0]} AND {bounds[2]}
            AND longitude BETWEEN {bounds[1]} AND {bounds[3]}
        """
        
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        stores_df = pd.DataFrame(data, columns=columns)
        
        cursor.close()
        conn.close()
        return stores_df
    
    stores_data = load_stores_for_viewport(bounds, st.session_state.zoom_level)
    
    # Filter stores based on sidebar selections
    filtered_stores = stores_data.copy()
    if not show_your_stores:
        filtered_stores = filtered_stores[filtered_stores['store_type'] != 'your']
    if not show_competitor_stores:
        filtered_stores = filtered_stores[filtered_stores['store_type'] != 'competitor']
    if not show_unaffiliated:
        filtered_stores = filtered_stores[filtered_stores['store_type'] != 'unaffiliated']
    
    # Store points layer
    if not filtered_stores.empty:
        # Color stores by type
        filtered_stores['color'] = filtered_stores['store_type'].map({
            'your': [0, 255, 0, 200],        # Green for your stores
            'competitor': [255, 0, 0, 200],   # Red for competitors
            'unaffiliated': [128, 128, 128, 150],  # Gray for unaffiliated
            'other': [200, 200, 200, 100]     # Light gray for others
        })
        
        # Size based on zoom level
        radius = 5000 if st.session_state.zoom_level < 6 else 2000 if st.session_state.zoom_level < 10 else 500
        
        layers.append(
            pdk.Layer(
                'ScatterplotLayer',
                data=filtered_stores,
                get_position='[longitude, latitude]',
                get_color='color',
                get_radius=radius,
                pickable=True,
                auto_highlight=True,
            )
        )
    
    # Load H3 cells only at detailed zoom levels
    if st.session_state.zoom_level >= 8:
        @st.cache_data(ttl=300)
        def load_h3_for_viewport(bounds, min_score):
            """Load H3 cells for current viewport"""
            conn = get_db_connection()
            if not conn:
                return pd.DataFrame()
            
            cursor = conn.cursor()
            
            # Limit H3 cells based on zoom for performance
            if st.session_state.zoom_level < 10:
                limit = 1000
            elif st.session_state.zoom_level < 12:
                limit = 5000
            else:
                limit = 10000
            
            cursor.execute(f"""
                SELECT 
                    h3_index,
                    latitude,
                    longitude,
                    population,
                    households,
                    median_income,
                    opportunity_score,
                    convenience_score,
                    nearest_grocery_distance,
                    grocery_stores_within_10min,
                    households * median_income * 0.124 as annual_potential
                FROM valhalla.4_app_data.h3_all_metrics
                WHERE latitude BETWEEN {bounds[0]} AND {bounds[2]}
                    AND longitude BETWEEN {bounds[1]} AND {bounds[3]}
                    AND opportunity_score >= {min_score}
                ORDER BY opportunity_score DESC
                LIMIT {limit}
            """)
            
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            h3_df = pd.DataFrame(data, columns=columns)
            
            cursor.close()
            conn.close()
            return h3_df
        
        h3_data = load_h3_for_viewport(bounds, min_opportunity)
        
        if not h3_data.empty:
            # H3 layer with opportunity-based coloring
            layers.append(
                pdk.Layer(
                    'H3HexagonLayer',
                    data=h3_data,
                    get_hexagon='h3_index',
                    get_fill_color='[255 - opportunity_score * 2, opportunity_score * 2.5, 50, 140]',
                    get_line_color=[255, 255, 255, 80],
                    line_width_min_pixels=0.5,
                    pickable=True,
                    auto_highlight=True,
                    extruded=show_3d,
                    get_elevation='annual_potential / 500000' if show_3d else 0,
                    elevation_scale=100 if show_3d else 0,
                )
            )
    
    # Create the deck
    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v11',
        initial_view_state=pdk.ViewState(
            latitude=st.session_state.map_center[0],
            longitude=st.session_state.map_center[1],
            zoom=st.session_state.zoom_level,
            pitch=45 if show_3d else 0,
            bearing=0,
        ),
        layers=layers,
        tooltip={
            "html": """
            <div style="background: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px; color: white;">
                {store_type === 'your' || store_type === 'competitor' || store_type === 'unaffiliated' ? 
                    `<b>${brand || 'Independent'}</b><br/>
                     ${name || 'Unknown'}<br/>
                     ${city}, ${state}<br/>
                     Type: ${store_type}` :
                    `<b>Opportunity Score:</b> ${opportunity_score}<br/>
                     <b>Population:</b> ${population}<br/>
                     <b>Annual Potential:</b> $${annual_potential}<br/>
                     <b>Nearest Grocery:</b> ${nearest_grocery_distance} mi`
                }
            </div>
            """,
            "style": {"color": "white"}
        }
    )
    
    st.pydeck_chart(deck)
    
    # Dynamic info panel based on zoom level
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.zoom_level < 8:
            your_stores_count = len(filtered_stores[filtered_stores['store_type'] == 'your']) if not filtered_stores.empty else 0
            st.metric("Your Stores Visible", f"{your_stores_count:,}")
        else:
            top_opp = h3_data['opportunity_score'].max() if 'h3_data' in locals() and not h3_data.empty else 0
            st.metric("Top Opportunity", f"{top_opp:.1f}")
    
    with col2:
        if st.session_state.zoom_level < 8:
            comp_stores_count = len(filtered_stores[filtered_stores['store_type'] == 'competitor']) if not filtered_stores.empty else 0
            st.metric("Competitor Stores", f"{comp_stores_count:,}")
        else:
            cells_shown = len(h3_data) if 'h3_data' in locals() else 0
            st.metric("H3 Cells Shown", f"{cells_shown:,}")
    
    with col3:
        if st.session_state.zoom_level < 8:
            total_stores = len(filtered_stores) if not filtered_stores.empty else 0
            st.metric("Total Stores", f"{total_stores:,}")
        else:
            avg_potential = h3_data['annual_potential'].mean() if 'h3_data' in locals() and not h3_data.empty else 0
            st.metric("Avg Potential", f"${avg_potential/1e6:.1f}M")
    
    with col4:
        st.metric("Zoom Level", st.session_state.zoom_level, current_zoom_desc)
    
    # Instructions
    st.info("""
    **🗺️ How to use this map:**
    • **Zoom out** (scroll up) - See store locations nationwide with state-level opportunity scores
    • **Zoom in** (scroll down) - H3 hexagons and drive-time isochrones appear at city level
    • **Click & drag** - Pan around the map
    • **Hover** - See details for any store or opportunity cell
    • **Filters** - Use the sidebar to toggle layers and adjust thresholds
    """)
    
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
    
    if not h3_data.empty:
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
                f"{top_sites.iloc[0]['nearest_grocery_distance']:.1f} mi",
                f"{top_sites.iloc[0]['convenience_score']:.1f}"
            ],
            'Site 2': [
                f"{top_sites.iloc[1]['opportunity_score']:.1f}",
                f"${top_sites.iloc[1]['annual_potential']/1e6:.1f}M",
                f"{top_sites.iloc[1]['population']:,}",
                f"${top_sites.iloc[1]['median_income']:,}",
                f"{top_sites.iloc[1]['nearest_grocery_distance']:.1f} mi",
                f"{top_sites.iloc[1]['convenience_score']:.1f}"
            ],
            'Site 3': [
                f"{top_sites.iloc[2]['opportunity_score']:.1f}",
                f"${top_sites.iloc[2]['annual_potential']/1e6:.1f}M",
                f"{top_sites.iloc[2]['population']:,}",
                f"${top_sites.iloc[2]['median_income']:,}",
                f"{top_sites.iloc[2]['nearest_grocery_distance']:.1f} mi",
                f"{top_sites.iloc[2]['convenience_score']:.1f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Radar chart for comparison
        categories = ['Opportunity', 'Population', 'Income', 'Convenience', 'Market Gap']
        
        fig = go.Figure()
        
        for i in range(min(3, len(top_sites))):
            fig.add_trace(go.Scatterpolar(
                r=[
                    top_sites.iloc[i]['opportunity_score'],
                    top_sites.iloc[i]['population'] / 1000,
                    top_sites.iloc[i]['median_income'] / 1000,
                    top_sites.iloc[i]['convenience_score'],
                    100 - top_sites.iloc[i].get('grocery_density', 5) * 10
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
            primary_html = f"""
            <div class="metric-card">
                <h4 style="color: #FF3621;">Primary Opportunity</h4>
                <p><b>Site 1</b> offers the highest opportunity score with strong demographics
                and minimal competition. The area shows high income levels with limited
                grocery access, creating ideal conditions for market entry.</p>
            </div>
            """
            st.markdown(primary_html, unsafe_allow_html=True)
        
        with col2:
            strategy_html = f"""
            <div class="metric-card">
                <h4 style="color: #00A972;">Expansion Strategy</h4>
                <p>Based on your {st.session_state.penetration} penetration scenario,
                focusing on the top 3 sites could generate ${(top_sites.head(3)['annual_potential'].sum() / 1e6):.1f}M
                in annual revenue potential.</p>
            </div>
            """
            st.markdown(strategy_html, unsafe_allow_html=True)
    else:
        st.info("No data available for analysis. Please check your database connection.")
    
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