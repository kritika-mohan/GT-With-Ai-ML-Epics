"""
AI-ML Assisted Game-Theoretic Framework for Sustainable Energy Optimization
Modern Dark Luxury Amber UI — Inspired by high-converting design systems.
Clean, intuitive, and beginner-friendly interface.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EnergyAI — Sustainable Energy Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# CUSTOM LUXURY DARK AMBER CSS (Inspired by Velocity Design System)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Theme Overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0A0A0C !important;
        color: #F3F4F6 !important;
    }

    /* Ambient Hero Glow Header */
    .hero-container {
        position: relative;
        text-align: center;
        padding: 50px 20px 40px 20px;
        background: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(255, 107, 0, 0.28), rgba(10, 10, 12, 0) 75%);
        border-bottom: 1px solid rgba(255, 107, 0, 0.15);
        margin-bottom: 35px;
        border-radius: 0 0 32px 32px;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        background: rgba(255, 107, 0, 0.12);
        border: 1px solid rgba(255, 107, 0, 0.4);
        border-radius: 9999px;
        color: #FFA559;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 18px;
        box-shadow: 0 0 20px rgba(255, 107, 0, 0.2);
    }

    .hero-headline {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        background: linear-gradient(180deg, #FFFFFF 30%, #A1A1AA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 16px;
    }

    .hero-subheadline {
        font-size: 1.15rem;
        color: #9CA3AF;
        max-width: 720px;
        margin: 0 auto 24px auto;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Feature Glass Card */
    .glass-card {
        background: linear-gradient(135deg, rgba(24, 24, 30, 0.75) 0%, rgba(15, 15, 20, 0.9) 100%);
        border: 1px solid rgba(255, 115, 0, 0.18);
        border-radius: 20px;
        padding: 24px;
        backdrop-filter: blur(16px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        border-color: rgba(255, 120, 0, 0.45);
        box-shadow: 0 8px 30px rgba(255, 107, 0, 0.15);
        transform: translateY(-2px);
    }

    .card-icon-halo {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: radial-gradient(circle, rgba(255, 107, 0, 0.35) 0%, rgba(255, 107, 0, 0.05) 100%);
        border: 1px solid rgba(255, 107, 0, 0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin-bottom: 16px;
        box-shadow: 0 0 16px rgba(255, 107, 0, 0.25);
    }

    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 8px;
    }

    .card-desc {
        font-size: 0.92rem;
        color: #94A3B8;
        line-height: 1.55;
    }

    /* Glowing Metric Cards */
    .kpi-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 22px 16px;
        background: linear-gradient(180deg, rgba(30, 24, 20, 0.6) 0%, rgba(16, 16, 22, 0.8) 100%);
        border: 1px solid rgba(255, 107, 0, 0.25);
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .kpi-val {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(180deg, #FFFFFF 40%, #FFA559 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
    }

    .kpi-tag {
        display: inline-block;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 9999px;
        background: rgba(255, 107, 0, 0.15);
        color: #FF8A3D;
        font-weight: 600;
        margin-top: 6px;
        border: 1px solid rgba(255, 107, 0, 0.3);
    }

    /* Primary Glowing Button */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B00 0%, #EA580C 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        border-radius: 9999px !important;
        padding: 12px 32px !important;
        border: none !important;
        box-shadow: 0 0 25px rgba(255, 107, 0, 0.45) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        letter-spacing: 0.02em !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 35px rgba(255, 107, 0, 0.7) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(18, 18, 24, 0.8);
        padding: 6px;
        border-radius: 9999px;
        border: 1px solid rgba(255, 107, 0, 0.15);
        margin-bottom: 28px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 9999px;
        color: #94A3B8;
        font-size: 0.92rem;
        font-weight: 600;
        padding: 0 24px;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 107, 0, 0.25), rgba(255, 107, 0, 0.08)) !important;
        color: #FFA559 !important;
        border: 1px solid rgba(255, 107, 0, 0.4) !important;
    }

    /* FAQ Box */
    .faq-item {
        background: rgba(18, 18, 24, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }
    .faq-q {
        font-weight: 700;
        color: #FFFFFF;
        font-size: 1rem;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .faq-a {
        color: #94A3B8;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    /* Bottom Ambient Bar */
    .bottom-bar {
        text-align: center;
        padding: 30px 20px;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 107, 0, 0.15);
        background: radial-gradient(ellipse 60% 80% at 50% 120%, rgba(255, 107, 0, 0.22), rgba(10, 10, 12, 0) 70%);
        border-radius: 32px 32px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SAFE DATA & MODEL LOADING
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
DATA_DIR = BASE_DIR / 'data' / 'processed'

@st.cache_resource
def load_champion_model():
    model_path = MODELS_DIR / 'best_model.pkl'
    meta_path = MODELS_DIR / 'model_metadata.json'
    if not model_path.exists():
        return None, None
    try:
        model = joblib.load(model_path)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        return model, meta
    except Exception as e:
        return None, None

@st.cache_data
def load_all_forecasts():
    csv_path = RESULTS_DIR / 'ml_forecast_for_game_theory.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df

@st.cache_data
def load_benchmarks():
    csv_path = RESULTS_DIR / 'model_comparison.csv'
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)

model, metadata = load_champion_model()
df_forecast = load_all_forecasts()
df_bench = load_benchmarks()

# -----------------------------------------------------------------------------
# HERO BANNER (Velocity-style glowing ambient dark amber)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="badge-pill">⚡ AI-ML & Game-Theoretic Framework</div>
    <h1 class="hero-headline">Predict Faster. Optimize Cleaner.</h1>
    <p class="hero-subheadline">
        An AI-powered energy intelligence system that forecasts grid electricity demand, tracks solar & wind availability, and uses Stackelberg Game Theory to lower electricity costs and carbon emissions.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HIGH-LEVEL VALUE METRICS (Simple and intuitive)
# -----------------------------------------------------------------------------
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown("""
    <div class="kpi-container">
        <div class="kpi-val">98.9%</div>
        <div class="kpi-label">Forecast Accuracy</div>
        <div class="kpi-tag">1.06% MAPE on National Grid</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown("""
    <div class="kpi-container">
        <div class="kpi-val">12,984</div>
        <div class="kpi-label">Hourly Grid Records</div>
        <div class="kpi-tag">Grid-India Open Energy Data</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown("""
    <div class="kpi-container">
        <div class="kpi-val">194 GW</div>
        <div class="kpi-label">Average Demand</div>
        <div class="kpi-tag">Peaks up to 248.7 GW</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown("""
    <div class="kpi-container">
        <div class="kpi-val">64.7 GW</div>
        <div class="kpi-label">Peak Solar Generation</div>
        <div class="kpi-tag">Afternoon Clean Power</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TABBED NAVIGATION (Clean and simple)
# -----------------------------------------------------------------------------
tab_pred, tab_charts, tab_bench, tab_game, tab_faq = st.tabs([
    "⚡ 1. Live Calculator (Test Model)",
    "📈 2. Visual Solar & Demand Charts",
    "🏆 3. Model Accuracy & Benchmark",
    "🎮 4. Game Theory Simulator",
    "❓ 5. Simple Explanations & FAQ"
])

# =============================================================================
# TAB 1: LIVE ML PREDICTION ENGINE (Simple, Beginner-Friendly Inputs)
# =============================================================================
with tab_pred:
    st.markdown("### ⚡ Live Electricity Demand Predictor")
    st.markdown("Select a time of day and season. The trained **XGBoost AI Model** will instantly predict how much electricity the country will consume and what the optimal electricity price tier should be.")

    col_in_a, col_in_b = st.columns([1, 1], gap="medium")

    with col_in_a:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon-halo">🕐</div>
            <div class="card-title">1. Time & Season Settings</div>
            <div class="card-desc">Choose the hour of the day and month you want to forecast.</div>
        </div>
        """, unsafe_allow_html=True)
        
        sim_hour = st.slider("Hour of Day (24-Hour Clock)", 0, 23, 14, help="0 is midnight, 12 is noon, 19 is 7 PM.")
        hour_labels = {
            sim_hour: f"{sim_hour}:00 ({'Morning Peak' if 7<=sim_hour<=10 else 'Solar Afternoon' if 11<=sim_hour<=16 else 'Evening Peak' if 18<=sim_hour<=22 else 'Off-Peak Night'})"
        }
        st.caption(f"Selected: **{hour_labels[sim_hour]}**")

        sim_month = st.selectbox(
            "Month of the Year",
            options=list(range(1, 13)),
            format_func=lambda m: ['January (Winter)', 'February', 'March (Spring)', 'April (Summer begins)', 'May (High Peak Cooling)', 'June (Monsoon/Summer)', 'July (Monsoon Wind)', 'August', 'September', 'October', 'November', 'December (Winter)'][m-1],
            index=4
        )

        sim_day_type = st.radio("Day Type", ["Weekday (Workday)", "Weekend (Lower Industrial Demand)"], horizontal=True)
        is_weekend = 1 if "Weekend" in sim_day_type else 0

    with col_in_b:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon-halo">⚡</div>
            <div class="card-title">2. Recent Grid Conditions</div>
            <div class="card-desc">Power usage follows strong habits. Adjust previous hour demand.</div>
        </div>
        """, unsafe_allow_html=True)

        recent_demand_gw = st.slider("Demand 1-Hour Ago (Gigawatts — GW)", 100.0, 250.0, 196.0, step=1.0, help="India's typical grid demand ranges between 120 GW and 240 GW.")
        recent_demand_mw = recent_demand_gw * 1000.0

        sim_solar_gw = st.slider("Expected Solar Power (GW)", 0.0, 65.0, 42.0 if 8 <= sim_hour <= 17 else 0.0, step=1.0)
        sim_solar_mw = sim_solar_gw * 1000.0

        sim_wind_gw = st.slider("Expected Wind Power (GW)", 0.0, 30.0, 8.5, step=0.5)
        sim_wind_mw = sim_wind_gw * 1000.0

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Calculate AI Prediction Now", use_container_width=True)

    if predict_clicked or True:  # Default display on load
        if model is not None and metadata is not None:
            # Build input vector matching trained schema
            hour_sin = np.sin(2 * np.pi * sim_hour / 24)
            hour_cos = np.cos(2 * np.pi * sim_hour / 24)
            dow = 6 if is_weekend else 2
            dow_sin = np.sin(2 * np.pi * dow / 7)
            dow_cos = np.cos(2 * np.pi * dow / 7)
            month_sin = np.sin(2 * np.pi * sim_month / 12)
            month_cos = np.cos(2 * np.pi * sim_month / 12)

            input_dict = {
                'hour': sim_hour,
                'day': 15,
                'day_of_week': dow,
                'month': sim_month,
                'year': 2025,
                'quarter': (sim_month - 1) // 3 + 1,
                'day_of_year': sim_month * 30 + 15,
                'is_weekend': is_weekend,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'dow_sin': dow_sin,
                'dow_cos': dow_cos,
                'month_sin': month_sin,
                'month_cos': month_cos,
                'demand_lag_1h': recent_demand_mw,
                'demand_lag_24h': recent_demand_mw * 0.98,
                'demand_lag_48h': recent_demand_mw * 0.97,
                'demand_lag_168h': recent_demand_mw * 0.96,
                'solar_lag_24h': sim_solar_mw,
                'wind_lag_24h': sim_wind_mw,
                'demand_rolling_mean_6h': recent_demand_mw,
                'demand_rolling_mean_24h': recent_demand_mw * 0.98,
                'demand_rolling_std_24h': 7500.0,
                'demand_rolling_max_24h': recent_demand_mw * 1.08,
                'demand_rolling_min_24h': recent_demand_mw * 0.88
            }

            pred_input = pd.DataFrame([input_dict])[metadata['features']]
            predicted_mw = float(model.predict(pred_input)[0])
            total_clean_mw = sim_solar_mw + sim_wind_mw
            net_load_mw = max(0.0, predicted_mw - total_clean_mw)
            clean_share_pct = min(100.0, (total_clean_mw / predicted_mw * 100)) if predicted_mw > 0 else 0.0

            # Time of use tier
            if (7 <= sim_hour <= 10) or (18 <= sim_hour <= 22):
                tier_badge = "🔴 PEAK HOURS (Higher Price to Save Grid)"
                price_rec = "₹7.50 / kWh"
            elif 11 <= sim_hour <= 16:
                tier_badge = "🟡 SOLAR VALLEY (Cheapest Clean Power)"
                price_rec = "₹3.80 / kWh"
            else:
                tier_badge = "🟢 OFF-PEAK (Overnight Standard)"
                price_rec = "₹4.50 / kWh"

            st.markdown("#### 🎯 AI Forecast Results")
            out_c1, out_c2, out_c3, out_c4 = st.columns(4)

            with out_c1:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-val">{predicted_mw/1000:.2f} GW</div>
                    <div class="kpi-label">Predicted Demand</div>
                    <div class="kpi-tag">{predicted_mw:,.0f} MW</div>
                </div>
                """, unsafe_allow_html=True)

            with out_c2:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-val">{net_load_mw/1000:.2f} GW</div>
                    <div class="kpi-label">Net Load Needed</div>
                    <div class="kpi-tag">Remaining Conventional Power</div>
                </div>
                """, unsafe_allow_html=True)

            with out_c3:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-val">{clean_share_pct:.1f}%</div>
                    <div class="kpi-label">Clean Energy Share</div>
                    <div class="kpi-tag">{total_clean_mw/1000:.1f} GW Clean Power</div>
                </div>
                """, unsafe_allow_html=True)

            with out_c4:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-val" style="font-size:1.6rem;color:#FFA559;">{price_rec}</div>
                    <div class="kpi-label">Recommended Tariff</div>
                    <div class="kpi-tag">{tier_badge}</div>
                </div>
                """, unsafe_allow_html=True)

            # Visual breakdown bar
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name='Conventional Power (Coal/Gas/Hydro)',
                x=['Power Supply Mix'],
                y=[net_load_mw / 1000],
                marker_color='#F43F5E'
            ))
            fig_bar.add_trace(go.Bar(
                name='Solar Power',
                x=['Power Supply Mix'],
                y=[sim_solar_mw / 1000],
                marker_color='#FF8800'
            ))
            fig_bar.add_trace(go.Bar(
                name='Wind Power',
                x=['Power Supply Mix'],
                y=[sim_wind_mw / 1000],
                marker_color='#10B981'
            ))
            fig_bar.update_layout(
                barmode='stack',
                title="Visual Supply Allocation to Satisfy Predicted Demand (Gigawatts)",
                template='plotly_dark',
                paper_bgcolor='#0E0E12',
                plot_bgcolor='#0E0E12',
                height=280,
                yaxis=dict(title="Gigawatts (GW)", gridcolor="rgba(255,255,255,0.08)"),
                margin=dict(l=40, r=40, t=50, b=30)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# TAB 2: VISUAL SOLAR & DEMAND CHARTS (The Duck Curve Explained)
# =============================================================================
with tab_charts:
    st.markdown("### 📈 Real Power Grid Trends & The Famous 'Duck Curve'")
    st.markdown("Explore the genuine Grid-India data. Notice how massive solar generation at 12:00 PM creates the **Duck Curve** (a deep valley in net demand).")

    if df_forecast is not None:
        # Date filter
        sample_weeks = df_forecast.head(168 * 2)  # 2 weeks
        
        fig_duck = go.Figure()
        # Diurnal averages
        hourly_grp = df_forecast.groupby('hour').mean(numeric_only=True)
        
        fig_duck.add_trace(go.Scatter(
            x=hourly_grp.index, y=hourly_grp['actual_demand_mw'] / 1000,
            mode='lines+markers', name='Total Electricity Demand (GW)',
            line=dict(color='#38BDF8', width=3)
        ))
        fig_duck.add_trace(go.Scatter(
            x=hourly_grp.index, y=hourly_grp['net_load_mw'] / 1000,
            mode='lines+markers', name='Net Load (Demand Minus Solar & Wind) (GW)',
            line=dict(color='#FF6B00', width=3.5)
        ))
        fig_duck.add_trace(go.Bar(
            x=hourly_grp.index, y=hourly_grp['solar_mw'] / 1000,
            name='Solar Power (GW)', marker_color='rgba(255, 170, 0, 0.45)'
        ))
        fig_duck.add_trace(go.Bar(
            x=hourly_grp.index, y=hourly_grp['wind_mw'] / 1000,
            name='Wind Power (GW)', marker_color='rgba(16, 185, 129, 0.45)'
        ))

        fig_duck.update_layout(
            title="The 24-Hour National Duck Curve: Solar Depresses Afternoon Grid Load",
            xaxis=dict(title="Hour of Day (0 = Midnight, 12 = Noon, 23 = 11 PM)", tickmode='linear', tick0=0, dtick=2, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Gigawatts (GW)", gridcolor="rgba(255,255,255,0.06)"),
            template='plotly_dark',
            paper_bgcolor='#0E0E12',
            plot_bgcolor='#0E0E12',
            height=440,
            barmode='stack',
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_duck, use_container_width=True)

        st.info("💡 **What is happening in this chart?** Total electricity demand (blue line) stays relatively steady between 180 and 210 GW. But around 12:00 PM, **solar generation surges** (yellow bars). As a result, the net load that conventional plants must produce plummets into a deep valley (orange line), creating the belly of the 'duck'.")

        # Actual vs Predicted over 1 week
        st.markdown("#### 🔍 AI Predictions Tracking Actual Demand (1-Week Inspection)")
        one_week = df_forecast.head(168)
        fig_trace = go.Figure()
        fig_trace.add_trace(go.Scatter(
            x=one_week.index, y=one_week['demand_scenario_high'] / 1000,
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_trace.add_trace(go.Scatter(
            x=one_week.index, y=one_week['demand_scenario_low'] / 1000,
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(255, 107, 0, 0.15)', name='95% Confidence Band'
        ))
        fig_trace.add_trace(go.Scatter(
            x=one_week.index, y=one_week['actual_demand_mw'] / 1000,
            mode='lines', name='Actual Grid Demand (GW)', line=dict(color='#38BDF8', width=2)
        ))
        fig_trace.add_trace(go.Scatter(
            x=one_week.index, y=one_week['predicted_demand_mw'] / 1000,
            mode='lines', name='XGBoost Predicted Demand (GW)', line=dict(color='#FF6B00', width=2, dash='dash')
        ))
        fig_trace.update_layout(
            title="Hourly Accuracy: Actual Demand vs. XGBoost Forecast with Uncertainty Bounds",
            xaxis=dict(title="Date & Time", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Gigawatts (GW)", gridcolor="rgba(255,255,255,0.06)"),
            template='plotly_dark',
            paper_bgcolor='#0E0E12',
            plot_bgcolor='#0E0E12',
            height=380,
            hovermode='x unified'
        )
        st.plotly_chart(fig_trace, use_container_width=True)

# =============================================================================
# TAB 3: MODEL ACCURACY & BENCHMARKS (Plain and Justified)
# =============================================================================
with tab_bench:
    st.markdown("### 🏆 Scientific Benchmark: Which Model Works Best?")
    st.markdown("We trained 5 different algorithms chronologically on **10,252 hours** and tested them on **2,564 unseen hours**. Gradient Boosting algorithms beat all traditional baselines.")

    if df_bench is not None:
        c_acc1, c_acc2 = st.columns([1, 1], gap="medium")

        with c_acc1:
            fig_mape = px.bar(
                df_bench, x='Model', y='MAPE (%)',
                text_auto='.2f',
                title="Forecast Error Rate (Lower is Better)",
                color='MAPE (%)',
                color_continuous_scale=['#10B981', '#FF6B00', '#EF4444']
            )
            fig_mape.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0E0E12',
                plot_bgcolor='#0E0E12',
                yaxis=dict(title="Error Rate (%)", gridcolor="rgba(255,255,255,0.06)")
            )
            st.plotly_chart(fig_mape, use_container_width=True)

        with c_acc2:
            fig_r2 = px.bar(
                df_bench, x='Model', y='R2',
                text_auto='.3f',
                title="R² Score: Variance Explained (Higher is Better, Max 1.0)",
                color='R2',
                color_continuous_scale=['#EF4444', '#FF6B00', '#10B981']
            )
            fig_r2.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0E0E12',
                plot_bgcolor='#0E0E12',
                yaxis=dict(title="R² Score", range=[0.8, 1.0], gridcolor="rgba(255,255,255,0.06)")
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown("##### 📋 Exact Performance Metrics Table")
        st.dataframe(df_bench.style.format({
            'MAE (MW)': '{:,.1f}',
            'RMSE (MW)': '{:,.1f}',
            'MAPE (%)': '{:.2f}%',
            'R2': '{:.4f}'
        }), use_container_width=True)

        st.markdown("""
        <div class="faq-item">
            <div class="faq-q">🤔 Why does 1.06% MAPE matter in real life?</div>
            <div class="faq-a">
                India's electricity grid operates at ~200,000 MW. A 1% error means our prediction is only off by ~2,000 MW. This is close enough for power plant operators to reserve the exact right amount of backup power, preventing blackouts and avoiding millions of dollars in wasted coal/gas fuel.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 4: STACKELBERG GAME THEORY SIMULATOR
# =============================================================================
with tab_game:
    st.markdown("### 🎮 Stackelberg Leader-Follower Pricing Game")
    st.markdown("""
    In our research paper, the **Electricity Utility is the Leader** and **Consumers are Followers**:
    - When electricity is scarce (evening peak), the Leader sets higher prices to discourage heavy usage.
    - When free solar energy floods the grid (noon), the Leader offers discounted prices.
    - Consumers respond by scheduling EV chargers and washing machines to run at noon instead of 8 PM!
    """)

    col_g1, col_g2 = st.columns([1, 2], gap="medium")

    with col_g1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon-halo">⚙️</div>
            <div class="card-title">Adjust Pricing Strategy</div>
            <div class="card-desc">Simulate different electricity tariff rates:</div>
        </div>
        """, unsafe_allow_html=True)

        g_peak_price = st.slider("Peak Tariff (Evening 18:00–22:00)", 6.0, 14.0, 9.0, step=0.5, format="₹%0.1f/kWh")
        g_solar_price = st.slider("Solar Valley Tariff (Noon 11:00–16:00)", 1.5, 5.5, 3.2, step=0.25, format="₹%0.1f/kWh")
        g_offpeak_price = st.slider("Night Tariff (Overnight 23:00–06:00)", 2.0, 6.0, 4.0, step=0.25, format="₹%0.1f/kWh")
        g_elasticity = st.slider("Consumer Price Sensitivity (Elasticity ε)", 0.05, 0.45, 0.22, step=0.01, help="How strongly people react to price differences.")

    with col_g2:
        if df_forecast is not None:
            sim_48h = df_forecast.head(48).copy()
            rate_mapping = {'peak': g_peak_price, 'standard': g_solar_price, 'off_peak': g_offpeak_price}
            sim_48h['price'] = sim_48h['pricing_tier'].map(rate_mapping)

            # Follower response: shift demand proportional to price gradient
            avg_p = sim_48h['price'].mean()
            price_delta = (sim_48h['price'] - avg_p) / avg_p
            shift = 1.0 - (g_elasticity * price_delta)
            sim_48h['shifted_demand_gw'] = (sim_48h['predicted_demand_mw'] * shift) / 1000
            sim_48h['baseline_demand_gw'] = sim_48h['predicted_demand_mw'] / 1000

            peak_shaved_gw = sim_48h['baseline_demand_gw'].max() - sim_48h['shifted_demand_gw'].max()
            co2_saved_tons = peak_shaved_gw * 1000 * 0.85 * 24 / 1000  # Estimate

            st.markdown("#### 🎯 Game Equilibrium Results")
            eq_a, eq_b, eq_c = st.columns(3)
            with eq_a:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-val">{peak_shaved_gw:.2f} GW</div>
                    <div class="kpi-label">Peak Demand Shaved</div>
                    <div class="kpi-tag">-{(peak_shaved_gw/sim_48h['baseline_demand_gw'].max()*100):.1f}% Peak Stress</div>
                </div>
                """, unsafe_allow_html=True)
            with eq_b:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-val">₹{abs(g_peak_price - g_solar_price):.1f}</div>
                    <div class="kpi-label">Price Incentive Gap</div>
                    <div class="kpi-tag">Peak vs. Solar Discount</div>
                </div>
                """, unsafe_allow_html=True)
            with eq_c:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-val">{co2_saved_tons:,.0f} T</div>
                    <div class="kpi-label">Estimated CO₂ Avoided</div>
                    <div class="kpi-tag">Daily Emissions Saved</div>
                </div>
                """, unsafe_allow_html=True)

            # Chart showing shifted demand vs original
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(
                x=sim_48h.index, y=sim_48h['baseline_demand_gw'],
                mode='lines', name='Baseline Load (Without Game Pricing)',
                line=dict(color='#F43F5E', dash='dot', width=2)
            ))
            fig_sim.add_trace(go.Scatter(
                x=sim_48h.index, y=sim_48h['shifted_demand_gw'],
                mode='lines', name='Optimized Demand (After Consumer Reaction)',
                line=dict(color='#10B981', width=3)
            ))
            fig_sim.add_trace(go.Scatter(
                x=sim_48h.index, y=sim_48h['solar_mw'] / 1000,
                mode='lines', name='Available Solar Generation (GW)',
                fill='tozeroy', fillcolor='rgba(255, 170, 0, 0.15)',
                line=dict(color='#FF8800', width=1)
            ))
            fig_sim.update_layout(
                title="Stackelberg Game Optimization: Flattening Peaks & Soaking Up Clean Solar Power",
                xaxis=dict(title="Time", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Gigawatts (GW)", gridcolor="rgba(255,255,255,0.06)"),
                template='plotly_dark',
                paper_bgcolor='#0E0E12',
                plot_bgcolor='#0E0E12',
                height=340,
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_sim, use_container_width=True)

# =============================================================================
# TAB 5: SIMPLE EXPLANATIONS & FAQ (Velocity style clean FAQ)
# =============================================================================
with tab_faq:
    st.markdown("### ❓ Simple Explanations & FAQ")
    st.markdown("Plain-language answers to explain this university research project to professors, examiners, or viva questions.")

    st.markdown("""
    <div class="faq-item">
        <div class="faq-q">⚡ 1. What is Electricity Demand Forecasting?</div>
        <div class="faq-a">
            Electricity travels across power lines at the speed of light and cannot easily be stored. Power plants must produce the exact amount of power people need every second. Demand forecasting uses historical trends to predict tomorrow's power usage in advance, preventing blackouts and avoiding over-generation.
        </div>
    </div>

    <div class="faq-item">
        <div class="faq-q">☀️ 2. What is Net Load and the 'Duck Curve'?</div>
        <div class="faq-a">
            <strong>Net Load = Total Demand − Solar Generation − Wind Generation.</strong><br>
            When everyone is at work around noon, solar panels generate massive amounts of clean power. This reduces the amount of power that coal or gas power plants need to generate, causing a deep dip in demand. When the sun sets at 6 PM, solar output drops to zero while people turn on lights and ACs at home, causing a steep ramp up in net load.
        </div>
    </div>

    <div class="faq-item">
        <div class="faq-q">🤖 3. How does the Machine Learning model work?</div>
        <div class="faq-a">
            We use <strong>XGBoost (Extreme Gradient Boosting)</strong>. It builds hundreds of decision trees sequentially, where each tree learns from the mistakes of the previous one. The model looks at 25 features including what hour it is, the day of the week, the month, what the demand was yesterday at the same hour, and the 24-hour moving average.
        </div>
    </div>

    <div class="faq-item">
        <div class="faq-q">🎯 4. What is the Stackelberg Leader-Follower Game?</div>
        <div class="faq-a">
            It is a game-theoretic economic model with two players:<br>
            • <strong>Leader (The Power Company):</strong> Chooses electricity prices across the day to minimize generation costs and stabilize the grid.<br>
            • <strong>Follower (The Consumers):</strong> Sees the prices and decides when to use electricity (e.g. charging an electric vehicle or running washing machines when power is cheap).<br>
            Through this feedback loop, both sides reach an equilibrium that reduces carbon emissions and saves money.
        </div>
    </div>

    <div class="faq-item">
        <div class="faq-q">📁 5. Where did the dataset come from?</div>
        <div class="faq-a">
            The data is from <strong>Grid-India (Power System Operation Corporation Limited)</strong>, published openly on Mendeley Data. It contains real hourly electricity demand, solar generation, and wind generation across India from January 2024 to June 2025.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BOTTOM AMBIENT FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="bottom-bar">
    <div class="badge-pill">University Research Project</div>
    <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">
        AI-ML Assisted Game-Theoretic Framework for Sustainable Energy Optimization
    </div>
    <div style="font-size: 0.85rem; color: #94A3B8;">
        Empirical Community Case Study Using Government Open Energy Data | Grid-India Telemetry
    </div>
</div>
""", unsafe_allow_html=True)
