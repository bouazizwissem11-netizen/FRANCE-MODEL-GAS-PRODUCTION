import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(page_title="France Model - Gas Production", layout="wide")

st.title("🌱 France Model - Gas Production Analysis")
st.markdown("### Recorded vs Fitted on the same graph")
st.markdown("### ✅ Supports commas (,) as decimal separator")

# ============================================
# FRANCE MODEL
# ============================================

def france_model(t, A, k, L):
    """France exponential model with lag"""
    t_adj = np.maximum(t - L, 0)
    return A * (1 - np.exp(-k * t_adj))

def fit_france(temps, pg):
    """Fit France model to data"""
    mask = temps > 0.1
    t = np.array(temps[mask])
    y = np.array(pg[mask])
    
    if len(t) < 4:
        return [np.nan, np.nan, np.nan], 0
    
    p0 = [max(y) * 1.2, 0.05, 0.5]
    bounds = ([0, 0, 0], [np.inf, np.inf, max(t)])
    
    try:
        popt, _ = curve_fit(france_model, t, y, p0=p0, maxfev=5000, bounds=bounds)
        y_pred = france_model(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return popt, r2
    except:
        return [np.nan, np.nan, np.nan], 0

# ============================================
# HELPER FUNCTION: Parse with comma
# ============================================

def parse_number(text):
    """Convert a string with comma as decimal separator to float"""
    text = text.strip()
    # Replace comma with dot
    text = text.replace(',', '.')
    try:
        return float(text)
    except:
        return np.nan

# ============================================
# DATA INPUT
# ============================================

st.subheader("1. Enter your PG data (ml/g MS)")

col1, col2 = st.columns(2)

with col1:
    st.write("**Times (h)** - one per line")
    times_input = st.text_area("Times", "0\n2\n4\n6\n8\n10\n12\n24\n48\n72", height=150)

with col2:
    st.write("**PG recorded (ml/g MS)** - one per line (supports commas)")
    pg_input = st.text_area("PG recorded", "0\n8,3\n17,6\n27,8\n38,1\n45,2\n56,5\n84,7\n98,1\n102,6", height=150)

# ============================================
# PROCESS BUTTON
# ============================================

if st.button("🚀 Draw Graph (Recorded vs Fitted)", type="primary"):
    
    try:
        # Parse times
        times = np.array([parse_number(x) for x in times_input.strip().split('\n') if x.strip()])
        times = times[~np.isnan(times)]
        
        # Parse PG recorded
        pg_recorded = np.array([parse_number(x) for x in pg_input.strip().split('\n') if x.strip()])
        pg_recorded = pg_recorded[~np.isnan(pg_recorded)]
        
        if len(times) != len(pg_recorded):
            st.error(f"❌ Number of time points ({len(times)}) and PG values ({len(pg_recorded)}) must match.")
            st.stop()
        
        if len(times) < 5:  # Need at least 5 points (including 0h)
            st.error("❌ Need at least 5 data points.")
            st.stop()
        
        # Remove 0h for fitting
        mask = times > 0.1
        times_filtered = times[mask]
        pg_filtered = pg_recorded[mask]
        
        if len(times_filtered) < 4:
            st.error("❌ Need at least 4 data points after removing 0h.")
            st.stop()
        
        st.success(f"✅ Data loaded: {len(times)} points (removed 0h, {len(times_filtered)} points used for fitting)")
        
        # Show a sample of parsed data
        with st.expander("📋 Preview parsed data"):
            preview = pd.DataFrame({
                "Time (h)": times,
                "PG recorded (ml/g MS)": pg_recorded
            })
            st.dataframe(preview, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error parsing data: {e}")
        st.stop()
    
    # Fit France model
    with st.spinner("Fitting France model..."):
        params, r2 = fit_france(times_filtered, pg_filtered)
    
    if np.isnan(params[0]):
        st.error("❌ France model fitting failed. Check your data.")
        st.stop()
    
    A, k, L = params
    
    # Calculate G24
    G24 = france_model(24, A, k, L)
    
    # ============================================
    # PLOT - Recorded vs Fitted on the same graph
    # ============================================
    
    st.subheader("2. Graph - Recorded vs Fitted")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot recorded data (blue circles)
    ax.plot(times, pg_recorded, 'o', color='#2F5597', markersize=8, label='Recorded')
    
    # Plot fitted curve (red line)
    t_fit = np.linspace(0, max(times), 200)
    pg_fit = france_model(t_fit, A, k, L)
    ax.plot(t_fit, pg_fit, '-', color='#C0504D', linewidth=2.5, label=f'Fitted (R²={r2:.4f})')
    
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('PG (ml/g MS)')
    ax.set_title('Gas Production - France Model (Recorded vs Fitted)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # ============================================
    # Display Parameters
    # ============================================
    
    st.subheader("3. Model Parameters")
    
    params_data = {
        "Parameter": ["A (ml/g MS)", "k (per h)", "Lag time (h)", "G24 (ml/g MS)", "R²"],
        "Value": [f"{A:.1f}", f"{k:.3f}", f"{L:.2f}", f"{G24:.1f}", f"{r2:.4f}"]
    }
    df_params = pd.DataFrame(params_data)
    st.dataframe(df_params, use_container_width=True)
    
    # ============================================
    # EXPORT
    # ============================================
    
    st.subheader("4. Export Graph")
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label="📥 Download graph (PNG)",
        data=buf,
        file_name="recorded_vs_fitted.png",
        mime="image/png"
    )