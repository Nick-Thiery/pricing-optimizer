# =============================================================================
# PRICING OPTIMIZATION SYSTEM — Version 3 (Streamlit Web App)
# Improvements over V2:
#   1. Full web interface — no terminal needed to operate
#   2. Upload any CSV file via drag and drop
#   3. Dropdown to select product
#   4. Slider to adjust price range interactively
#   5. Live charts that update instantly
#   6. Summary table for all products at the bottom
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =============================================================================
# SECTION 1: PAGE CONFIGURATION
# This sets up the browser tab title, icon, and layout.
# This MUST be the first Streamlit command in the script.
# =============================================================================

st.set_page_config(
    page_title="Pricing Optimiser",
    page_icon="💰",
    layout="wide"   # "wide" uses the full browser width
)

# =============================================================================
# SECTION 2: HEADER
# st.title(), st.markdown() etc. work like print() but output to the web page.
# =============================================================================

st.title("Pricing Optimisation System - Thiery Advising Group")
st.markdown("**For small food & beverage businesses in Singapore**")
st.markdown("Upload your sales data, select a product, and get an instant pricing recommendation.")
st.divider()

# =============================================================================
# SECTION 3: FILE UPLOAD
# st.file_uploader() creates a drag-and-drop upload box in the browser.
# It returns None if no file has been uploaded yet.
# =============================================================================

uploaded_file = st.file_uploader(
    "📂 Upload your sales CSV file",
    type=["csv"],
    help="CSV must have columns: date, product_name, price, quantity_sold, cost_per_unit"
)

# If no file uploaded yet, show instructions and stop
# st.stop() is like an early return — nothing below it runs
if uploaded_file is None:
    st.info("👆 Upload a CSV file to get started. Use the sample file from Version 2 (sales_data_v2.csv) to test.")

    # Show expected format so the user knows what to upload
    st.markdown("#### Expected CSV format:")
    sample = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "product_name": ["Teh Tarik", "Teh Tarik"],
        "price": [2.00, 2.50],
        "quantity_sold": [120, 95],
        "cost_per_unit": [0.80, 0.80]
    })
    st.dataframe(sample, hide_index=True)
    st.stop()

# =============================================================================
# SECTION 4: LOAD AND VALIDATE DATA
# =============================================================================

# Read the uploaded file into a DataFrame
# In V2 we used pd.read_csv("filename.csv") — here we pass the uploaded file object directly
df = pd.read_csv(uploaded_file)

# Check that all required columns exist
required_columns = ["product_name", "price", "quantity_sold", "cost_per_unit"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"❌ Your CSV is missing these columns: {missing}")
    st.stop()

# Show a small success message and preview of the data
st.success(f"✅ File loaded — {len(df)} rows, {df['product_name'].nunique()} products found")

with st.expander("👁 Preview raw data"):
    st.dataframe(df, hide_index=True)

st.divider()

# =============================================================================
# SECTION 5: SIDEBAR CONTROLS
# The sidebar is the panel on the left side of a Streamlit app.
# We put controls there so they don't clutter the main page.
# =============================================================================

st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Adjust these to change the analysis.")

# Dropdown to pick a product
# st.sidebar.selectbox() creates a dropdown menu
products = sorted(df["product_name"].unique())
selected_product = st.sidebar.selectbox("Select product to analyse:", products)

# Slider for price range
# st.sidebar.slider() creates a range slider with two handles
price_min, price_max = st.sidebar.slider(
    "Price range to test (SGD):",
    min_value=0.10,
    max_value=15.00,
    value=(0.50, 7.00),   # default values (tuple = range slider)
    step=0.10,
    format="SGD %.2f"
)

st.sidebar.divider()
st.sidebar.markdown("**How to use:**")
st.sidebar.markdown("1. Upload your CSV above")
st.sidebar.markdown("2. Select a product")
st.sidebar.markdown("3. Adjust price range if needed")
st.sidebar.markdown("4. Scroll down for full summary")

# =============================================================================
# SECTION 6: CORE ANALYSIS FUNCTION
# Same logic as V2 but returns data instead of printing it,
# because in Streamlit we display with st. commands, not print()
# =============================================================================

def analyse_product(product_name, product_df, price_min, price_max):
    """
    Fits demand model, calculates revenue and profit curves.
    Returns a dictionary of results and arrays for plotting.
    """
    cost = product_df["cost_per_unit"].mean()

    # Fit linear demand model
    X = product_df["price"].values.reshape(-1, 1)
    y = product_df["quantity_sold"].values
    model = LinearRegression()
    model.fit(X, y)

    a = model.intercept_
    b = -model.coef_[0]
    r2 = model.score(X, y)

    # Scan price range
    price_range = np.arange(price_min, price_max + 0.10, 0.10)
    predicted_qty = np.clip(model.predict(price_range.reshape(-1, 1)), 0, None)
    predicted_revenue = price_range * predicted_qty
    predicted_profit = (price_range - cost) * predicted_qty

    # Find optima
    best_revenue_idx = np.argmax(predicted_revenue)
    best_profit_idx = np.argmax(predicted_profit)

    optimal_price_revenue = price_range[best_revenue_idx]
    optimal_revenue = predicted_revenue[best_revenue_idx]
    optimal_price_profit = price_range[best_profit_idx]
    optimal_profit = predicted_profit[best_profit_idx]

    avg_price = product_df["price"].mean()
    avg_qty = max(0, a - b * avg_price)
    current_profit = (avg_price - cost) * avg_qty
    profit_gain = optimal_profit - current_profit

    return {
        # Model info
        "a": a, "b": b, "r2": r2, "cost": cost,
        # Price arrays for plotting
        "price_range": price_range,
        "predicted_qty": predicted_qty,
        "predicted_revenue": predicted_revenue,
        "predicted_profit": predicted_profit,
        # Key results
        "optimal_price_revenue": optimal_price_revenue,
        "optimal_revenue": optimal_revenue,
        "optimal_price_profit": optimal_price_profit,
        "optimal_profit": optimal_profit,
        "avg_price": avg_price,
        "current_profit": current_profit,
        "profit_gain": profit_gain,
    }

# =============================================================================
# SECTION 7: SINGLE PRODUCT ANALYSIS (main page)
# =============================================================================

product_df = df[df["product_name"] == selected_product].copy()
r = analyse_product(selected_product, product_df, price_min, price_max)

st.subheader(f"📊 Analysis: {selected_product}")

# --- Key metrics in a row of cards ---
# st.columns() divides the page into side-by-side sections
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Optimal Price (Profit)",
    value=f"SGD {r['optimal_price_profit']:.2f}",
    delta=f"{r['optimal_price_profit'] - r['avg_price']:+.2f} vs current avg"
)
col2.metric(
    label="Max Profit / Period",
    value=f"SGD {r['optimal_profit']:.2f}"
)
col3.metric(
    label="Profit Gain vs Current",
    value=f"SGD {r['profit_gain']:.2f}"
)
col4.metric(
    label="Model Quality (R²)",
    value=f"{r['r2']:.3f}",
    help="1.0 = perfect fit. Below 0.5 = weak model, collect more data."
)

# Warning if model fit is weak
if r["r2"] < 0.5:
    st.warning(f"⚠️ R² = {r['r2']:.2f} — the model fit is weak. Try to collect sales data at more varied price points.")

# --- Charts ---
# In Streamlit, we create matplotlib figures and pass them to st.pyplot()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Pricing Analysis — {selected_product}", fontsize=14, fontweight="bold")

# Left: Demand curve
ax1.scatter(product_df["price"], product_df["quantity_sold"],
            color="steelblue", s=60, zorder=5, label="Actual sales data")
ax1.plot(r["price_range"], r["predicted_qty"],
         color="orange", linewidth=2,
         label=f"Q = {r['a']:.1f} - {r['b']:.1f}P")
ax1.axvline(r["optimal_price_profit"], color="red", linestyle="--", linewidth=1.5,
            label=f"Profit optimum: SGD {r['optimal_price_profit']:.2f}")
ax1.set_xlabel("Price (SGD)", fontsize=11)
ax1.set_ylabel("Quantity Sold", fontsize=11)
ax1.set_title("Demand Curve", fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Revenue + Profit
ax2.plot(r["price_range"], r["predicted_revenue"],
         color="steelblue", linewidth=2, label="Predicted Revenue")
ax2.plot(r["price_range"], r["predicted_profit"],
         color="steelblue", linewidth=2, linestyle="--", label="Predicted Profit")
ax2.scatter([r["optimal_price_revenue"]], [r["optimal_revenue"]],
            color="orange", s=100, zorder=5,
            label=f"Max Revenue @ SGD {r['optimal_price_revenue']:.2f}")
ax2.scatter([r["optimal_price_profit"]], [r["optimal_profit"]],
            color="red", s=150, marker="*", zorder=5,
            label=f"Max Profit @ SGD {r['optimal_price_profit']:.2f}")
ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
ax2.set_xlabel("Price (SGD)", fontsize=11)
ax2.set_ylabel("SGD", fontsize=11)
ax2.set_title("Revenue & Profit Curves", fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)      # This is how you show a matplotlib chart in Streamlit
plt.close()         # Always close the figure after showing to free memory

# --- Demand model details ---
with st.expander("📐 View demand model details"):
    st.markdown(f"""
    - **Demand equation:** Q = {r['a']:.2f} − {r['b']:.2f} × P
    - **Interpretation:** For every SGD 1.00 price increase, ~{r['b']:.0f} fewer units sold
    - **Cost per unit:** SGD {r['cost']:.2f}
    - **Revenue optimum:** SGD {r['optimal_price_revenue']:.2f} → Revenue SGD {r['optimal_revenue']:.2f}
    - **Profit optimum:** SGD {r['optimal_price_profit']:.2f} → Profit SGD {r['optimal_profit']:.2f}
    - **R² score:** {r['r2']:.3f}
    """)

st.divider()

# =============================================================================
# SECTION 8: SUMMARY TABLE — ALL PRODUCTS
# =============================================================================

st.subheader("📋 Summary — All Products")

summary_rows = []
for product in products:
    pdf = df[df["product_name"] == product].copy()
    res = analyse_product(product, pdf, price_min, price_max)
    summary_rows.append({
        "Product": product,
        "Cost/Unit (SGD)": round(res["cost"], 2),
        "Current Avg Price (SGD)": round(res["avg_price"], 2),
        "Current Profit/Period (SGD)": round(res["current_profit"], 2),
        "Optimal Price — Revenue (SGD)": round(res["optimal_price_revenue"], 2),
        "Optimal Price — Profit (SGD)": round(res["optimal_price_profit"], 2),
        "Max Profit/Period (SGD)": round(res["optimal_profit"], 2),
        "Potential Profit Gain (SGD)": round(res["profit_gain"], 2),
        "R²": round(res["r2"], 3),
    })

summary_df = pd.DataFrame(summary_rows)

# Highlight the profit gain column — positive = green, negative = red
st.dataframe(
    summary_df.style.background_gradient(
        subset=["Potential Profit Gain (SGD)"],
        cmap="RdYlGn"   # red = bad, yellow = neutral, green = good
    ),
    hide_index=True,
    use_container_width=True
)

st.divider()
st.caption("⚠️ This tool optimises based on a linear demand model (Q = a − bP). "
           "Results are estimates only. Always validate recommendations with real-world testing. "
           "Version 3 — Pricing Optimisation System for Singapore F&B.")
