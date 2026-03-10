# =============================================================================
# PRICING OPTIMIZATION SYSTEM — Version 4
# Thiery Advising Group — Branded Edition
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Thiery Advising Group — Pricing Tool",
    page_icon="🌿",
    layout="wide"
)

# =============================================================================
# CUSTOM CSS — Green & white, clean, modern, friendly
# =============================================================================

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    /* Base */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide default Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp {
        background-color: #f7faf7;
    }

    /* Top branded header bar */
    .brand-header {
        background: white;
        padding: 20px 32px;
        border-radius: 16px;
        margin-bottom: 4px;
        border: 1px solid #e0efe6;
        box-shadow: 0 1px 8px rgba(0,0,0,0.05);
    }
    .brand-title {
        font-family: 'DM Serif Display', serif;
        font-size: 26px;
        color: #1a3d2b;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .brand-subtitle {
        font-size: 13px;
        color: #2d9e5f;
        margin-top: 3px;
        font-weight: 500;
    }
    .brand-contact {
        font-size: 13px;
        color: #888;
        text-align: right;
        line-height: 1.8;
    }
    .brand-contact a {
        color: #2d9e5f;
        text-decoration: none;
        font-weight: 600;
    }

    /* Tool description card */
    .intro-card {
        background: white;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        border-left: 4px solid #2d9e5f;
        box-shadow: 0 1px 8px rgba(0,0,0,0.06);
        font-size: 15px;
        color: #444;
        line-height: 1.6;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.06);
        border-top: 3px solid #2d9e5f;
    }
    [data-testid="metric-container"] label {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #888 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #1a6b3c !important;
    }

    /* Section headers */
    h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: #1a3d2b !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e8f5ee;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        font-weight: 600;
        color: #1a3d2b;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    /* Buttons */
    .stButton button {
        background-color: #2d9e5f;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stButton button:hover {
        background-color: #1a6b3c;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 8px;
        border: 2px dashed #b8e0c8;
    }

    /* Success/info/warning messages */
    .stSuccess {
        background-color: #edfaf3 !important;
        border-left-color: #2d9e5f !important;
        color: #1a6b3c !important;
    }

    /* Divider */
    hr {
        border-color: #e0efe6;
        margin: 24px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #aaa;
        font-size: 12px;
        margin-top: 40px;
        border-top: 1px solid #e8f5ee;
    }
    .footer a {
        color: #2d9e5f;
        text-decoration: none;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #1a3d2b !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# BRANDED HEADER
# =============================================================================

# Header with logo
col_logo, col_title, col_contact = st.columns([1, 4, 2])

with col_logo:
    try:
        st.image("tag_logo.png", width=180)
    except:
        st.markdown("<div style='font-size:40px; padding-top:8px;'>🌿</div>", unsafe_allow_html=True)

with col_title:
    st.markdown("""
    <div style='padding-top: 12px;'>
        <div style='font-family: DM Serif Display, serif; font-size: 26px;
        color: #1a3d2b; font-weight: 700; letter-spacing: -0.3px;'>
        Thiery Advising Group</div>
        <div style='font-size: 13px; color: #2d9e5f; margin-top: 4px; font-weight: 500;'>
        Pricing Optimisation Tool · Singapore F&B</div>
    </div>
    """, unsafe_allow_html=True)

with col_contact:
    st.markdown("""
    <div style='text-align: right; padding-top: 18px; font-size: 13px; color: #888;'>
        Questions? Get in touch<br>
        <a href='mailto:thieryadvisinggroup@gmail.com'
        style='color: #2d9e5f; font-weight: 600; text-decoration: none;'>
        thieryadvisinggroup@gmail.com</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color: #e0efe6; margin: 16px 0 20px 0;'>", unsafe_allow_html=True)

st.markdown("""
<div class="intro-card">
    Upload your sales data and instantly find the price that maximises your profit.
    Built on a linear demand model — simple, transparent, and explainable to any business owner.
</div>
""", unsafe_allow_html=True)

# =============================================================================
# FILE UPLOAD
# =============================================================================

uploaded_file = st.file_uploader(
    "📂 Upload your sales CSV file",
    type=["csv"],
    help="CSV must have columns: date, product_name, price, quantity_sold, cost_per_unit"
)

if uploaded_file is None:
    st.info("👆 Upload a CSV to get started. Don't have one? Use **sales_data_v2.csv** from the setup guide.")
    st.markdown("#### Expected format:")
    sample = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "product_name": ["Teh Tarik", "Teh Tarik"],
        "price": [2.00, 2.50],
        "quantity_sold": [120, 95],
        "cost_per_unit": [0.80, 0.80]
    })
    st.dataframe(sample, hide_index=True)
    st.stop()

df = pd.read_csv(uploaded_file)

required_columns = ["product_name", "price", "quantity_sold", "cost_per_unit"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

st.success(f"✅ {len(df)} rows loaded · {df['product_name'].nunique()} products found")

with st.expander("👁 Preview raw data"):
    st.dataframe(df, hide_index=True)

st.divider()

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("""
<div style='padding: 12px 0 4px 0;'>
    <div style='font-family: DM Serif Display, serif; font-size: 20px; color: #1a3d2b;'>⚙️ Settings</div>
    <div style='font-size: 12px; color: #888; margin-top: 2px;'>Adjust the analysis below</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

products = sorted(df["product_name"].unique())
selected_product = st.sidebar.selectbox("Select product:", products)

price_min, price_max = st.sidebar.slider(
    "Price range to test (SGD):",
    min_value=0.10,
    max_value=15.00,
    value=(0.50, 7.00),
    step=0.10,
    format="SGD %.2f"
)

st.sidebar.divider()

st.sidebar.markdown("""
<div style='font-size: 12px; color: #888; line-height: 1.8;'>
    <b style='color: #1a3d2b;'>How to use</b><br>
    1. Upload your CSV<br>
    2. Select a product<br>
    3. Adjust price range<br>
    4. Read the recommendation<br>
    5. Scroll for full summary
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("""
<div style='font-size: 11px; color: #bbb; text-align: center;'>
    Thiery Advising Group<br>
    <a href='mailto:thieryadvisinggroup@gmail.com' style='color: #2d9e5f;'>thieryadvisinggroup@gmail.com</a>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================

def analyse_product(product_name, product_df, price_min, price_max):
    cost = product_df["cost_per_unit"].mean()
    X = product_df["price"].values.reshape(-1, 1)
    y = product_df["quantity_sold"].values
    model = LinearRegression()
    model.fit(X, y)
    a = model.intercept_
    b = -model.coef_[0]
    r2 = model.score(X, y)

    price_range = np.arange(price_min, price_max + 0.10, 0.10)
    predicted_qty = np.clip(model.predict(price_range.reshape(-1, 1)), 0, None)
    predicted_revenue = price_range * predicted_qty
    predicted_profit = (price_range - cost) * predicted_qty

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
        "a": a, "b": b, "r2": r2, "cost": cost,
        "price_range": price_range,
        "predicted_qty": predicted_qty,
        "predicted_revenue": predicted_revenue,
        "predicted_profit": predicted_profit,
        "optimal_price_revenue": optimal_price_revenue,
        "optimal_revenue": optimal_revenue,
        "optimal_price_profit": optimal_price_profit,
        "optimal_profit": optimal_profit,
        "avg_price": avg_price,
        "current_profit": current_profit,
        "profit_gain": profit_gain,
    }

# =============================================================================
# SINGLE PRODUCT ANALYSIS
# =============================================================================

product_df = df[df["product_name"] == selected_product].copy()
r = analyse_product(selected_product, product_df, price_min, price_max)

st.subheader(f"📊 {selected_product}")

# Metric cards
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Recommended Price",
    f"SGD {r['optimal_price_profit']:.2f}",
    delta=f"{r['optimal_price_profit'] - r['avg_price']:+.2f} vs current"
)
col2.metric("Max Profit / Period", f"SGD {r['optimal_profit']:.2f}")
col3.metric("Potential Profit Gain", f"SGD {r['profit_gain']:.2f}")
col4.metric("Model Quality (R²)", f"{r['r2']:.3f}")

if r["r2"] < 0.5:
    st.warning("⚠️ R² is below 0.5 — the model fit is weak. Collect more sales data at varied price points for better accuracy.")

st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

# Charts with branded colour scheme
GREEN = "#2d9e5f"
LIGHT_GREEN = "#a8f0c6"
DARK_GREEN = "#1a6b3c"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#f7faf7")

for ax in [ax1, ax2]:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e0efe6")
    ax.spines["bottom"].set_color("#e0efe6")
    ax.tick_params(colors="#888", labelsize=10)
    ax.grid(True, color="#e8f5ee", linewidth=0.8)

# Left: Demand curve
ax1.scatter(product_df["price"], product_df["quantity_sold"],
            color=GREEN, s=60, zorder=5, label="Actual sales data", edgecolors="white", linewidths=0.8)
ax1.plot(r["price_range"], r["predicted_qty"],
         color=DARK_GREEN, linewidth=2.5, label=f"Q = {r['a']:.1f} − {r['b']:.1f}P")
ax1.axvline(r["optimal_price_profit"], color="#e05a2b", linestyle="--", linewidth=1.8,
            label=f"Optimal: SGD {r['optimal_price_profit']:.2f}")
ax1.set_xlabel("Price (SGD)", fontsize=11, color="#555")
ax1.set_ylabel("Quantity Sold", fontsize=11, color="#555")
ax1.set_title("Demand Curve", fontsize=13, fontweight="bold", color="#1a3d2b", pad=12)
ax1.legend(fontsize=9, framealpha=0.9, edgecolor="#e0efe6")

# Right: Revenue + Profit
ax2.plot(r["price_range"], r["predicted_revenue"],
         color=LIGHT_GREEN, linewidth=2.5, label="Predicted Revenue")
ax2.plot(r["price_range"], r["predicted_profit"],
         color=GREEN, linewidth=2.5, linestyle="--", label="Predicted Profit")
ax2.scatter([r["optimal_price_revenue"]], [r["optimal_revenue"]],
            color="#f0a500", s=100, zorder=5, edgecolors="white", linewidths=1,
            label=f"Max Revenue @ SGD {r['optimal_price_revenue']:.2f}")
ax2.scatter([r["optimal_price_profit"]], [r["optimal_profit"]],
            color="#e05a2b", s=160, marker="*", zorder=5,
            label=f"Max Profit @ SGD {r['optimal_price_profit']:.2f}")
ax2.axhline(0, color="#ccc", linewidth=1, linestyle=":")
ax2.set_xlabel("Price (SGD)", fontsize=11, color="#555")
ax2.set_ylabel("SGD", fontsize=11, color="#555")
ax2.set_title("Revenue & Profit Curves", fontsize=13, fontweight="bold", color="#1a3d2b", pad=12)
ax2.legend(fontsize=9, framealpha=0.9, edgecolor="#e0efe6")

plt.tight_layout(pad=2)
st.pyplot(fig)
plt.close()

# Model details expander
with st.expander("📐 View demand model details"):
    st.markdown(f"""
    - **Demand equation:** Q = {r['a']:.2f} − {r['b']:.2f} × P
    - **Interpretation:** Every SGD 1.00 price increase → ~{r['b']:.0f} fewer units sold
    - **Cost per unit:** SGD {r['cost']:.2f}
    - **Revenue optimum:** SGD {r['optimal_price_revenue']:.2f} → Revenue SGD {r['optimal_revenue']:.2f}
    - **Profit optimum:** SGD {r['optimal_price_profit']:.2f} → Profit SGD {r['optimal_profit']:.2f}
    - **R² score:** {r['r2']:.3f}
    """)

st.divider()

# =============================================================================
# SUMMARY TABLE — ALL PRODUCTS
# =============================================================================

st.subheader("📋 Full Summary — All Products")

summary_rows = []
for product in products:
    pdf = df[df["product_name"] == product].copy()
    res = analyse_product(product, pdf, price_min, price_max)
    summary_rows.append({
        "Product": product,
        "Cost/Unit (SGD)": round(res["cost"], 2),
        "Current Avg Price (SGD)": round(res["avg_price"], 2),
        "Current Profit/Period (SGD)": round(res["current_profit"], 2),
        "Optimal Price — Profit (SGD)": round(res["optimal_price_profit"], 2),
        "Max Profit/Period (SGD)": round(res["optimal_profit"], 2),
        "Potential Gain (SGD)": round(res["profit_gain"], 2),
        "R²": round(res["r2"], 3),
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(
    summary_df.style.background_gradient(subset=["Potential Gain (SGD)"], cmap="Greens"),
    hide_index=True,
    use_container_width=True
)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div class="footer">
    Built by <b>Thiery Advising Group</b> ·
    <a href="mailto:thieryadvisinggroup@gmail.com">thieryadvisinggroup@gmail.com</a>
    · Pricing model based on linear demand regression (Q = a − bP)
    · Results are estimates only — always validate before implementing
</div>
""", unsafe_allow_html=True)
