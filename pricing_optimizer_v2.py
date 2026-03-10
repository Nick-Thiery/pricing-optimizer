# =============================================================================
# PRICING OPTIMIZATION SYSTEM — Version 2
# Improvements over V1:
#   1. Optimises for PROFIT (not just revenue)
#   2. Analyses ALL products automatically in a loop
#   3. Prints a clean summary table at the end
#   4. Charts now show revenue AND profit curves together
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =============================================================================
# SECTION 1: SETTINGS
# =============================================================================

import os
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sales_data_v2.csv")   # Updated CSV now includes cost_per_unit
PRICE_MIN = 0.50                  # Lowest price to test (SGD)
PRICE_MAX = 7.00                  # Highest price to test (SGD)
PRICE_STEP = 0.10                 # Step size when scanning prices

# Colour scheme for charts — one colour per product
# If you add more products, add more colours here
COLOURS = ["steelblue", "coral", "mediumseagreen", "mediumpurple", "goldenrod"]

# =============================================================================
# SECTION 2: LOAD DATA
# =============================================================================

df = pd.read_csv(CSV_FILE)

# Get a list of all unique product names in the CSV
# In Java: like getting all unique values from a Set
products = df["product_name"].unique()

print("\n" + "="*60)
print("  PRICING OPTIMISATION SYSTEM — Version 2")
print("="*60)
print(f"  Products found in data: {list(products)}")
print(f"  Total rows loaded: {len(df)}")
print("="*60)

# =============================================================================
# SECTION 3: DEFINE A FUNCTION TO ANALYSE ONE PRODUCT
# =============================================================================
# In V1 we wrote all the logic once for one product.
# In V2, we wrap it in a FUNCTION so we can call it for each product.
# In Java terms: this is like writing a method you call in a for-loop.

def analyse_product(product_name, product_df, colour, ax_row):
    """
    Fits a demand model for one product and plots revenue + profit curves.

    Parameters:
        product_name  : string, e.g. "Teh Tarik"
        product_df    : DataFrame filtered to just this product's rows
        colour        : string, chart colour for this product
        ax_row        : list of 2 matplotlib axes (chart areas) for this product
    
    Returns:
        A dictionary with the key results (used for the summary table later)
    """

    # --- Get the cost per unit (assumed constant for this product) ---
    # We take the average in case there are tiny rounding differences in the CSV
    cost = product_df["cost_per_unit"].mean()

    # --- Fit the linear demand model: Q = a - bP ---
    X = product_df["price"].values.reshape(-1, 1)
    y = product_df["quantity_sold"].values

    model = LinearRegression()
    model.fit(X, y)

    a = model.intercept_      # y-intercept of demand line
    b = -model.coef_[0]       # price sensitivity (negated so b is positive)
    r2 = model.score(X, y)    # how well the line fits (0 to 1)

    # --- Scan across price range ---
    price_range = np.arange(PRICE_MIN, PRICE_MAX + PRICE_STEP, PRICE_STEP)

    # Predicted quantity at each price (clipped to 0 minimum)
    predicted_qty = np.clip(model.predict(price_range.reshape(-1, 1)), 0, None)

    # Revenue = Price × Quantity  (same as V1)
    predicted_revenue = price_range * predicted_qty

    # Profit = (Price - Cost) × Quantity  ← NEW in V2
    # This is the key difference: we subtract cost before multiplying
    # At prices below cost, profit is negative (you're losing money per unit)
    predicted_profit = (price_range - cost) * predicted_qty

    # --- Find optimal prices ---
    # For revenue: same as V1
    best_revenue_idx = np.argmax(predicted_revenue)
    optimal_price_revenue = price_range[best_revenue_idx]
    optimal_revenue = predicted_revenue[best_revenue_idx]

    # For profit: same logic, but applied to profit array
    best_profit_idx = np.argmax(predicted_profit)
    optimal_price_profit = price_range[best_profit_idx]
    optimal_profit = predicted_profit[best_profit_idx]

    # Current average price in the data (for comparison)
    avg_price = product_df["price"].mean()
    avg_qty = max(0, a - b * avg_price)
    current_profit = (avg_price - cost) * avg_qty

    # --- Print results for this product ---
    print(f"\n  Product: {product_name}")
    print(f"  Cost per unit: SGD {cost:.2f}")
    print(f"  Demand model: Q = {a:.1f} - {b:.1f}P   (R² = {r2:.3f})")
    print(f"  ── Revenue optimum:  SGD {optimal_price_revenue:.2f}  →  Revenue SGD {optimal_revenue:.2f}")
    print(f"  ── Profit optimum:   SGD {optimal_price_profit:.2f}  →  Profit  SGD {optimal_profit:.2f}")
    print(f"  ── Current avg price: SGD {avg_price:.2f}  →  Profit  SGD {current_profit:.2f}")
    gain = optimal_profit - current_profit
    print(f"  ── Profit gain if repriced: SGD {gain:.2f} per period")

    if r2 < 0.5:
        print(f"  ⚠️  Warning: R² = {r2:.2f} — model fit is weak. Collect more varied price data.")

    # --- Plot: Left chart = Demand curve, Right chart = Revenue + Profit ---
    ax1, ax2 = ax_row

    # LEFT: Demand curve
    ax1.scatter(product_df["price"], product_df["quantity_sold"],
                color=colour, s=50, zorder=5, label="Actual data")
    ax1.plot(price_range, predicted_qty,
             color=colour, linewidth=2, label=f"Q = {a:.1f} - {b:.1f}P")
    ax1.axvline(optimal_price_profit, color="red", linestyle="--", linewidth=1.5,
                label=f"Profit optimum: SGD {optimal_price_profit:.2f}")
    ax1.set_title(f"{product_name} — Demand Curve", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Price (SGD)")
    ax1.set_ylabel("Quantity Sold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # RIGHT: Revenue AND Profit curves together
    ax2.plot(price_range, predicted_revenue,
             color=colour, linewidth=2, linestyle="-", label="Predicted Revenue")
    ax2.plot(price_range, predicted_profit,
             color=colour, linewidth=2, linestyle="--", label="Predicted Profit")

    # Mark revenue optimum (circle)
    ax2.scatter([optimal_price_revenue], [optimal_revenue],
                color="orange", s=80, zorder=5,
                label=f"Max Revenue @ SGD {optimal_price_revenue:.2f}")

    # Mark profit optimum (star) — this is the one to act on
    ax2.scatter([optimal_price_profit], [optimal_profit],
                color="red", s=120, marker="*", zorder=5,
                label=f"Max Profit @ SGD {optimal_price_profit:.2f}")

    # Draw a horizontal line at profit = 0 (break-even reference)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")

    ax2.set_title(f"{product_name} — Revenue & Profit", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Price (SGD)")
    ax2.set_ylabel("SGD")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Return key results as a dictionary — used to build the summary table
    return {
        "Product": product_name,
        "Cost/Unit (SGD)": round(cost, 2),
        "Current Avg Price (SGD)": round(avg_price, 2),
        "Current Profit/Period (SGD)": round(current_profit, 2),
        "Optimal Price — Revenue (SGD)": round(optimal_price_revenue, 2),
        "Optimal Price — Profit (SGD)": round(optimal_price_profit, 2),
        "Max Profit/Period (SGD)": round(optimal_profit, 2),
        "Profit Gain (SGD)": round(gain, 2),
        "R²": round(r2, 3),
    }

# =============================================================================
# SECTION 4: LOOP THROUGH ALL PRODUCTS
# =============================================================================
# This is the loop that calls our function for each product.
# In V1, we did this manually for one product.
# Now it runs automatically for however many products are in the CSV.

# Set up the chart grid:
# - One ROW per product
# - Two COLUMNS: left = demand curve, right = revenue/profit curve
n = len(products)   # number of products
fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
fig.suptitle("Pricing Optimisation — All Products (V2)", fontsize=16, fontweight="bold")

# If there's only 1 product, axes won't be a 2D list — this fixes that edge case
if n == 1:
    axes = [axes]

# This list will collect results from each product for the summary table
summary_results = []

for i, product in enumerate(products):
    # Filter data for this product
    product_df = df[df["product_name"] == product].copy()

    # Pick a colour (cycles if more products than colours)
    colour = COLOURS[i % len(COLOURS)]

    # Run the analysis and collect results
    result = analyse_product(product, product_df, colour, axes[i])
    summary_results.append(result)

# Tighten chart layout
plt.tight_layout()

# Save chart
chart_file = "pricing_chart_v2_all_products.png"
plt.savefig(chart_file, dpi=150, bbox_inches="tight")
print(f"\n  Chart saved as: {chart_file}")
plt.show()

# =============================================================================
# SECTION 5: PRINT SUMMARY TABLE
# =============================================================================
# Convert results list into a DataFrame — great for displaying as a clean table
summary_df = pd.DataFrame(summary_results)

print("\n" + "="*60)
print("  SUMMARY TABLE — All Products")
print("="*60)

# Print each column clearly since terminal tables can be hard to read
for _, row in summary_df.iterrows():
    print(f"\n  {row['Product']}")
    print(f"    Cost per unit:          SGD {row['Cost/Unit (SGD)']:.2f}")
    print(f"    Current avg price:      SGD {row['Current Avg Price (SGD)']:.2f}")
    print(f"    Current profit/period:  SGD {row['Current Profit/Period (SGD)']:.2f}")
    print(f"    → Optimal price (profit): SGD {row['Optimal Price — Profit (SGD)']:.2f}")
    print(f"    → Max profit/period:      SGD {row['Max Profit/Period (SGD)']:.2f}")
    print(f"    → Potential profit gain:  SGD {row['Profit Gain (SGD)']:.2f}")
    print(f"    Model quality (R²):     {row['R²']}")

print("\n" + "="*60)
print("  KEY REMINDER:")
print("  Revenue optimum ≠ Profit optimum.")
print("  Always use the PROFIT optimum when making real decisions.")
print("  These figures are per observation period in your data.")
print("="*60 + "\n")
