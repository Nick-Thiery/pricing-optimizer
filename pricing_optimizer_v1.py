# =============================================================================
# PRICING OPTIMIZATION SYSTEM — Version 1
# For small food & beverage businesses in Singapore
# Author: Nick
# Description: Reads sales data, fits a linear demand curve (Q = a - bP),
#              and finds the price that maximises predicted revenue.
# =============================================================================

# --- SECTION 1: Import libraries ---
# In Java, you'd import classes. In Python, you import libraries like this.
import pandas as pd          # for reading and working with tabular data (like Excel/CSV)
import numpy as np           # for numerical calculations (arrays, math operations)
import matplotlib.pyplot as plt  # for drawing charts
from sklearn.linear_model import LinearRegression  # for fitting the demand curve

# =============================================================================
# SECTION 2: SETTINGS — Change these to analyse a different product
# =============================================================================

CSV_FILE = "sales_data.csv"        # The name of your CSV file
PRODUCT = "Teh Tarik"              # Which product to analyse
PRICE_MIN = 1.00                   # Lowest price to test (SGD)
PRICE_MAX = 6.00                   # Highest price to test (SGD)
PRICE_STEP = 0.10                  # How finely to step through prices

# =============================================================================
# SECTION 3: LOAD AND FILTER DATA
# =============================================================================

# Read the CSV file into a DataFrame (think of this like a Java ArrayList of rows)
df = pd.read_csv(CSV_FILE)

# Filter rows to only keep the product we care about
# In Java: you'd loop through rows and check a condition.
# In Python/pandas: you write it as a one-liner "mask"
product_df = df[df["product_name"] == PRODUCT].copy()

# Check we actually found some data
if product_df.empty:
    print(f"Error: No data found for product '{PRODUCT}'. Check your CSV file.")
    exit()

print(f"\n{'='*50}")
print(f"  Pricing Optimiser — {PRODUCT}")
print(f"{'='*50}")
print(f"  Data points loaded: {len(product_df)}")
print(f"  Price range in data: SGD {product_df['price'].min():.2f} – {product_df['price'].max():.2f}")
print(f"  Qty range in data:   {product_df['quantity_sold'].min()} – {product_df['quantity_sold'].max()} units")

# =============================================================================
# SECTION 4: FIT THE LINEAR DEMAND MODEL  (Q = a - bP)
# =============================================================================
# We want to find 'a' and 'b' such that Q = a - bP fits our data best.
# This is a LINEAR REGRESSION problem: we're finding a straight line
# through our (price, quantity) scatter plot.
#
# sklearn's LinearRegression needs the input (price) as a 2D array.
# In Java terms: instead of a float[], it wants a float[][]
# That's what .reshape(-1, 1) does — turns a 1D list into a column of values.

X = product_df["price"].values.reshape(-1, 1)   # Prices (input / "feature")
y = product_df["quantity_sold"].values           # Quantities (output / "target")

# Create and train the model
# This is the magic line — sklearn finds the best 'a' and 'b' for you
model = LinearRegression()
model.fit(X, y)

# Extract a and b from the fitted model
# model.intercept_ = 'a' (the y-intercept)
# model.coef_[0]   = the slope (negative because demand goes DOWN as price goes UP)
a = model.intercept_          # e.g. ~200 (units sold if price were 0)
b = -model.coef_[0]           # We negate because the slope is negative; b should be positive

print(f"\n  Demand Model: Q = {a:.2f} - {b:.2f} × P")
print(f"  Interpretation: For every SGD 1.00 price increase,")
print(f"  you sell approximately {b:.1f} fewer units.")

# Model quality: R² score (how well the line fits the data)
# R² of 1.0 = perfect fit. R² of 0.0 = the model explains nothing.
r_squared = model.score(X, y)
print(f"\n  Model fit (R² score): {r_squared:.3f}")
if r_squared < 0.5:
    print("  ⚠️  Warning: R² is low. Your data may not follow a clear linear pattern.")
    print("     Consider collecting more varied price data.")

# =============================================================================
# SECTION 5: CALCULATE REVENUE ACROSS A RANGE OF PRICES
# =============================================================================
# Now we use our model to PREDICT what would happen at prices we haven't tried.
# We step through prices from PRICE_MIN to PRICE_MAX in small increments.

# np.arange works like a for loop: range(PRICE_MIN, PRICE_MAX, PRICE_STEP)
# but it supports decimal steps (Python's built-in range() only does integers)
price_range = np.arange(PRICE_MIN, PRICE_MAX + PRICE_STEP, PRICE_STEP)

# Predict quantity at each price using our model: Q = a - b*P
# We reshape again because the model expects a 2D input
predicted_qty = model.predict(price_range.reshape(-1, 1))

# Clip predicted quantity so it never goes below 0
# (It makes no sense to sell -5 units — that's just a model artifact)
predicted_qty = np.clip(predicted_qty, 0, None)

# Revenue = Price × Quantity
predicted_revenue = price_range * predicted_qty

# =============================================================================
# SECTION 6: FIND THE OPTIMAL PRICE
# =============================================================================
# np.argmax returns the INDEX of the highest value in an array.
# This is like finding the position of the maximum in a Java array using a loop,
# but in one line.

best_index = np.argmax(predicted_revenue)
optimal_price = price_range[best_index]
optimal_qty = predicted_qty[best_index]
optimal_revenue = predicted_revenue[best_index]

# Also calculate the mathematical optimum: P* = a / (2b)
# This comes from calculus (setting derivative of Revenue to 0)
# Useful as a cross-check
if b > 0:
    math_optimal = a / (2 * b)
else:
    math_optimal = optimal_price  # fallback if b is weird

print(f"\n{'='*50}")
print(f"  RECOMMENDED PRICE: SGD {optimal_price:.2f}")
print(f"  Predicted quantity at this price: {optimal_qty:.0f} units")
print(f"  Predicted revenue at this price:  SGD {optimal_revenue:.2f}")
print(f"  (Mathematical check: P* = a/2b = SGD {math_optimal:.2f})")
print(f"{'='*50}")

# Also show what's happening at current average price (as a comparison)
avg_price = product_df["price"].mean()
avg_qty_predicted = max(0, a - b * avg_price)
avg_revenue_predicted = avg_price * avg_qty_predicted
print(f"\n  Your average price in data: SGD {avg_price:.2f}")
print(f"  Predicted revenue at avg price: SGD {avg_revenue_predicted:.2f}")
revenue_gain = optimal_revenue - avg_revenue_predicted
print(f"  Potential revenue gain: SGD {revenue_gain:.2f} per period")

# =============================================================================
# SECTION 7: PLOT THE CHARTS
# =============================================================================
# We'll create two side-by-side charts:
#   Left chart:  Price vs Quantity (the demand curve)
#   Right chart: Price vs Revenue  (the revenue curve)

# Create a figure with 2 subplots side by side
# fig is the whole canvas; axes is a list of the two chart areas
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Give the whole figure a title
fig.suptitle(f"Pricing Analysis — {PRODUCT}", fontsize=16, fontweight='bold')

# ---- LEFT CHART: Price vs Quantity ----
ax1 = axes[0]

# Plot the raw data points as a scatter plot (actual observations)
ax1.scatter(
    product_df["price"],
    product_df["quantity_sold"],
    color="steelblue",
    label="Actual sales data",
    zorder=5,        # draw on top of the line
    s=60             # dot size
)

# Plot the fitted demand line (our model's prediction)
ax1.plot(
    price_range,
    predicted_qty,
    color="orange",
    linewidth=2,
    label=f"Demand model: Q = {a:.1f} - {b:.1f}P"
)

# Draw a vertical dashed line at the optimal price
ax1.axvline(
    x=optimal_price,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Optimal price (SGD {optimal_price:.2f})"
)

# Labels and formatting
ax1.set_xlabel("Price (SGD)", fontsize=12)
ax1.set_ylabel("Quantity Sold (units)", fontsize=12)
ax1.set_title("Demand Curve: Price vs Quantity", fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)  # light grid lines

# ---- RIGHT CHART: Price vs Revenue ----
ax2 = axes[1]

# Plot the revenue curve
ax2.plot(
    price_range,
    predicted_revenue,
    color="green",
    linewidth=2,
    label="Predicted revenue"
)

# Mark the optimal price with a red dot and a vertical line
ax2.scatter(
    [optimal_price],
    [optimal_revenue],
    color="red",
    s=100,
    zorder=5,
    label=f"Max revenue: SGD {optimal_revenue:.2f}\nat price SGD {optimal_price:.2f}"
)
ax2.axvline(
    x=optimal_price,
    color="red",
    linestyle="--",
    linewidth=1.5
)

# Mark the average current price with a grey dot
ax2.scatter(
    [avg_price],
    [avg_revenue_predicted],
    color="grey",
    s=80,
    zorder=5,
    label=f"Current avg price: SGD {avg_price:.2f}"
)

# Labels and formatting
ax2.set_xlabel("Price (SGD)", fontsize=12)
ax2.set_ylabel("Predicted Revenue (SGD)", fontsize=12)
ax2.set_title("Revenue Curve: Price vs Predicted Revenue", fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Adjust spacing between the two charts
plt.tight_layout()

# Save the chart as a PNG file (useful for a future web app)
output_filename = f"pricing_chart_{PRODUCT.replace(' ', '_')}.png"
plt.savefig(output_filename, dpi=150, bbox_inches="tight")
print(f"\n  Chart saved as: {output_filename}")

# Show the chart on screen
plt.show()

print("\n  Done! Review the chart and recommendations above.")
print("  Remember: This model optimises REVENUE, not PROFIT.")
print("  To find true optimal price, subtract your costs in Version 2.\n")
