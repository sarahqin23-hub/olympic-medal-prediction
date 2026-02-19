"""
02_train_models.py

This script performs feature engineering and trains machine learning models
to predict Olympic medal counts.

Steps:
    1. Load the modeling dataset.
    2. Create the previous_medals feature using historical results.
    3. Merge GDP data from the World Bank.
    4. Remove observations with missing GDP values.
    5. Train Linear Regression and Random Forest models.
    6. Evaluate model performance using MAE and RMSE.
    7. Visualize prediction results

The script is designed to ensure reproducibility of the modeling results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ===== locate data directory =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(BASE_DIR, "summer_model_table.csv")

# ===== load data =====
df = pd.read_csv(DATA_FILE)

print("Loaded:", df.shape)

# ===== sort by country and year =====
df = df.sort_values(["country_3_letter_code", "year"])

# ===== generate previous_medals =====
df["previous_medals"] = (
    df.groupby("country_3_letter_code")["total_medals"]
    .shift(1)
)

# first olympic ,fill with 0
df["previous_medals"] = df["previous_medals"].fillna(0)

print(df.head(10))

# ===== load GDP（World Bank）and merge =====
GDP_FILE = os.path.join(BASE_DIR, "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_155.csv")  # ← 改成你的实际文件名

gdp_raw = pd.read_csv(GDP_FILE, skiprows=4)  # World Bank 前面有说明行
print("Loaded GDP raw:", gdp_raw.shape)
print("GDP columns:", gdp_raw.columns[:10])

# world bank files contain description rows at the top
# World Bank country code column is usually named "Country Code"
id_cols = ["Country Name", "Country Code"]
year_cols = [c for c in gdp_raw.columns if c.isdigit()]  # '1960'...'2022'

gdp_long = (
    gdp_raw[id_cols + year_cols]
    .melt(id_vars=id_cols, var_name="year", value_name="gdp")
    .rename(columns={"Country Code": "country_3_letter_code"})
)

gdp_long["year"] = gdp_long["year"].astype(int)

print("GDP long:", gdp_long.shape)
print(gdp_long.head())

# Merge with Olympic table (left join: keep all Olympic records)
df = df.merge(
    gdp_long[["country_3_letter_code", "year", "gdp"]],
    on=["country_3_letter_code", "year"],
    how="left"
)

print("\nAfter merge:", df.shape)
print("GDP missing %:", (df["gdp"].isna().mean() * 100).round(2))

print(df.head(10))

# ===== Keep only rows with available GDP data =====
df = df.dropna(subset=["gdp"])

print("\nAfter removing missing GDP:")
print(df.shape)
print("Year range:", df["year"].min(), "-", df["year"].max())

print(df.columns)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ===== 1. Define X and y =====
X = df[["previous_medals", "gdp"]]
y = df["total_medals"]

# ===== 2. Split into training and test sets =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ===== 3. Train Linear Regression  =====
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# ===== 4. Prediction =====
y_pred = model_lr.predict(X_test)

# ===== 5. Compute error metrics =====
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nLinear Regression Results")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

from sklearn.ensemble import RandomForestRegressor

# ===== Random Forest =====
model_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model_rf.fit(X_train, y_train)

# Prediction
y_pred_rf = model_rf.predict(X_test)

# Error metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\nRandom Forest Results")
print("MAE:", round(mae_rf, 2))
print("RMSE:", round(rmse_rf, 2))

print("\nLinear Regression coefficients:")
print(dict(zip(X.columns, model_lr.coef_)))

# ===== Prediction vs Actual Scatter Plot =====

plt.figure(figsize=(7, 7))

# Linear Regression
plt.scatter(
    y_test, y_pred,
    alpha=0.6,
    label="Linear Regression"
)

# Random Forest
plt.scatter(
    y_test, y_pred_rf,
    alpha=0.6,
    label="Random Forest"
)

# Perfect prediction line (y = x)
min_val = min(y_test.min(), y_pred.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred.max(), y_pred_rf.max())

plt.plot([min_val, max_val],
         [min_val, max_val],
         linestyle="--",
         color="black",
         label="Perfect Prediction")

plt.xlabel("Actual Medals")
plt.ylabel("Predicted Medals")
plt.title("Predicted vs Actual Medal Counts (Test Set)")
plt.legend()

plt.show()
