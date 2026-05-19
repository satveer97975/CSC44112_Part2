# ============================================================
#  RETAIL SALES FORECASTING — Full Standalone Analysis
#  Run with: python3 sales_forecasting_analysis.py
#  Requires: pandas numpy matplotlib seaborn scikit-learn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output folder for charts
os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("   RETAIL SALES FORECASTING — ANALYSIS STARTING")
print("=" * 60)

# ── SECTION 1: LOAD DATA ─────────────────────────────────────
print("\n[1/6] Loading dataset...")

try:
    df = pd.read_csv('Walmart.csv')
except FileNotFoundError:
    print("\n❌ ERROR: Walmart.csv not found!")
    print("   Please download it from:")
    print("   https://www.kaggle.com/datasets/yasserh/walmart-dataset")
    print("   and place it in the same folder as this script.\n")
    exit()

print(f"✅ Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())

# ── SECTION 2: EDA ───────────────────────────────────────────
print("\n[2/6] Running Exploratory Data Analysis...")

print("\n--- Basic Info ---")
print(f"Shape        : {df.shape}")
print(f"Missing vals : {df.isnull().sum().sum()}")
print(f"Duplicates   : {df.duplicated().sum()}")
print(f"\n--- Weekly Sales Summary ---")
print(f"Min    : ${df['Weekly_Sales'].min():>15,.2f}")
print(f"Max    : ${df['Weekly_Sales'].max():>15,.2f}")
print(f"Mean   : ${df['Weekly_Sales'].mean():>15,.2f}")
print(f"Median : ${df['Weekly_Sales'].median():>15,.2f}")
print(f"Std    : ${df['Weekly_Sales'].std():>15,.2f}")

# Chart 1: Sales Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Weekly Sales Distribution', fontsize=14, fontweight='bold')
axes[0].hist(df['Weekly_Sales'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Histogram')
axes[0].set_xlabel('Weekly Sales ($)')
axes[0].set_ylabel('Frequency')
axes[1].boxplot(df['Weekly_Sales'], patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[1].set_title('Box Plot')
axes[1].set_ylabel('Weekly Sales ($)')
plt.tight_layout()
plt.savefig('outputs/1_sales_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Chart saved: outputs/1_sales_distribution.png")

# Chart 2: Sales by Store
store_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
plt.figure(figsize=(16, 5))
store_sales.plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Average Weekly Sales by Store', fontsize=14, fontweight='bold')
plt.xlabel('Store Number')
plt.ylabel('Average Weekly Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/2_sales_by_store.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart saved: outputs/2_sales_by_store.png")
print(f"   Top store    : Store {store_sales.index[0]}  (${store_sales.iloc[0]:,.0f}/week)")
print(f"   Bottom store : Store {store_sales.index[-1]} (${store_sales.iloc[-1]:,.0f}/week)")

# Chart 3: Holiday vs Non-Holiday
holiday_avg = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
plt.figure(figsize=(7, 5))
bars = plt.bar(['Non-Holiday', 'Holiday'], holiday_avg.values,
               color=['steelblue', 'coral'], edgecolor='white', width=0.5)
plt.title('Avg Weekly Sales: Holiday vs Non-Holiday', fontsize=13, fontweight='bold')
plt.ylabel('Average Weekly Sales ($)')
for bar, val in zip(bars, holiday_avg.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'${val:,.0f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/3_holiday_vs_nonholiday.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart saved: outputs/3_holiday_vs_nonholiday.png")

# Chart 4: Correlation Heatmap
plt.figure(figsize=(9, 6))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, vmin=-1, vmax=1, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/4_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart saved: outputs/4_correlation_heatmap.png")

# ── SECTION 3: FEATURE ENGINEERING ──────────────────────────
print("\n[3/6] Feature Engineering...")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month']   = df['Date'].dt.month
df['Week']    = df['Date'].dt.isocalendar().week.astype(int)
df['Year']    = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter
df_model = df.drop(columns=['Date'])

print("✅ Extracted features: Month, Week, Year, Quarter")
print(f"   Final feature count: {df_model.shape[1] - 1} features + 1 target")

# ── SECTION 4: MODEL TRAINING ────────────────────────────────
print("\n[4/6] Training Models...")

X = df_model.drop(columns=['Weekly_Sales'])
y = df_model['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"   Train samples : {X_train.shape[0]:,}")
print(f"   Test samples  : {X_test.shape[0]:,}")

models = {
    'Linear Regression':  LinearRegression(),
    'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\n   Training {name}...", end=" ")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        'MAE':  mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2':   r2_score(y_test, preds),
        'preds': preds
    }
    print("done ✅")

# ── SECTION 5: RESULTS ───────────────────────────────────────
print("\n[5/6] Evaluating Results...")
print("\n" + "=" * 55)
print(f"  {'Model':<22} {'MAE':>10} {'RMSE':>12} {'R²':>8}")
print("=" * 55)
for name, r in results.items():
    print(f"  {name:<22} ${r['MAE']:>9,.0f} ${r['RMSE']:>10,.0f} {r['R2']:>8.4f}")
print("=" * 55)

best = max(results, key=lambda x: results[x]['R2'])
print(f"\n🏆 Best Model: {best} (R² = {results[best]['R2']:.4f})")

# Chart 5: Model Comparison
metrics_df = pd.DataFrame({n: {'MAE': v['MAE'], 'RMSE': v['RMSE'], 'R2': v['R2']}
                            for n, v in results.items()}).T
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#C44E52', '#4C72B0', '#55A868']
for i, (metric, title, fmt) in enumerate([
    ('MAE',  'Mean Absolute Error\n(lower is better)', '${:,.0f}'),
    ('RMSE', 'Root Mean Squared Error\n(lower is better)', '${:,.0f}'),
    ('R2',   'R² Score\n(higher is better)', '{:.3f}')
]):
    bars = axes[i].bar(metrics_df.index, metrics_df[metric],
                       color=colors, edgecolor='white', width=0.5)
    axes[i].set_title(title, fontsize=11, fontweight='bold')
    axes[i].set_xticklabels(metrics_df.index, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, metrics_df[metric]):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() * 1.01, fmt.format(val),
                     ha='center', fontsize=9, fontweight='bold')
plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart saved: outputs/5_model_comparison.png")

# Chart 6: Actual vs Predicted
best_preds = results[best]['preds']
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_preds, alpha=0.3, color='steelblue', s=10)
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
plt.title(f'Actual vs Predicted — {best}', fontsize=13, fontweight='bold')
plt.xlabel('Actual Weekly Sales ($)')
plt.ylabel('Predicted Weekly Sales ($)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/6_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart saved: outputs/6_actual_vs_predicted.png")

# Chart 7: Feature Importance
rf_model = models['Random Forest']
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
importance.plot(kind='barh', color='steelblue', edgecolor='white')
plt.title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/7_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart saved: outputs/7_feature_importance.png")

# ── SECTION 6: SUMMARY ───────────────────────────────────────
print("\n[6/6] Done! Summary:")
print("\n📁 All charts saved in the 'outputs/' folder:")
print("   1_sales_distribution.png")
print("   2_sales_by_store.png")
print("   3_holiday_vs_nonholiday.png")
print("   4_correlation_heatmap.png")
print("   5_model_comparison.png")
print("   6_actual_vs_predicted.png")
print("   7_feature_importance.png")
print(f"\n🏆 Best model : {best}")
print(f"   R² Score   : {results[best]['R2']:.4f}")
print(f"   MAE        : ${results[best]['MAE']:,.2f}")
print(f"   RMSE       : ${results[best]['RMSE']:,.2f}")
print("\n✅ Analysis complete!")