import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

DATA_PATH = "data/passengers.csv"
FORECAST_PATH = "outputs/forecast_160d.csv"

SHOW_PLOT = False

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

df_forecast = pd.read_csv(FORECAST_PATH)
df_forecast['date'] = pd.to_datetime(df_forecast['date'])
df_forecast = df_forecast.sort_values('date')
df_forecast = df_forecast.set_index('date')
df_forecast_monthly = df_forecast.resample('MS').mean().reset_index()

print(df_forecast_monthly)


# 1. Line Plot (MOST IMPORTANT for time series)
plt.figure(figsize=(24,8))
plt.plot(df['date'], df['total'], marker='o')
plt.title("Total Over Time")
plt.xlabel("Date")
plt.ylabel("Total")
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(f"{output_dir}/line_plot.png", dpi=100)
print("1. Line Plot saved!")
if SHOW_PLOT:
    plt.show()

# 1.1 Line Plot Forecast (Data with Forecast 160 days)
plt.figure(figsize=(24,8))
plt.plot(df['date'], df['total'], marker='o', color='blue', label='Actual')
plt.plot(df_forecast_monthly['date'], df_forecast_monthly['total'], marker='o', color='orange', label='Forecast')
plt.title("Forecast 160 Days")
plt.xlabel("Date")
plt.ylabel("Total")
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(f"{output_dir}/line_plot_forecast_160d.png", dpi=100)
print("1.1 Line Plot Forecast saved!")
if SHOW_PLOT:
    plt.show()

# 2. Histogram (Shows distribution shape)
plt.figure()
plt.hist(df['total'], bins=20)
plt.title("Distribution of Total")
plt.xlabel("Total")
plt.ylabel("Frequency")
plt.savefig(f"{output_dir}/histogram.png", dpi=100)
print("2. Histogram saved!")
if SHOW_PLOT:
    plt.show()

# 3. Boxplot (Simple Outlier Detector)
plt.figure()
plt.boxplot(df['total'])
plt.title("Boxplot of Total")
plt.savefig(f"{output_dir}/boxplot.png", dpi=100)
print("3. Boxplot saved!")
if SHOW_PLOT:
    plt.show()

# 4. Rolling Mean + Rolling Std (Shows abnormal deviations)
df['rolling_mean'] = df['total'].rolling(window=12).mean()
df['rolling_std'] = df['total'].rolling(window=12).std()
plt.figure()
plt.plot(df['date'], df['total'])
plt.plot(df['date'], df['rolling_mean'])
plt.fill_between(df['date'],
                 df['rolling_mean'] - 2*df['rolling_std'],
                 df['rolling_mean'] + 2*df['rolling_std'],
                 alpha=0.2)
plt.title("Rolling Mean and Std")
plt.savefig(f"{output_dir}/rolling_mean_std.png", dpi=100)
print("4. Rolling Mean + Rolling Std saved!")
if SHOW_PLOT:
    plt.show()

# 5. Z-Score Plot
df['z_score'] = (df['total'] - df['total'].mean()) / df['total'].std()
plt.figure()
plt.plot(df['date'], df['z_score'])
plt.axhline(3)
plt.axhline(-3)
plt.title("Z-Score Over Time")
plt.savefig(f"{output_dir}/z_score.png", dpi=100)
print("5. Z-Score Plot saved!")
if SHOW_PLOT:
    plt.show()

# 6. Seasonal Plot (Month-by-Month Pattern)
df['month'] = df['date'].dt.month
plt.figure()
for year, group in df.groupby(df['date'].dt.year):
    plt.plot(group['month'], group['total'])

plt.title("Seasonal Pattern by Year")
plt.xlabel("Month")
plt.ylabel("Total")
plt.savefig(f"{output_dir}/seasonal_plot.png", dpi=100)
print("6. Seasonal Plot saved!")
if SHOW_PLOT:
    plt.show()

# 7. Yearly Boxplot (Seasonality Strength)
plt.figure()
df.boxplot(column='total', by='month')
plt.title("Monthly Distribution")
plt.suptitle("")
plt.savefig(f"{output_dir}/yearly_boxplot.png", dpi=100)
print("7. Yearly Boxplot saved!")
if SHOW_PLOT:
    plt.show()

# 8. Decomposition (VERY IMPORTANT)
from statsmodels.tsa.seasonal import seasonal_decompose
df_ts = df.set_index('date')
result = seasonal_decompose(df_ts['total'], model='additive', period=12)
result.plot()
plt.savefig(f"{output_dir}/ecomposition.png", dpi=100)
print("8. Decomposition saved!")
if SHOW_PLOT:
    plt.show()

# 9. ACF & PACF (For Model Selection) [If you’re testing ARIMA/SARIMA]
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df_ts['total'])
plt.savefig(f"{output_dir}/acf.png", dpi=100)
print("9. ACF saved!")
if SHOW_PLOT:
    plt.show()
plot_pacf(df_ts['total'])
plt.savefig(f"{output_dir}/pacf.png", dpi=100)
print("9. PACF saved!")
if SHOW_PLOT:
    plt.show()

# 10. Structural Break Visualization (Important for 2020)
# 2020 is clearly a regime change, Plot pre-2020 vs post-2020:
plt.figure()
plt.plot(df[df['date'] < '2020-01-01']['date'],
         df[df['date'] < '2020-01-01']['total'])

plt.plot(df[df['date'] >= '2020-01-01']['date'],
         df[df['date'] >= '2020-01-01']['total'])

plt.title("Pre vs Post 2020")
plt.savefig(f"{output_dir}/pre_vs_post_2020.png", dpi=100)
print("10. Structural Break Visualization saved!")
if SHOW_PLOT:
    plt.show()
