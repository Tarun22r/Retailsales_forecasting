import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings

warnings.filterwarnings('ignore')

# 1. Load libraries with robust Prophet check
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from prophet import fbprophet
        PROPHET_AVAILABLE = True
    except ImportError:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

print(f'Prophet available: {PROPHET_AVAILABLE}')

# 2. Load dataset
# Ensure the file exists in your current directory
try:
    df = pd.read_csv('RetailSales.csv')
    print('Shape:', df.shape)
except FileNotFoundError:
    print("Error: RetailSales.csv not found. Please check the file path.")

# 3. Preprocess
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
# Remove rows with invalid dates and sort
df = df.dropna(subset=['Date']).sort_values('Date')

# Ensure numeric sales, forcing errors to NaN then filling with 0
df['Total_Sales'] = pd.to_numeric(df['Total_Sales'], errors='coerce').fillna(0)

print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print("-" * 30)
df.info()  # Fixed: removed display() wrapper
print("-" * 30)

# 4. Aggregate to daily sales
# Using resample('D') is often safer for time series gaps
daily = df.groupby('Date')['Total_Sales'].sum().resample('D').sum().to_frame(name='Sales')
print("Daily Head:\n", daily.head())

# 5. EDA - plots
plt.figure(figsize=(12, 5))
plt.plot(daily.index, daily['Sales'], color='#2ca02c', label='Actual Sales')
plt.title('Daily Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Summary Tables
cat_summary = df.groupby('Category')['Total_Sales'].sum().sort_values(ascending=False).to_frame()
top_products = df.groupby('Product_Name')['Total_Sales'].sum().sort_values(ascending=False).head(10).to_frame()

print("\nTop Categories:\n", cat_summary)

# 6. Forecasting - next 30 days
H = 30
if PROPHET_AVAILABLE:
    # Prophet requires columns 'ds' and 'y'
    prophet_df = daily.reset_index().rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    # Initialize and fit
    m = Prophet(daily_seasonality=False, yearly_seasonality=True)
    m.fit(prophet_df)
    
    # Create future dates
    future = m.make_future_dataframe(periods=H)
    fcst = m.predict(future)
    
    # Extract only the forecast period
    forecast_results = fcst.tail(H).set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]
    forecast = forecast_results.rename(columns={'yhat': 'Forecast', 'yhat_lower': 'Lower', 'yhat_upper': 'Upper'})
else:
    # SARIMAX fallback
    series = daily['Sales']
    # (1,1,1) x (1,1,1,7) is a standard baseline for weekly retail cycles
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7), 
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    
    pred = res.get_forecast(steps=H)
    idx = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=H, freq='D')
    
    forecast = pd.DataFrame({
        'Forecast': pred.predicted_mean.values,
        'Lower': pred.conf_int().iloc[:, 0].values,
        'Upper': pred.conf_int().iloc[:, 1].values
    }, index=idx)

# Clip negative values to 0 (Sales can't be negative)
forecast = forecast.clip(lower=0)

# Save results
try:
    forecast.to_csv('forecast_results.csv')
    print(f'\nForecast success! Saved to forecast_results.csv')
    print(forecast.head())
except Exception as e:
    print(f"Could not save CSV: {e}")