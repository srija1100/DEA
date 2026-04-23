import streamlit as st
import pandas as pd
import plotly.express as px

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

st.set_page_config(page_title="Air Quality Analysis Dashboard", layout="wide")

st.title("🌍 Air Quality Analysis Dashboard")

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------
df = pd.read_csv("air_pollution_large_dataset.csv")
df['date'] = pd.to_datetime(df['date'])

st.success("Dataset Loaded Successfully!")

# ---------------------------------------------------
# Sidebar filter
# ---------------------------------------------------
st.sidebar.header("Filter Data")
year = st.sidebar.selectbox("Select Year", sorted(df['date'].dt.year.unique()))
df_filtered = df[df['date'].dt.year == year]

# ---------------------------------------------------
# KPI Metrics
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Average AQI", round(df_filtered['aqi'].mean(), 2))
col2.metric("Max AQI", df_filtered['aqi'].max())
col3.metric("Min AQI", df_filtered['aqi'].min())
col4.metric("Average PM2.5", round(df_filtered['pm2_5'].mean(), 2))

st.markdown("---")

# ---------------------------------------------------
# AQI Trend
# ---------------------------------------------------
st.subheader("📈 AQI Trend Over Time")
fig = px.line(df_filtered, x="date", y="aqi", title="AQI Trend")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# Pollution Comparison
# ---------------------------------------------------
st.subheader("📊 Pollutant Comparison")
pollutants = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'ozone']
fig2 = px.box(df_filtered, y=pollutants, title="Pollutant Distribution")
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------
st.subheader("🔥 Correlation Between Pollutants")
corr = df_filtered[pollutants + ['aqi']].corr()
fig3 = px.imshow(corr, text_auto=True, title="Correlation Matrix")
st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# 🔮 AQI PREDICTION USING XGBOOST (ADVANCED MODEL)
# =====================================================

st.markdown("---")
st.subheader("🚀 AQI Prediction Using XGBoost")

# Features and Target
X = df[pollutants]
y = df['aqi']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost Model
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write("### 📊 Model Performance")
st.write("R² Score:", round(r2, 3))
st.write("Mean Absolute Error:", round(mae, 2))

# ---------------------------------------------------
# User Input Prediction
# ---------------------------------------------------
st.write("### Enter Pollution Levels to Predict AQI")

col1, col2, col3 = st.columns(3)

pm2_5 = col1.number_input("PM2.5", min_value=0.0)
pm10 = col1.number_input("PM10", min_value=0.0)

no2 = col2.number_input("NO2", min_value=0.0)
so2 = col2.number_input("SO2", min_value=0.0)

co = col3.number_input("CO", min_value=0.0)
ozone = col3.number_input("Ozone", min_value=0.0)

if st.button("Predict AQI"):
    input_data = [[pm2_5, pm10, no2, so2, co, ozone]]
    prediction = model.predict(input_data)
    st.success(f"Predicted AQI: {round(prediction[0], 2)}")
