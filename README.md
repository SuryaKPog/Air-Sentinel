# Air-Sentinel
# 🌫️ Air Quality Prediction & Alert System

This project predicts air quality levels (e.g., **PM2.5**, **CO**, **Ozone**) for various cities and sends **real-time alerts** to citizens in case of hazardous pollution.
The system also generates **precautionary guidelines** using OpenAI and visualizes the data using **heatmaps**.

## 🚀 Features

- ✅ Real-time AQI fetch using OpenWeatherMap API
- 📊 Predictive model using Random Forest Regressor
- 🔔 Alert system via SMS using Twilio API
- 🌐 Safety guideline generation with OpenAI (ChatGPT)
- 🗺️ Heatmap visualization of pollutant levels using Folium
- 📉 Historical data analysis with visualizations
- 📦 Model serialization using `joblib`

---

## 🧠 Model Type

- **Regression-based Prediction**
- Algorithm: `RandomForestRegressor`

---

## 📦 Tech Stack

| Component        | Tech Used                     |
|------------------|-------------------------------|
| ML Model         | Scikit-learn (Random Forest)  |
| Realtime Data    | OpenWeatherMap API            |
| Alerts           | Twilio API (SMS Notifications)|
| Heatmaps         | Folium + NumPy                |
| Visualization    | Matplotlib                    |
| AI Guidelines    | OpenAI GPT-3.5 Turbo          |
| Deployment Ready | Model exported via Joblib     |

---

## 📁 Dataset

- [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- [Kaggle Air Quality Data](https://www.kaggle.com/datasets)
