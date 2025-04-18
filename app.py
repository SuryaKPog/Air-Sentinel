import pandas as pd
import requests
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from twilio.rest import Client
import numpy as np
import openai  # Make sure to install openai package
import os  

# 1) Fetch live air‐quality from OpenWeatherMap
API_KEY = 'your_weather_api_key_here'
LAT, LON = '13.0827', '80.2707'  # Updated to Chennai coordinates
url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
resp = requests.get(url).json()
live_aqi = resp['list'][0]['main']['aqi']
live_components = resp['list'][0]['components']
print(resp)  # Add this line to check the response from the API
print(f"Live AQI Level: {live_aqi}")
print("Live pollutant concentrations (µg/m³):")
for p, v in live_components.items():
    print(f"  {p}: {v}")

# 2) Load & clean historical dataset
data = pd.read_csv('AirQuality.csv', delimiter=';')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')

numeric_cols = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
    'T', 'RH', 'AH'
]
for col in numeric_cols:
    data[col] = data[col].astype(str).str.replace(',', '', regex=False)
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].fillna(data[col].mean())

# drop unused columns
data.drop(columns=['Date','Time','Unnamed: 15','Unnamed: 16'], inplace=True, errors='ignore')

# 3) Prepare features & target
X = data[numeric_cols]
y = data['CO(GT)']

# 4) Train/test split & scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5) Train RandomForest with tuned hyperparameters
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# 6) Evaluate
y_train_pred = model.predict(X_train_scaled)
y_test_pred  = model.predict(X_test_scaled)
train_r2 = r2_score(y_train, y_train_pred)
test_r2  = r2_score(y_test,  y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse  = mean_squared_error(y_test,  y_test_pred)
print(f"\nModel evaluation:")
print(f"  Train R²: {train_r2:.4f}, Train MSE: {train_mse:.2f}")
print(f"  Test  R²: {test_r2:.4f}, Test  MSE: {test_mse:.2f}")

# 7) Send SMS alert if live AQI exceeds threshold
account_sid = 'twilio_account_sid'  # Replace with your Twilio account SID
auth_token  = 'twilio_auth_token'  # Replace with your Twilio auth token    
client = Client(account_sid, auth_token)

THRESHOLD_AQI = 1  # set your AQI threshold
if live_aqi > THRESHOLD_AQI:
    msg = client.messages.create(
        body=f"⚠ Live AQI is {live_aqi} (threshold {THRESHOLD_AQI})",
        from_='‪+19xxxxxxxxxx‬',
        to='‪+91xxxxxxxxxx‬'
    )
    print(f"SMS alert sent: {msg.sid}")

    # 8) Use OpenAI API to generate guidelines for survival and precautions
    # Use OpenAI's new API format
# Set your OpenAI API key
client = openai.OpenAI(api_key="your_openai_api_key_here")
# Set OpenAI API key

# Prompt for ChatGPT
prompt = (
    f"The air quality index (AQI) is currently {live_aqi}, which is above the safe limit. "
    "Please provide simple and effective safety guidelines for the public during this high pollution level."
)

# Call the ChatGPT model
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an environmental safety advisor."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150,
    temperature=0.7,
)

guidelines = response.choices[0].message.content.strip()
print("\nGuidelines for surviving high pollution:")
print(guidelines)

# 9) Generate & save heatmap (using a fixed location for demonstration)
# Jitter logic for surrounding area
lat_center, lon_center = float(LAT), float(LON)
n = len(data)  # number of data points

lat_jitter = np.random.normal(loc=0, scale=0.02, size=n)  # ~2.2 km offset
lon_jitter = np.random.normal(loc=0, scale=0.02, size=n)

# Generate the heat data with jittered lat/lon
data['Latitude']  = lat_center + lat_jitter
data['Longitude'] = lon_center + lon_jitter
heat_data = data[['Latitude', 'Longitude', 'CO(GT)']].values.tolist()

# Create map centered around Chennai
m = folium.Map(location=[lat_center, lon_center], zoom_start=10)

# Add heatmap to the map
HeatMap(heat_data, radius=8, blur=12).add_to(m)

# Save the heatmap as HTML
m.save('air_quality_heatmap_chennai.html')
print("✅ Heatmap for Chennai saved as 'air_quality_heatmap_chennai.html'")