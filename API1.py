import requests

url = "https://disease.sh/v3/covid-19/countries/india"
r = requests.get(url)
data = r.json()

print(data)

import pandas as pd

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])
print(df)

import matplotlib.pyplot as plt

labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
plt.show()
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate random historical data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

print(df_historical.head())

# Features and target variable
X = df_historical[["day"]]
y = df_historical["cases"]

# Standardize features (SVR performs better with scaled data)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train SVR model
model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)  # RBF kernel for non-linearity
model.fit(X_train, y_train)

# Predict next day's cases (Day 31)
next_day_scaled = scaler_X.transform([[31]])
predicted_cases_scaled = model.predict(next_day_scaled)

# Convert prediction back to original scale
predicted_cases = scaler_y.inverse_transform([[predicted_cases_scaled[0]]])[0][0]

print(f"Predicted cases for Day 31: {int(predicted_cases)}")
import streamlit as st

st.title("COVID-19 Cases Prediction-in INDIA")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User Input
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction = model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")s