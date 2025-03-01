import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

url = "https://disease.sh/v3/covid-19/countries/india"
r = requests.get(url)
data = r.json()

if data:
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

    df = pd.DataFrame([covid_data])
    print(df)

    labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
    values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("COVID-19 Data for India")
    st.pyplot(plt)

    np.random.seed(42)
    historical_cases = np.random.randint(30000, 70000, size=30)
    historical_deaths = np.random.randint(500, 2000, size=30)

    df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
    df_historical["day"] = range(1, 31)

    print(df_historical.head())

    X = df_historical[["day"]]
    y = df_historical["cases"]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)

    st.title("COVID-19 Cases Prediction in India")
    st.write("Predicting COVID-19 cases for a given day based on generated historical data.")

    day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

    if st.button("Predict"):
        next_day_scaled = scaler_X.transform([[day_input]])
        predicted_cases_scaled = model.predict(next_day_scaled)

        predicted_cases = scaler_y.inverse_transform([[predicted_cases_scaled[0]]])[0][0]

        st.write(f"Predicted cases for day {day_input}: {int(predicted_cases)}")
else:
    st.write("Failed to fetch COVID-19 data for India.")
