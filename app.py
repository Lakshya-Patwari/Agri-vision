from flask import Flask, render_template, request
import numpy as np
import pickle
import requests
import config
from datetime import datetime

def get_season():
    month = datetime.now().month

    if month in [6, 7, 8, 9]:   # June–Sept
        return "Kharif"
    elif month in [10, 11, 12, 1]:  # Oct–Jan
        return "Rabi"
    else:  # Feb–May
        return "Summer"

# -------------------------------
# CREATE APP
# -------------------------------
app = Flask(__name__)

# -------------------------------
# LOAD MODELS
# -------------------------------
crop_model = pickle.load(open("crop_model.pkl", "rb"))
profit_model = pickle.load(open("profit_model.pkl", "rb"))
yield_model = pickle.load(open("yield_model.pkl", "rb"))
cost_model = pickle.load(open("cost_model.pkl", "rb"))
crop_season = pickle.load(open("crop_season.pkl", "rb"))

season_map = {
    "Kharif": 0,
    "Rabi": 1,
    "Summer": 2
}

season = get_season()
season_encoded = season_map[season]

# (optional old files)
try:
    accuracy = pickle.load(open("accuracy.pkl", "rb"))
except:
    accuracy = 0

try:
    crop_weather = pickle.load(open("crop_weather.pkl", "rb"))
except:
    crop_weather = None


# -------------------------------
# WEATHER FUNCTION
# -------------------------------
def weather_fetch(city):
    api_key = config.weather_api_key

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

    try:
        response = requests.get(url, timeout=5)
        data = response.json()
    except:
        return None, None

    if "main" not in data:
        return None, None

    temperature = data["main"]["temp"] - 273.15
    humidity = data["main"]["humidity"]

    return temperature, humidity


# -------------------------------
# HOME PAGE
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# FERTILIZER RECOMMENDATION
# -------------------------------
def fertilizer_recommendation(N, P, K, ph):

    problems = []
    suggestions = []

    if N < 40:
        problems.append("Nitrogen level is low")
        suggestions.extend([
            "Apply Urea or Ammonium Nitrate",
            "Add compost or manure",
            "Grow legumes like peas/beans"
        ])

    if P < 40:
        problems.append("Phosphorus level is low")
        suggestions.extend([
            "Use DAP or SSP fertilizer",
            "Add bone meal",
            "Use organic compost"
        ])

    if K < 40:
        problems.append("Potassium level is low")
        suggestions.extend([
            "Apply MOP fertilizer",
            "Add wood ash",
            "Use potassium sulfate"
        ])

    if ph < 5.5:
        problems.append("Soil is too acidic")
        suggestions.extend([
            "Apply lime",
            "Add wood ash",
            "Use compost"
        ])

    elif ph > 7.5:
        problems.append("Soil is too alkaline")
        suggestions.extend([
            "Add sulfur",
            "Use organic matter",
            "Apply ammonium fertilizers"
        ])

    if not problems:
        problems.append("Soil is balanced")
        suggestions.append("Maintain with compost and crop rotation")

    return problems, suggestions


# -------------------------------
# PREDICT ROUTE
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # get inputs
    N = float(request.form["nitrogen"])
    P = float(request.form["phosphorous"])
    K = float(request.form["potassium"])
    city = request.form["city"]
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])
    
    season = get_season()
    

    # fetch weather
    temp, humidity = weather_fetch(city)
   
   

    if temp is None:
        return render_template(
            "result.html",
            prediction="Invalid City",

            predicted_profit=0,
            predicted_yield=0,

            temperature=0,
            humidity=0,
            avg_temp=0,
            avg_humidity=0,

            N=N, P=P, K=K, ph=ph,
            rainfall=rainfall,
            city=city,

            accuracy=accuracy,
            problems=["Invalid city entered"],
            fertilizer_tips=["Check spelling or try another city"]
        )

    # prepare input
    data = np.array([[N, P, K, temp, humidity, ph, rainfall, season_encoded]])

    # -------------------------------
    # PREDICTIONS
    # -------------------------------
    prediction = crop_model.predict(data)[0]
    predicted_profit = profit_model.predict(data)[0]
    predicted_yield = yield_model.predict(data)[0]
    predicted_yield_kg = predicted_yield * 1000

    # -------------------------------
    # WEATHER COMPARISON
    # -------------------------------
    if crop_weather is not None and prediction in crop_weather.index:
        avg_temp = crop_weather.loc[prediction, "temperature"]
        avg_humidity = crop_weather.loc[prediction, "humidity"]
    else:
        avg_temp = 0
        avg_humidity = 0
    if prediction in crop_season.index:
        recommended_season = crop_season[prediction]
    else:
        recommended_season = "Unknown"

    # -------------------------------
    # SOIL ANALYSIS
    # -------------------------------
    problems, fertilizer_tips = fertilizer_recommendation(N, P, K, ph)

    # -------------------------------
    # RENDER RESULT
    # -------------------------------
    costs = cost_model.predict(data)[0]

    seed_cost = costs[0]
    fertilizer_cost = costs[1]
    manure_cost = costs[2]
    pesticide_cost = costs[3]
    irrigation_cost = costs[4]
    machinery_cost = costs[5]
    misc_cost = costs[6]
    total_cost = costs[7]
    predicted_revenue = costs[8]
    return render_template(
        "result.html",
        prediction=prediction,

        predicted_profit=predicted_profit,
        predicted_yield=predicted_yield_kg,

        temperature=temp,
        humidity=humidity,
        avg_temp=avg_temp,
        avg_humidity=avg_humidity,

        N=N, P=P, K=K, ph=ph,
        rainfall=rainfall,
        city=city,

        accuracy=accuracy,
        problems=problems,
        fertilizer_tips=fertilizer_tips,
        seed_cost=seed_cost,
        fertilizer_cost=fertilizer_cost,
        manure_cost=manure_cost,
        pesticide_cost=pesticide_cost,
        irrigation_cost=irrigation_cost,
        machinery_cost=machinery_cost,
        misc_cost=misc_cost,
        total_cost=total_cost,
        predicted_revenue=predicted_revenue,
        recommended_season=recommended_season,
        season=season,

)


# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)