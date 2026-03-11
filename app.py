from flask import Flask, render_template, request
import numpy as np
import pickle
import requests
import config

# create flask app
app = Flask(__name__)

# load trained model
model = pickle.load(open("model.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))
crop_weather = pickle.load(open("crop_weather.pkl", "rb"))
def weather_fetch(city):

    api_key = config.weather_api_key

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

    response = requests.get(url)

    data = response.json()

    # check if API returned valid weather data
    if "main" not in data:
        print("API Error:", data)
        return None, None

    temperature = data["main"]["temp"] - 273.15
    humidity = data["main"]["humidity"]

    return temperature, humidity

# home page
@app.route("/")
def home():
    return render_template("index.html")
    
def fertilizer_recommendation(N, P, K, ph):

    problems = []
    suggestions = []

    # Nitrogen
    if N < 40:
        problems.append("Nitrogen level is low")
        suggestions.extend([
            "Apply Nitrogen fertilizers like Urea or Ammonium Nitrate.",
            "Add compost or well-decomposed manure.",
            "Grow nitrogen fixing crops like peas or beans."
        ])

    # Phosphorus
    if P < 40:
        problems.append("Phosphorus level is low")
        suggestions.extend([
            "Apply DAP or Single Super Phosphate fertilizer.",
            "Add bone meal or rock phosphate.",
            "Use phosphorus-rich organic compost."
        ])

    # Potassium
    if K < 40:
        problems.append("Potassium level is low")
        suggestions.extend([
            "Apply Muriate of Potash (MOP).",
            "Add wood ash to increase potassium.",
            "Use potassium sulfate fertilizer."
        ])

    # pH recommendations
    if ph < 5.5:
        problems.append("Soil is too acidic")
        suggestions.extend([
            "Apply agricultural lime to increase soil pH.",
            "Add wood ash to reduce soil acidity.",
            "Use organic compost to stabilize soil pH."
        ])

    elif ph > 7.5:
        problems.append("Soil is too alkaline")
        suggestions.extend([
            "Add elemental sulfur to reduce soil pH.",
            "Mix organic matter like compost or manure.",
            "Use ammonium sulfate fertilizers."
        ])

    # Balanced soil
    if not problems:
        problems.append("Soil nutrients and pH are balanced")
        suggestions.append("Maintain soil health using organic compost and crop rotation.")

    return problems, suggestions


# prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # get values from form
    N = float(request.form["nitrogen"])
    P = float(request.form["phosphorous"])
    K = float(request.form["potassium"])
    city = request.form["city"]

    temp, humidity = weather_fetch(city)

    if temp is None:
       return "City not found or API error"
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    # create input array
    data = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    # prediction
    prediction = model.predict(data)[0]
    
    avg_temp = crop_weather.loc[prediction, "temperature"]
    avg_humidity = crop_weather.loc[prediction, "humidity"]
   
    problems, fertilizer_tips = fertilizer_recommendation(N, P, K, ph)
   

    return render_template(
    "result.html",
    prediction=prediction,
    
    temperature=temp,
    humidity=humidity,
    avg_temp=avg_temp,
    avg_humidity=avg_humidity,
    N=N,
    P=P,
    K=K,
    ph=ph,
    rainfall=rainfall,
    city=city,
    accuracy=accuracy,
    problems=problems,
    fertilizer_tips=fertilizer_tips
    )
    


if __name__ == "__main__":
    app.run(debug=True)