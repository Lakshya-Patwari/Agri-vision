import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("C:\\Users\\Lakshya\\Agri-vision\\data\\final_agriculture_dataset.csv")

print("Dataset loaded successfully")
print("Columns:", df.columns)

# -------------------------------
# CREATE PROFIT COLUMN
# -------------------------------
df["profit"] = df["revenue"] - df["total_cost"]

# -------------------------------
# ENCODE SEASON
# -------------------------------
season_mapping = {
    "Kharif": 0,
    "Rabi": 1,
    "Summer": 2
}

df["season_encoded"] = df["season"].map(season_mapping)

# -------------------------------
# FEATURES (MATCH FLASK INPUT)
# -------------------------------
features = ['N','P','K','temperature','humidity','ph','rainfall','season_encoded']

X = df[features]

# -------------------------------
# TARGETS
# -------------------------------
y_crop = df["crop_name"]
y_profit = df["profit"]
y_yield = df["yield_per_acre"]

# cost + revenue targets
cost_columns = [
    'seed_cost', 'fertilizer_cost', 'manure_cost',
    'pesticide_cost', 'irrigation_cost',
    'machinery_cost', 'misc_cost',
    'total_cost', 'revenue'
]

Y_costs = df[cost_columns]

# -------------------------------
# TRAIN TEST SPLIT (for crop)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_crop, test_size=0.2, random_state=42
)

# -------------------------------
# MODELS
# -------------------------------

# 🌾 Crop Model
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train, y_train)

# 💰 Profit Model
profit_model = RandomForestRegressor(n_estimators=100, random_state=42)
profit_model.fit(X, y_profit)

# 📈 Yield Model
yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
yield_model.fit(X, y_yield)

# 💸 Cost + Revenue Model
cost_model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, random_state=42)
)
cost_model.fit(X, Y_costs)

# -------------------------------
# CROP WEATHER (FOR UI)
# -------------------------------
crop_weather = df.groupby("crop_name")[["temperature", "humidity"]].mean()
# crop → season mapping
crop_season = df.groupby("crop_name")["season"].first()

pickle.dump(crop_season, open("crop_season.pkl", "wb"))
# -------------------------------
# SAVE ALL FILES
# -------------------------------
pickle.dump(crop_model, open("crop_model.pkl", "wb"))
pickle.dump(profit_model, open("profit_model.pkl", "wb"))
pickle.dump(yield_model, open("yield_model.pkl", "wb"))
pickle.dump(cost_model, open("cost_model.pkl", "wb"))
pickle.dump(crop_weather, open("crop_weather.pkl", "wb"))

print("✅ All models trained & saved successfully!")
