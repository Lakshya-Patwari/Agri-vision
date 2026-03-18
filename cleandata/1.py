import pandas as pd

# Load dataset
df = pd.read_csv("C:\\Users\\Lakshya\\Agri-vision\\cleandata\\cost-of-cultivation.csv")

# Standardize crop names
df["crop_name"] = df["crop_name"].str.lower()

# Rename crops to match recommendation dataset
crop_mapping = {
    "paddy": "rice",
    "gram": "chickpea",
    "moong": "mungbean",
    "urad": "blackgram",
    "masur": "lentil",
    "lentil": "lentil",
    "tur (arhar)": "pigeonpeas",
    "arhar": "pigeonpeas",
    "soyabean": "kidneybeans",
    "cotton": "coffee"
}

df["crop_name"] = df["crop_name"].replace(crop_mapping)

# Crops we want
selected_crops = [
    "rice",
    "maize",
    "chickpea",
    "mungbean",
    "blackgram",
    "lentil",
    "pigeonpeas",
    "cotton",
    "jute",
    "kidneybeans",
    "coffee"
]

# Filter dataset
df = df[df["crop_name"].isin(selected_crops)]

# Keep only useful columns
columns_needed = [
    "state_name",
    "crop_name",
    "derived_yield",
    "opr_cost_seed",
    "opr_cost_fertilizer",
    "opr_cost_manure",
    "opr_cost_insecticides",
    "opr_cost_irrigation_chrg",
    "opr_cost_misc",
    "fix_cost",
    "main_product_value"
]

df = df[columns_needed]

# Rename columns to simpler names
df = df.rename(columns={
    "opr_cost_seed": "seed_cost",
    "opr_cost_fertilizer": "fertilizer_cost",
    "opr_cost_manure": "manure_cost",
    "opr_cost_insecticides": "pesticide_cost",
    "opr_cost_irrigation_chrg": "irrigation_cost",
    "opr_cost_misc": "misc_cost",
    "fix_cost": "machinery_cost",
    "derived_yield": "yield_per_hectare",
    "main_product_value": "revenue"
})

# Calculate total cultivation cost
df["total_cost"] = (
    df["seed_cost"] +
    df["fertilizer_cost"] +
    df["manure_cost"] +
    df["pesticide_cost"] +
    df["irrigation_cost"] +
    df["misc_cost"] +
    df["machinery_cost"]
)

# Sort crops together
df = df.sort_values(by="crop_name")
df["yield_per_acre"] = df["yield_per_hectare"] / 2.47105
df.drop(columns=["yield_per_hectare"], inplace=True)

# Save clean dataset
df.to_csv("clean_agriculture_dataset.csv", index=False)

print("Dataset cleaned and saved as clean_agriculture_dataset.csv")