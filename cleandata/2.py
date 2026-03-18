import pandas as pd

df1 = pd.read_csv("C:\\Users\\Lakshya\\Agri-vision\\cleandata\\clean_agriculture_dataset.csv")   # cost dataset
df2 = pd.read_csv("C:\\Users\\Lakshya\\Agri-vision\\cleandata\\crop_recommendation.csv")  # soil dataset
df1["crop_name"] = df1["crop_name"].str.lower()
df2["label"] = df2["label"].str.lower()

common_crops = [
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

df1 = df1[df1["crop_name"].isin(common_crops)]
df2 = df2[df2["label"].isin(common_crops)]

df2 = df2.rename(columns={"label": "crop_name"})

merged_df = pd.merge(df1, df2, on="crop_name", how="inner")
season_map = {
    "rice": "kharif",
    "maize": "kharif",
    "chickpea": "rabi",
    "mungbean": "kharif",
    "blackgram": "kharif",
    "lentil": "rabi",
    "pigeonpeas": "kharif",
    "cotton": "kharif",
    "jute": "kharif",
    "kidneybeans": "rabi",
    "coffee": "perennial"
}

merged_df["season"] = merged_df["crop_name"].map(season_map)

merged_df.to_csv("final_agriculture_dataset.csv", index=False)