import pandas as pd

# Load dataset
df = pd.read_csv("C:\\Users\\Lakshya\\Agri-vision\\cleandata\\IPL AUCTION E-CELL - AUCTION RECORD.csv")

# Sort by PLAYER RATING (highest rating first)
sorted_df = df.sort_values(by="PLAYER RATING", ascending=False)

# Display result
print(sorted_df)

# Optional: save the sorted dataset
sorted_df.to_csv("C:\\Users\\Lakshya\\Agri-vision\\cleandata\\sorted_by_rating.csv", index=False)