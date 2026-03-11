import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

# average weather per crop
crop_weather = df.groupby("label")[["temperature", "humidity"]].mean()


# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# calculate accuracy
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

# save model
pickle.dump(model, open("model.pkl", "wb"))

# save accuracy
pickle.dump(accuracy, open("accuracy.pkl", "wb"))

pickle.dump(crop_weather, open("crop_weather.pkl", "wb"))