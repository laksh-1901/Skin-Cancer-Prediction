import joblib
import pandas as pd

# Load model
model = joblib.load("models/model.pkl")

scaler = joblib.load("models/scaler.pkl")

columns = joblib.load("models/columns.pkl")

yes_no_cols = [
    'Family_History',
    'Itchiness',
    'Bleeding',
    'Asymmetry',
    'Border_Irregularity',
    'Color_Variation',
    'Evolution'
]


def get_user_input():

    user_data = {}

    for col in [
        "Age",
        "Mole_Count",
        "Diameter_mm"
    ]:

        user_data[col] = float(input(f"{col}: "))

    for col in yes_no_cols:

        user_data[col] = int(
            input(f"{col} (1=yes 0=no): ")
        )

    gender = input(
        "Gender (Male/Female): "
    ).capitalize()

    user_data["Gender_Male"] = 1 if gender == "Male" else 0

    skin = input(
        "Skin Type (Combination/Dry/Normal/Oily/Sensitive): "
    ).capitalize()

    for s in [
        "Combination",
        "Dry",
        "Normal",
        "Oily",
        "Sensitive"
    ]:

        user_data[f"Skin_Type_{s}"] = 1 if skin == s else 0

    sun = input(
        "Sun Exposure (High/Low/Moderate): "
    ).capitalize()

    for s in [
        "High",
        "Low",
        "Moderate"
    ]:

        user_data[f"Sun_Exposure_{s}"] = 1 if sun == s else 0

    return pd.DataFrame([user_data])


user = get_user_input()

user = user.reindex(
    columns=columns,
    fill_value=0
)

user_scaled = scaler.transform(user)

prediction = model.predict(user_scaled)[0]

print()

if prediction == 1:

    print("High Risk of Skin Cancer")

else:

    print("No Skin Cancer Detected")
