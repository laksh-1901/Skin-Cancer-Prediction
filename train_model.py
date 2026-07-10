import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("dataset/skin_cancer_dataset.csv")

# Convert Yes/No columns
yes_no_cols = [
    'Family_History',
    'Itchiness',
    'Bleeding',
    'Asymmetry',
    'Border_Irregularity',
    'Color_Variation',
    'Evolution'
]

df[yes_no_cols] = df[yes_no_cols].replace({
    "Yes": 1,
    "No": 0
})

# Remove missing values
df.dropna(inplace=True)

# One-hot encoding
cat_cols = [
    "Gender",
    "Skin_Type",
    "Sun_Exposure"
]

df = pd.get_dummies(
    df,
    columns=cat_cols,
    drop_first=True
)

# Encode target
le = LabelEncoder()

df["target"] = le.fit_transform(df["Skin_Cancer"])

X = df.drop(["Skin_Cancer", "target"], axis=1)

y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Balance dataset
smote = SMOTE(random_state=42)

X_train_bal, y_train_bal = smote.fit_resample(
    X_train,
    y_train
)

# Standardize
scaler = StandardScaler()

X_train_bal = scaler.fit_transform(X_train_bal)

X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=500)

model.fit(
    X_train_bal,
    y_train_bal
)

# Evaluate
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print(f"Accuracy : {accuracy*100:.2f}%")

# Save model
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/columns.pkl")

print("Model Saved Successfully.")
