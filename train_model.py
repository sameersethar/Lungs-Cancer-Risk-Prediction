import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.upper()

# ✅ Convert Gender
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

# ✅ Convert ALL 1/2 → 0/1 (VERY IMPORTANT)
for col in df.columns:
    if col not in ['GENDER', 'AGE', 'LUNG_CANCER']:
        df[col] = df[col].replace({1: 0, 2: 1})

# ✅ Convert target
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Check data
print("Shape:", df.shape)
print("Class Distribution:\n", df['LUNG_CANCER'].value_counts())

# Features & target
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy=accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nReport:\n", classification_report(y_test, y_pred))

# Save model + columns
joblib.dump((model, X.columns.tolist()), "model.pkl")

print("\n✅ Model trained successfully!")