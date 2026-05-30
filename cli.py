import joblib
import train_model
# Load model
model, columns = joblib.load("model.pkl")

def get_binary(prompt):
    while True:
        try:
            val = int(input(f"{prompt} (0 = No, 1 = Yes): "))
            if val in [0, 1]:
                return val
        except:
            pass
        print("❌ Enter 0 or 1 only.")

def get_gender():
    while True:
        try:
            val = int(input("Gender (1 = Male, 0 = Female): "))
            if val in [0, 1]:
                return val
        except:
            pass
        print("❌ Invalid input.")

def get_age():
    while True:
        try:
            val = int(input("Age: "))
            if 1 <= val <= 120:
                return val
        except:
            pass
        print("❌ Enter valid age.")

def main():
    print("\n🫁 Lung Cancer Prediction CLI\n")

    data = {
        'GENDER': get_gender(),
        'AGE': get_age(),
        'SMOKING': get_binary("Smoking"),
        'YELLOW_FINGERS': get_binary("Yellow Fingers"),
        'ANXIETY': get_binary("Anxiety"),
        'PEER_PRESSURE': get_binary("Peer Pressure"),
        'CHRONIC_DISEASE': get_binary("Chronic Disease"),
        'FATIGUE': get_binary("Fatigue"),
        'ALLERGY': get_binary("Allergy"),
        'WHEEZING': get_binary("Wheezing"),
        'ALCOHOL_CONSUMING': get_binary("Alcohol Consuming"),
        'COUGHING': get_binary("Coughing"),
        'SHORTNESS_OF_BREATH': get_binary("Shortness of Breath"),
        'SWALLOWING_DIFFICULTY': get_binary("Difficulty Swallowing"),
        'CHEST_PAIN': get_binary("Chest Pain")
    }

    # Ensure correct order
    sample = [[data[col] for col in columns]]

    # Prediction
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    print("\n🔍 Result:")
    print(f"📊 Risk Probability: {prob:.2f}")

    if pred == 1:
        print("⚠️ High Risk of Lung Cancer")
        print("👉 Consult a doctor.")
    else:
        print("✅ Low Risk of Lung Cancer")
        print("👉 Stay healthy.")



if __name__ == "__main__":
    main()