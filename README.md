# Lung Cancer Risk Prediction System

A machine learning-based command-line tool that predicts the risk of lung cancer based on patient symptoms and demographic information.

---

## 📌 Overview

Early detection of lung cancer significantly improves survival rates. This project uses a **Random Forest Classifier** trained on a dataset of 3,000 patient records to assess lung cancer risk based on 15 clinical features including age, gender, and common symptoms.

The system outputs a **risk probability score** and a **YES/NO prediction**, helping users understand whether they should seek further medical evaluation.

> ⚠️ **Disclaimer:** This tool is for educational and screening purposes only. It is **not a medical diagnosis**. Always consult a licensed healthcare professional.

---

## 📁 Project Structure

```
Lungs_Cancer_Prediction/
│
├── dataset.csv         # Patient dataset (3,000 records, 15 features)
├── train_model.py      # Model training script
├── cli.py              # Command-line prediction interface
├── model.pkl           # Saved trained model (Random Forest + feature columns)
└── README.md           # Project documentation
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | Lung cancer symptoms dataset |
| Total Records | 3,000 patients |
| Features | 15 (demographics + symptoms) |
| Target | `LUNG_CANCER` — YES / NO |
| Class Distribution | YES: 1,518 (50.6%) — NO: 1,482 (49.4%) |
| Missing Values | None |

### Features Used

| Feature | Type | Description |
|---|---|---|
| `GENDER` | Binary | 1 = Male, 0 = Female |
| `AGE` | Numeric | Patient age |
| `SMOKING` | Binary | Smoker (1 = Yes, 0 = No) |
| `YELLOW_FINGERS` | Binary | Yellow finger discoloration |
| `ANXIETY` | Binary | Anxiety symptoms |
| `PEER_PRESSURE` | Binary | Social/peer pressure |
| `CHRONIC_DISEASE` | Binary | Pre-existing chronic illness |
| `FATIGUE` | Binary | Persistent fatigue |
| `ALLERGY` | Binary | Allergic conditions |
| `WHEEZING` | Binary | Wheezing during breathing |
| `ALCOHOL_CONSUMING` | Binary | Regular alcohol use |
| `COUGHING` | Binary | Persistent cough |
| `SHORTNESS_OF_BREATH` | Binary | Breathing difficulty |
| `SWALLOWING_DIFFICULTY` | Binary | Difficulty swallowing |
| `CHEST_PAIN` | Binary | Chest pain or discomfort |

---

## 🤖 Model

| Property | Details |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Trees | 100 (`n_estimators=100`) |
| Class Weighting | `class_weight='balanced'` |
| Train / Test Split | 80% / 20% (stratified) |
| Random State | 42 (reproducible) |

### Model Performance

| Metric | Score |
|---|---|
| Accuracy | 53.5% |
| Precision (Cancer) | 0.54 |
| Recall (Cancer) | 0.54 |
| F1-Score (Cancer) | 0.54 |

> **Note on Accuracy:** The 53.5% accuracy reflects limitations in the dataset — specifically, `AGE` dominates feature importance (30%) while clinically critical features like `SMOKING` rank unexpectedly low. This suggests possible noise in the dataset. In a real-world deployment, dataset validation with medical professionals would be essential.

### Top Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | AGE | 30.3% |
| 2 | ANXIETY | ~5.5% |
| 3 | CHEST_PAIN | ~5.2% |
| 4 | ALLERGY | ~5.0% |
| 5 | PEER_PRESSURE | ~4.9% |

---

## ⚙️ Installation

### Prerequisites

- Python 3.7+
- pip

### Install Dependencies

```bash
pip install pandas scikit-learn joblib
```

---

## 🚀 Usage

### Step 1 — Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess `dataset.csv`
- Train the Random Forest model
- Save the model to `model.pkl`
- Print accuracy and classification report

**Sample output:**
```
Shape: (3000, 16)
Class Distribution:
 1    1518
 0    1482

Accuracy: 0.535

✅ Model trained successfully!
```

### Step 2 — Run Predictions

```bash
python cli.py
```

You will be prompted to enter patient information one field at a time:

```
🫁 Lung Cancer Prediction CLI

Gender (1 = Male, 0 = Female): 1
Age: 58
Smoking (0 = No, 1 = Yes): 1
Yellow Fingers (0 = No, 1 = Yes): 1
Anxiety (0 = No, 1 = Yes): 0
...

🔍 Result:
📊 Risk Probability: 0.73
⚠️  High Risk of Lung Cancer
👉 Consult a doctor.
```

---

## 🔍 How It Works

```
User Input (CLI)
      │
      ▼
Feature Encoding (Gender, Age, Binary Symptoms)
      │
      ▼
Load model.pkl (Random Forest + Feature Columns)
      │
      ▼
model.predict()        →  YES / NO
model.predict_proba()  →  Risk Probability (0.0 – 1.0)
      │
      ▼
Display Result + Recommendation
```

The model uses `predict_proba()` to return a continuous risk score rather than a hard binary prediction — this is medically more useful as it conveys the degree of risk, not just a category.

---

## 🛠️ Preprocessing Steps

1. **Column Cleaning** — Strip whitespace and convert to uppercase
2. **Gender Encoding** — `M → 1`, `F → 0`
3. **Symptom Encoding** — Symptom values originally stored as `1/2` (No/Yes) are converted to `0/1`
4. **Target Encoding** — `YES → 1`, `NO → 0`
5. **Stratified Split** — Maintains class ratio in both train and test sets

---

## 🔮 Future Improvements

- [ ] Add cross-validation (k-fold) for more reliable evaluation
- [ ] Tune hyperparameters with `GridSearchCV`
- [ ] Try XGBoost or LightGBM for potentially better performance
- [ ] Build a web interface using Streamlit or FastAPI
- [ ] Add SHAP explainability to show which symptoms influenced each prediction
- [ ] Validate dataset with a medical domain expert
- [ ] Containerize with Docker for easy deployment

---

## 🧰 Tech Stack

- **Language:** Python 3
- **ML Library:** scikit-learn
- **Data:** pandas
- **Model Persistence:** joblib
- **Interface:** Command Line (CLI)

---

## 👤 Author

**Sameer Sethar**
- Portfolio: [Click Here](https://sameersethar.me)
- GitHub: [Click here](https://github.com/sameersethar)
- LinkedIn: [Click Here](https://linkedin.com/in/sameer-sethar)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> *Built for educational purposes. Not intended for clinical use.*
